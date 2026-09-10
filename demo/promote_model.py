"""Phase 7 - Promotion gate: promote the dev model to PROD (service account only).

Runs as ML_DEPLOY_SVC. In CI this is the GitHub OIDC service user SVC_ML_DEPLOY
(keyless / Workload Identity Federation). A data scientist (ML_DEV_ROLE) CANNOT
run this against prod - only the pipeline can.

Steps (all in prod):
  1. Build the prod ABT with the SAME transform used in dev.
  2. Register the prod feature store views (ACCOUNT_PROFILE + ACCOUNT_RISK).
  3. Promote the dev model artifact into the PROD registry (load from dev, log to
     prod) and set the prod default version.
  4. Batch inference: create/refresh a scheduled TASK that scores recent prod
     events into ML_FRAUD_PRODUCTION.ANALYTICS.PREDICTIONS (native SQL inference).

Local dry-run (as an admin holding ML_DEPLOY_SVC):
  SNOWFLAKE_CONNECTION_NAME=<your-connection> .venv/bin/python mlops/promote_model.py --dev-version V2
In CI the role/identity is provided by OIDC; pass --dev-version (or use default).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from snowpark_session import create_snowpark_session
from transforms.base_features import build_abt

from snowflake.ml.feature_store import FeatureStore, Entity, FeatureView, CreationMode
from snowflake.ml.registry import Registry

REQUEST_CTX = {"AMOUNT_PAID", "IS_CROSS_CURRENCY", "IS_CROSS_BORDER",
               "IS_HIGH_RISK_FORMAT", "AMOUNT_TO_AVG_RATIO"}
PROFILE_HIST = {"HIST_TXN_COUNT", "HIST_AVG_AMOUNT", "HIST_STD_AMOUNT", "HIST_MAX_AMOUNT",
                "HIST_DISTINCT_RECEIVERS", "HIST_DISTINCT_RECEIVER_BANKS",
                "HIST_DISTINCT_COUNTRIES", "HIST_FOREIGN_CCY_SHARE", "HIST_HIGH_RISK_SHARE"}


def register_prod_fs(fs, session):
    account = Entity(name=C.ENTITY_NAME, join_keys=[C.ENTITY_JOIN_KEY])
    fs.register_entity(account)
    profile_df = session.sql(f"""
        SELECT {C.ENTITY_JOIN_KEY}, HIST_TXN_COUNT, HIST_AVG_AMOUNT, HIST_STD_AMOUNT,
               HIST_MAX_AMOUNT, HIST_DISTINCT_RECEIVERS, HIST_DISTINCT_RECEIVER_BANKS,
               HIST_DISTINCT_COUNTRIES, HIST_FOREIGN_CCY_SHARE, HIST_HIGH_RISK_SHARE
        FROM {C.TBL_ACCOUNT_HISTORY}""")
    fs.register_feature_view(FeatureView(name=C.FV_PROFILE, entities=[account],
        feature_df=profile_df, refresh_freq="1 day", desc="prod profile"),
        version=C.FV_VERSION, overwrite=True)
    risk_df = session.sql(f"""
        SELECT {C.ENTITY_JOIN_KEY},
               HIST_STD_AMOUNT / NULLIF(HIST_AVG_AMOUNT,0) AS HIST_AMOUNT_CV,
               HIST_DISTINCT_RECEIVERS / NULLIF(HIST_TXN_COUNT,0) AS HIST_RECEIVER_FANOUT
        FROM {C.TBL_ACCOUNT_HISTORY}""")
    fs.register_feature_view(FeatureView(name="ACCOUNT_RISK", entities=[account],
        feature_df=risk_df, refresh_freq="1 day", desc="prod risk signals"),
        version=C.FV_VERSION, overwrite=True)
    print("  prod feature views registered: ACCOUNT_PROFILE, ACCOUNT_RISK")


def _prod_has_version(prod_reg, ver):
    try:
        m = prod_reg.get_model(C.MODEL_NAME)
        return any(str(v.version_name).upper() == ver.upper() for v in m.versions())
    except Exception:
        return False


def promote_model(session, dev_version):
    dev_reg = Registry(session=session, database_name=C.DEV_DATABASE, schema_name=C.REGISTRY_SCHEMA)
    prod_reg = Registry(session=session, database_name=C.PROD_DATABASE, schema_name=C.REGISTRY_SCHEMA)
    m = dev_reg.get_model(C.MODEL_NAME)
    ver = dev_version or str(m.default.version_name)

    if _prod_has_version(prod_reg, ver):
        print(f"  prod {C.MODEL_NAME}/{ver} already exists - skipping re-log (idempotent)")
    else:
        print(f"  promoting dev {C.MODEL_NAME}/{ver} -> prod registry")
        # Model load is owner-only: assume the owning dev role for the load, then
        # switch back to the deploy role to write prod.
        session.sql(f"USE ROLE {C.DEV_ROLE}").collect()
        mv = m.version(ver)
        loaded = mv.load(force=True)
        metrics = mv.show_metrics()
        signatures = {f["target_method"]: f["signature"] for f in mv.show_functions()}
        session.sql(f"USE ROLE {C.DEPLOY_ROLE}").collect()
        prod_reg.log_model(
            model=loaded, model_name=C.MODEL_NAME, version_name=ver,
            signatures=signatures, metrics=metrics,
            comment=f"Promoted from dev {C.MODEL_NAME}/{ver} via CI (service account).")

    session.sql(f"USE ROLE {C.DEPLOY_ROLE}").collect()
    prod_reg.get_model(C.MODEL_NAME).default = ver
    prod_mv = prod_reg.get_model(C.MODEL_NAME).version(ver)
    fn = next(f for f in prod_mv.show_functions() if f["target_method"].lower() == "predict_proba")
    sig_names = [str(s.name).upper() for s in fn["signature"].inputs]
    return ver, sig_names


def feature_expr(name):
    if name.startswith("PAYMENT_FORMAT_"):
        suf = name[len("PAYMENT_FORMAT_"):]
        return f"IFF(UPPER(REPLACE(s.PAYMENT_FORMAT,' ','_'))='{suf}',1,0)"
    if name in REQUEST_CTX:
        return f"COALESCE(s.{name},0)"
    if name == "HIST_AMOUNT_CV":
        return "COALESCE(h.HIST_STD_AMOUNT/NULLIF(h.HIST_AVG_AMOUNT,0),0)"
    if name == "HIST_RECEIVER_FANOUT":
        return "COALESCE(h.HIST_DISTINCT_RECEIVERS/NULLIF(h.HIST_TXN_COUNT,0),0)"
    if name in PROFILE_HIST:
        return f"COALESCE(h.{name},0)"
    return "0"


def batch_score(session, sig_names):
    model_fqn = f"{C.PROD_DATABASE}.{C.REGISTRY_SCHEMA}.{C.MODEL_NAME}"
    abt = C.abt_fqn("prod")
    hist = C.TBL_ACCOUNT_HISTORY
    args = ", ".join(feature_expr(n) for n in sig_names)
    # Score the most recent day of prod events (a daily batch slice).
    scoring_sql = f"""
        CREATE OR REPLACE TABLE {C.PREDICTIONS} AS
        WITH scoring AS (
            SELECT s.{C.ENTITY_JOIN_KEY}, s.EVENT_TS, s.PAYMENT_FORMAT, s.IS_LAUNDERING AS LABEL,
                   {model_fqn}!PREDICT_PROBA({args}) AS PRED
            FROM {abt} s
            LEFT JOIN {hist} h ON s.{C.ENTITY_JOIN_KEY} = h.{C.ENTITY_JOIN_KEY}
            WHERE s.EVENT_TS >= (SELECT DATEADD(day,-1,MAX(EVENT_TS)) FROM {abt})
        )
        SELECT {C.ENTITY_JOIN_KEY}, EVENT_TS, PAYMENT_FORMAT, LABEL,
               PRED:output_feature_1::FLOAT AS FRAUD_SCORE,
               CURRENT_TIMESTAMP() AS SCORED_AT
        FROM scoring
    """
    session.sql(scoring_sql).collect()
    n = session.sql(f"SELECT COUNT(*) C FROM {C.PREDICTIONS}").collect()[0]["C"]
    print(f"  PREDICTIONS populated: {n:,} rows")

    # Recurring daily batch task with the same statement.
    session.sql(f"""
        CREATE OR REPLACE TASK {C.BATCH_TASK}
          WAREHOUSE = {C.WAREHOUSE}
          SCHEDULE = 'USING CRON 0 6 * * * UTC'
          AS {scoring_sql}""").collect()
    session.sql(f"ALTER TASK {C.BATCH_TASK} SUSPEND").collect()  # created suspended
    print(f"  batch task created (suspended): {C.BATCH_TASK}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-version", default=None, help="dev model version to promote (default: dev default)")
    args = ap.parse_args()

    session = create_snowpark_session()
    session.sql(f"USE ROLE {C.DEPLOY_ROLE}").collect()  # service account role
    # Enable secondary roles so the deploy role can load the dev-owned model
    # (model load is owner-only; ML_DEPLOY_SVC inherits ML_DEV_ROLE).
    try:
        session.sql("USE SECONDARY ROLES ALL").collect()
    except Exception:
        pass
    session.sql(f"USE WAREHOUSE {C.WAREHOUSE}").collect()
    session.sql(f"USE SCHEMA {C.PROD_DATABASE}.{C.FEATURE_STORE_SCHEMA}").collect()
    print(f"PROMOTION as {C.DEPLOY_ROLE} -> {C.PROD_DATABASE}")

    print("1) Build prod ABT (same transform as dev)")
    build_abt(session, "prod")

    print("2) Register prod feature store")
    fs = FeatureStore(session=session, database=C.PROD_DATABASE, name=C.FEATURE_STORE_SCHEMA,
                      default_warehouse=C.WAREHOUSE, creation_mode=CreationMode.CREATE_IF_NOT_EXIST)
    register_prod_fs(fs, session)

    print("3) Promote model dev -> prod registry")
    ver, sig_names = promote_model(session, args.dev_version)

    print("4) Batch inference (native SQL) -> PREDICTIONS + scheduled task")
    batch_score(session, sig_names)

    print(f"\nPromotion complete: {C.MODEL_NAME}/{ver} live in prod; batch scoring ready.")
    session.close()


if __name__ == "__main__":
    main()
