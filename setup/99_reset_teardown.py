"""Teardown / reset for the RT Feature Store fraud-detection demo.

Removes EVERYTHING this demo created, in dependency order. The priority is the
Postgres-backed Online Feature Store service, which runs 24/7 (HA) and accrues
cost even when idle -- it is dropped FIRST.

The shared compute pool (MLOPS_CPU_M_POOL) is intentionally NOT dropped: it is a
pre-existing pool this demo only borrows. We just drop our service off it.

Safety: this is destructive. It runs in DRY-RUN mode by default and only tears
down when you pass --yes.

Usage:
  SNOWFLAKE_CONNECTION_NAME=demo156_keypair .venv/bin/python setup/99_reset_teardown.py         # dry run
  SNOWFLAKE_CONNECTION_NAME=demo156_keypair .venv/bin/python setup/99_reset_teardown.py --yes   # execute
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from snowpark_session import create_snowpark_session

# Objects created outside config.py (ad hoc, during the notebook/EAI work).
EAI_NAME = "RT_FS_DEMO_EAI"
NETWORK_RULE = f"{C.DATABASE}.{C.FEATURE_STORE_SCHEMA}.RT_FS_EGRESS"
SECRET_NAME = f"{C.DATABASE}.{C.FEATURE_STORE_SCHEMA}.DEMO_PAT"

DRY = True  # flipped by --yes


def step(desc, fn):
    """Run one best-effort teardown step; never abort the whole reset on error."""
    if DRY:
        print(f"  [dry-run] would: {desc}")
        return
    try:
        fn()
        print(f"  [ok]  {desc}")
    except Exception as e:
        print(f"  [skip] {desc}  ->  {type(e).__name__}: {str(e)[:160]}")


def sql(session, statement):
    return lambda: session.sql(statement).collect()


def main():
    global DRY
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="actually execute the teardown")
    ap.add_argument("--drop-pool", action="store_true",
                    help="ALSO drop the shared compute pool (default: leave it)")
    args = ap.parse_args()
    DRY = not args.yes

    print("=" * 72)
    print("RT FEATURE STORE FRAUD DEMO -- TEARDOWN", "(DRY RUN)" if DRY else "(EXECUTING)")
    print("=" * 72)

    session = create_snowpark_session()
    session.sql(f"USE WAREHOUSE {C.WAREHOUSE}").collect()

    # --- Phase 1: Feature Store objects, then the Postgres online service ------
    # drop_online_service() FAILS while feature views still exist ("active Feature
    # Views exist ... Remove all Feature Views first"), so tear those down first.
    print("\n[1] Feature Store objects + Postgres online service (THE cost item):")
    def teardown_fs():
        from snowflake.ml.feature_store import FeatureStore, CreationMode
        fs = FeatureStore(session=session, database=C.DATABASE,
                          name=C.FEATURE_STORE_SCHEMA, default_warehouse=C.WAREHOUSE,
                          creation_mode=CreationMode.FAIL_IF_NOT_EXIST)
        # 1) feature group, 2) feature views, 3) stream source, 4) entity
        for delete, obj in (
            (fs.delete_feature_group, lambda: fs.get_feature_group(C.FEATURE_GROUP, C.FV_VERSION.lower())),
        ):
            try: delete(obj())
            except Exception as e: print(f"      (feature group: {type(e).__name__})")
        for fv_name in (C.FV_PROFILE, C.FV_VELOCITY, C.FV_REALTIME):
            try: fs.delete_feature_view(fs.get_feature_view(fv_name, C.FV_VERSION))
            except Exception as e: print(f"      ({fv_name}: {type(e).__name__})")
        try: fs.delete_stream_source(C.STREAM_SOURCE)
        except Exception as e: print(f"      (stream source: {type(e).__name__})")
        try: fs.delete_entity(C.ENTITY_NAME)
        except Exception as e: print(f"      (entity: {type(e).__name__})")
        # 5) now the online service can be dropped -> stops Postgres HA
        fs.drop_online_service()
    step("delete FVs/feature group/stream/entity, then fs.drop_online_service()", teardown_fs)

    # --- Phase 2: SPCS real-time inference service -----------------------------
    print("\n[2] SPCS inference service:")
    step(f"DROP SERVICE {C.DATABASE}.{C.FEATURE_STORE_SCHEMA}.{C.INFERENCE_SERVICE}",
         sql(session, f"DROP SERVICE IF EXISTS {C.DATABASE}.{C.FEATURE_STORE_SCHEMA}.{C.INFERENCE_SERVICE}"))

    # --- Phase 3: registered model (all versions) ------------------------------
    print("\n[3] Model registry:")
    step(f"DROP MODEL {C.DATABASE}.{C.FEATURE_STORE_SCHEMA}.{C.MODEL_NAME}",
         sql(session, f"DROP MODEL IF EXISTS {C.DATABASE}.{C.FEATURE_STORE_SCHEMA}.{C.MODEL_NAME}"))

    # --- Phase 4: account-level objects (need ACCOUNTADMIN) --------------------
    print("\n[4] External access integration + feature-store roles (ACCOUNTADMIN):")
    step("USE ROLE ACCOUNTADMIN", sql(session, "USE ROLE ACCOUNTADMIN"))
    # EAI first: it references the secret, so it must go before the DB (secret) drop.
    step(f"DROP EXTERNAL ACCESS INTEGRATION {EAI_NAME}",
         sql(session, f"DROP EXTERNAL ACCESS INTEGRATION IF EXISTS {EAI_NAME}"))
    step(f"DROP ROLE {C.FS_PRODUCER_ROLE}",
         sql(session, f"DROP ROLE IF EXISTS {C.FS_PRODUCER_ROLE}"))
    step(f"DROP ROLE {C.FS_CONSUMER_ROLE}",
         sql(session, f"DROP ROLE IF EXISTS {C.FS_CONSUMER_ROLE}"))
    step("USE ROLE SYSADMIN", sql(session, "USE ROLE SYSADMIN"))

    # --- Phase 5: database (sweeps schemas/tables/DTs/stream/secret/net rule) ---
    print("\n[5] Database (cascades: RAW, CURATED, FEATURE_STORE, tables, dynamic tables,")
    print("    stream source, feature views, secret DEMO_PAT, network rule RT_FS_EGRESS):")
    step(f"DROP DATABASE {C.DATABASE} CASCADE",
         sql(session, f"DROP DATABASE IF EXISTS {C.DATABASE} CASCADE"))

    # --- Phase 6: shared compute pool (left alone by default) ------------------
    print("\n[6] Compute pool:")
    if args.drop_pool:
        step(f"DROP COMPUTE POOL {C.INFERENCE_COMPUTE_POOL}",
             sql(session, f"DROP COMPUTE POOL IF EXISTS {C.INFERENCE_COMPUTE_POOL}"))
    else:
        print(f"  [keep] {C.INFERENCE_COMPUTE_POOL} is a shared/pre-existing pool -- NOT dropped.")
        print("         Our service was removed above so it can auto-suspend. Pass --drop-pool to remove it.")

    # --- Phase 7: safety net -- verify no orphaned Postgres HA instance remains -
    # The online service is backed by an account-level Postgres instance
    # (FS_RUNTIME_PG_*). If anything above left it behind, it keeps billing.
    print("\n[7] Safety net -- orphaned Postgres instance check:")
    def sweep_pg():
        rows = session.sql("SHOW POSTGRES INSTANCES").collect()
        leftovers = [r["name"] for r in rows if str(r["name"]).upper().startswith("FS_RUNTIME_PG")]
        if not leftovers:
            print(f"      SHOW POSTGRES INSTANCES: {len(rows)} total, no FS_RUNTIME_PG* orphans.")
            return
        for name in leftovers:
            print(f"      dropping orphaned Postgres instance: {name}")
            session.sql(f'DROP POSTGRES INSTANCE IF EXISTS "{name}"').collect()
    step("SHOW POSTGRES INSTANCES + drop any FS_RUNTIME_PG* orphan", sweep_pg)

    print("\n" + "=" * 72)
    if DRY:
        print("DRY RUN complete -- nothing was changed. Re-run with --yes to execute.")
    else:
        print("Teardown complete. The Postgres online service and all demo objects are gone.")
        print("Code stays in Git; re-run setup/01-04 + demo/register_realtime_fv.py to rebuild.")
    print("=" * 72)
    session.close()


if __name__ == "__main__":
    main()
