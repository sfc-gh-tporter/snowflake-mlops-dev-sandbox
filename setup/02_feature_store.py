"""Phase 4 - Env-aware feature store (BATCH default).

Registers the dev (or prod) feature store and an offline ACCOUNT_PROFILE feature
view sourced from prod ACCOUNT_HISTORY. Batch inference and training read these
features; there is NO Postgres online service here (no 24/7 HA cost).

The optional real-time/online path (Postgres online store + stream/real-time FVs
+ SPCS service) lives under setup/online/ and is run only on demand.

Run (dev):
  SNOWFLAKE_CONNECTION_NAME=<your-connection> ML_ENV=dev .venv/bin/python setup/02_feature_store.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from snowpark_session import create_snowpark_session

from snowflake.ml.feature_store import FeatureStore, Entity, FeatureView, CreationMode


def main():
    env = C.ML_ENV
    role = C.DEV_ROLE if env == "dev" else "ACCOUNTADMIN"
    session = create_snowpark_session()
    session.sql(f"USE ROLE {role}").collect()
    session.sql(f"USE WAREHOUSE {C.WAREHOUSE}").collect()
    session.sql(f"USE SCHEMA {C.DATABASE}.{C.FEATURE_STORE_SCHEMA}").collect()
    print(f"Feature store: {C.DATABASE}.{C.FEATURE_STORE_SCHEMA} (env={env}, role={role})")

    fs = FeatureStore(
        session=session, database=C.DATABASE, name=C.FEATURE_STORE_SCHEMA,
        default_warehouse=C.WAREHOUSE, creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
    )

    print("Register entity", C.ENTITY_NAME)
    account = Entity(name=C.ENTITY_NAME, join_keys=[C.ENTITY_JOIN_KEY],
                     desc="Sender account (composite BANK_ID_ACCOUNT_NUMBER)")
    fs.register_entity(account)

    # ACCOUNT_PROFILE: static per-account profile features (offline / batch).
    # Sourced from prod ACCOUNT_HISTORY; keyed by ACCOUNT_ID (no timestamp -> latest).
    print("Register ACCOUNT_PROFILE (batch/offline FV, sourced from prod)")
    profile_df = session.sql(f"""
        SELECT {C.ENTITY_JOIN_KEY},
               HIST_TXN_COUNT, HIST_AVG_AMOUNT, HIST_STD_AMOUNT, HIST_MAX_AMOUNT,
               HIST_DISTINCT_RECEIVERS, HIST_DISTINCT_RECEIVER_BANKS,
               HIST_DISTINCT_COUNTRIES, HIST_FOREIGN_CCY_SHARE, HIST_HIGH_RISK_SHARE
        FROM {C.TBL_ACCOUNT_HISTORY}
    """)
    profile_fv = FeatureView(
        name=C.FV_PROFILE, entities=[account], feature_df=profile_df,
        refresh_freq="1 day",
        desc="Per-account lifetime profile features (batch/offline).",
    )
    fs.register_feature_view(profile_fv, version=C.FV_VERSION, overwrite=True)
    print("  registered", C.FV_PROFILE, C.FV_VERSION)

    # Verify: source rows + read the FV back from the store.
    print(f"  ACCOUNT_PROFILE source rows: {profile_df.count():,}")
    got = fs.get_feature_view(C.FV_PROFILE, C.FV_VERSION)
    print(f"  get_feature_view OK: {got.name}/{got.version}")
    print("\nBatch feature store setup complete.")
    session.close()


if __name__ == "__main__":
    main()
