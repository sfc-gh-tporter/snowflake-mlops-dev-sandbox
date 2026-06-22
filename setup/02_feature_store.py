"""Setup 02 - Feature Store, online (Postgres) service, and feature views.

Creates:
  * Producer/consumer account roles + grants (RBAC) - run as ACCOUNTADMIN
  * FeatureStore in FRAUD_RT_DEMO.FEATURE_STORE
  * ACCOUNT entity (composite ACCOUNT_ID)
  * Postgres-backed online service (provisions in several minutes)
  * ACCOUNT_PROFILE      - batch online FV (slow-moving profile)
  * TRANSACTION_EVENTS   - stream source (schema for the Ingest API)
  * ACCOUNT_VELOCITY     - stream FV w/ CONTINUOUS time-windowed aggregations
  * FRAUD_FEATURES       - feature group (profile + velocity)

Idempotent: re-running skips online-service creation if already RUNNING and
re-registers feature views with overwrite=True.

Run: SNOWFLAKE_CONNECTION_NAME=<conn> .venv/bin/python setup/02_feature_store.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from snowpark_session import create_snowpark_session

from snowflake.ml.feature_store import (
    FeatureStore, Entity, CreationMode, FeatureView,
    OnlineConfig, OnlineStoreType, StreamSource, StreamConfig,
    Feature, FeatureGroup, online_service,
)
from snowflake.ml.feature_store.spec.enums import FeatureAggregationMethod
from snowflake.snowpark import functions as F
from snowflake.snowpark.types import (
    StructType, StructField, StringType, DoubleType, LongType,
    TimestampType, TimestampTimeZone,
)


def passthrough(df):
    """Stream transformation: events already carry the columns we aggregate."""
    return df


def run(session, sql, label=None):
    if label:
        print(f"  - {label}")
    return session.sql(sql).collect()


def setup_rbac(session):
    """Create producer/consumer roles + grants (needs ACCOUNTADMIN)."""
    print("RBAC (as ACCOUNTADMIN):")
    run(session, "USE ROLE ACCOUNTADMIN")
    db, sch = C.DATABASE, C.FEATURE_STORE_SCHEMA
    fqn = f"{db}.{sch}"
    p, c = C.FS_PRODUCER_ROLE, C.FS_CONSUMER_ROLE
    stmts = [
        f"CREATE SCHEMA IF NOT EXISTS {fqn}",
        f"CREATE ROLE IF NOT EXISTS {p}",
        f"CREATE ROLE IF NOT EXISTS {c}",
        f"GRANT ROLE {p} TO ROLE SYSADMIN",
        f"GRANT ROLE {c} TO ROLE {p}",
        # Producer
        f"GRANT CREATE DYNAMIC TABLE ON SCHEMA {fqn} TO ROLE {p}",
        f"GRANT CREATE VIEW ON SCHEMA {fqn} TO ROLE {p}",
        f"GRANT CREATE TAG ON SCHEMA {fqn} TO ROLE {p}",
        f"GRANT CREATE TABLE ON SCHEMA {fqn} TO ROLE {p}",
        f"GRANT CREATE DATASET ON SCHEMA {fqn} TO ROLE {p}",
        f"GRANT USAGE ON DATABASE {db} TO ROLE {p}",
        f"GRANT USAGE ON SCHEMA {fqn} TO ROLE {p}",
        f"GRANT USAGE ON WAREHOUSE {C.WAREHOUSE} TO ROLE {p}",
        # Consumer
        f"GRANT USAGE ON DATABASE {db} TO ROLE {c}",
        f"GRANT USAGE ON SCHEMA {fqn} TO ROLE {c}",
        f"GRANT SELECT, MONITOR ON FUTURE DYNAMIC TABLES IN SCHEMA {fqn} TO ROLE {c}",
        f"GRANT SELECT, MONITOR ON ALL DYNAMIC TABLES IN SCHEMA {fqn} TO ROLE {c}",
        f"GRANT SELECT, REFERENCES ON FUTURE VIEWS IN SCHEMA {fqn} TO ROLE {c}",
        f"GRANT SELECT, REFERENCES ON ALL VIEWS IN SCHEMA {fqn} TO ROLE {c}",
        f"GRANT USAGE ON WAREHOUSE {C.WAREHOUSE} TO ROLE {c}",
        # Producer must read the curated source tables
        f"GRANT USAGE ON SCHEMA {db}.{C.CURATED_SCHEMA} TO ROLE {p}",
        f"GRANT SELECT ON ALL TABLES IN SCHEMA {db}.{C.CURATED_SCHEMA} TO ROLE {p}",
        f"GRANT SELECT ON FUTURE TABLES IN SCHEMA {db}.{C.CURATED_SCHEMA} TO ROLE {p}",
    ]
    for s in stmts:
        run(session, s, s[:70])
    run(session, "USE ROLE SYSADMIN")
    print("  RBAC done; back to SYSADMIN")


def ensure_online_service(fs):
    print("Online service (Postgres):")
    try:
        status = fs.get_online_service_status()
        if getattr(status, "status", None) == "RUNNING":
            print("  already RUNNING - skipping creation")
            return status
    except Exception as e:
        print(f"  no existing service ({type(e).__name__}); creating")
    fs.create_online_service(C.FS_PRODUCER_ROLE, C.FS_CONSUMER_ROLE)
    print("  provisioning (polling until RUNNING, up to ~10 min)...")
    status = fs.get_online_service_status()
    waited = 0
    while getattr(status, "status", None) != "RUNNING":
        time.sleep(30)
        waited += 30
        status = fs.get_online_service_status()
        print(f"    [{waited}s] status={getattr(status,'status',None)}")
        if waited > 900:
            raise TimeoutError("Online service did not reach RUNNING in 15 min")
    print("  RUNNING")
    return status


def main():
    session = create_snowpark_session()
    setup_rbac(session)
    run(session, f"USE WAREHOUSE {C.WAREHOUSE}")

    print("\nFeatureStore init")
    fs = FeatureStore(
        session=session,
        database=C.DATABASE,
        name=C.FEATURE_STORE_SCHEMA,
        default_warehouse=C.WAREHOUSE,
        creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
    )

    # --- Entity ------------------------------------------------------------
    print("Register entity", C.ENTITY_NAME)
    account = Entity(name=C.ENTITY_NAME, join_keys=[C.ENTITY_JOIN_KEY],
                     desc="Sender account (composite BANK_ID_ACCOUNT_NUMBER)")
    fs.register_entity(account)

    # --- Online service (must exist before online FVs) ---------------------
    ensure_online_service(fs)

    # --- Batch online FV: ACCOUNT_PROFILE ----------------------------------
    print("\nRegister ACCOUNT_PROFILE (batch online FV)")
    profile_df = session.table(C.TBL_ACCOUNT_HISTORY)
    profile_fv = FeatureView(
        name=C.FV_PROFILE,
        entities=[account],
        feature_df=profile_df,
        timestamp_col="LAST_ACTIVITY_TS",
        refresh_freq="1 minute",
        online_config=OnlineConfig(
            enable=True, target_lag="10s", store_type=OnlineStoreType.POSTGRES),
        desc="Slow-moving per-account profile: amounts, dispersion, entity type",
    )
    fs.register_feature_view(profile_fv, version=C.FV_VERSION, overwrite=True)

    # --- Stream source: TRANSACTION_EVENTS ---------------------------------
    print("Register stream source", C.STREAM_SOURCE)
    event_schema = StructType([
        StructField(C.ENTITY_JOIN_KEY, StringType()),
        StructField("EVENT_TS", TimestampType(TimestampTimeZone.NTZ)),
        StructField("AMOUNT_PAID", DoubleType()),
        StructField("RECEIVER_ACCOUNT_ID", StringType()),
        StructField("RECEIVER_BANK", StringType()),
        StructField("IS_CROSS_CURRENCY", LongType()),
        StructField("IS_HIGH_RISK_FORMAT", LongType()),
    ])
    event_stream = StreamSource(
        name=C.STREAM_SOURCE, schema=event_schema,
        desc="Real-time payment events for velocity features",
    )
    fs.register_stream_source(event_stream)

    # --- Stream FV: ACCOUNT_VELOCITY (continuous time-windowed aggs) --------
    print("Register ACCOUNT_VELOCITY (stream FV, CONTINUOUS aggregation)")
    backfill_df = session.table(
        f"{C.DATABASE}.{C.CURATED_SCHEMA}.TXN_EVENTS"
    ).select(
        F.col(C.ENTITY_JOIN_KEY),
        F.col("EVENT_TS"),
        F.col("AMOUNT_PAID").cast("double").alias("AMOUNT_PAID"),
        F.col("RECEIVER_ACCOUNT_ID"),
        F.col("RECEIVER_BANK"),
        F.col("IS_CROSS_CURRENCY").cast("bigint").alias("IS_CROSS_CURRENCY"),
        F.col("IS_HIGH_RISK_FORMAT").cast("bigint").alias("IS_HIGH_RISK_FORMAT"),
    )
    stream_cfg = StreamConfig(
        stream_source=event_stream,
        transformation_fn=passthrough,
        backfill_df=backfill_df,
    )
    velocity_features = [
        Feature.count("AMOUNT_PAID", "1h").alias("TXN_COUNT_1H"),
        Feature.count("AMOUNT_PAID", "24h").alias("TXN_COUNT_24H"),
        Feature.sum("AMOUNT_PAID", "24h").alias("AMT_SUM_24H"),
        Feature.sum("AMOUNT_PAID", "48h").alias("AMT_SUM_48H"),
        Feature.approx_count_distinct("RECEIVER_BANK", "24h").alias("DISTINCT_BANKS_24H"),
        Feature.approx_count_distinct("RECEIVER_ACCOUNT_ID", "24h").alias("DISTINCT_RECEIVERS_24H"),
        Feature.sum("IS_CROSS_CURRENCY", "24h").alias("CROSS_CCY_CNT_24H"),
        Feature.sum("IS_HIGH_RISK_FORMAT", "24h").alias("HIGH_RISK_CNT_24H"),
    ]
    velocity_fv = FeatureView(
        name=C.FV_VELOCITY,
        entities=[account],
        stream_config=stream_cfg,
        timestamp_col="EVENT_TS",
        refresh_freq="1 minute",
        feature_granularity="1 hour",
        features=velocity_features,
        online_config=OnlineConfig(enable=True, store_type=OnlineStoreType.POSTGRES),
        feature_aggregation_method=FeatureAggregationMethod.CONTINUOUS,
        desc="Real-time per-account velocity: counts, spend, distinct banks, risk",
    )
    fs.register_feature_view(velocity_fv, version=C.FV_VERSION, overwrite=True)

    # --- Feature group: FRAUD_FEATURES -------------------------------------
    print("Register FRAUD_FEATURES (feature group)")
    reg_profile = fs.get_feature_view(C.FV_PROFILE, C.FV_VERSION)
    reg_velocity = fs.get_feature_view(C.FV_VELOCITY, C.FV_VERSION)
    fg = FeatureGroup(
        name=C.FEATURE_GROUP,
        features=[reg_profile, reg_velocity],
        auto_prefix=False,
        desc="Profile + velocity features for AML fraud scoring",
    )
    fs.register_feature_group(fg, C.FV_VERSION.lower())

    # --- Verify online reads -----------------------------------------------
    print("\nVerification:")
    sample = session.table(C.TBL_ACCOUNT_HISTORY).select(C.ENTITY_JOIN_KEY).limit(3).collect()
    keys = [[r[C.ENTITY_JOIN_KEY]] for r in sample]
    print(f"  sample keys: {keys}")
    try:
        online = fs.read_feature_view(reg_profile, keys=keys, store_type="online")
        online.show()
    except Exception as e:
        print(f"  (online read not ready yet: {e})")
    print("\nSetup 02 complete.")
    session.close()


if __name__ == "__main__":
    main()
