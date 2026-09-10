"""OPTIONAL online path - Postgres-backed Online Feature Store + stream velocity.

NOT part of the default batch demo. Runs 24/7 (Postgres HA) => costs money on a
demo account. See setup/online/README.md. Tear down with setup/99_reset_teardown.py.

Env-aware: builds the online service in the current ML_ENV's feature store,
backfilling the velocity stream FV from prod CURATED.TXN_EVENTS.

  SNOWFLAKE_CONNECTION_NAME=<your-connection> ML_ENV=dev SNOWFLAKE_PAT=... \
    .venv/bin/python setup/online/setup_online.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config as C
from snowpark_session import create_snowpark_session

from snowflake.ml.feature_store import (
    FeatureStore, Entity, FeatureView, OnlineConfig, OnlineStoreType,
    StreamSource, StreamConfig, Feature, CreationMode,
)
from snowflake.ml.feature_store.spec.enums import FeatureAggregationMethod
from snowflake.snowpark.types import (
    StructType, StructField, StringType, DoubleType, LongType,
    TimestampType, TimestampTimeZone,
)
import snowflake.snowpark.functions as F


def passthrough(df):
    return df


def ensure_online_service(fs):
    print("Online service (Postgres):")
    try:
        st = fs.get_online_service_status()
        if getattr(st, "status", None) == "RUNNING":
            print("  already RUNNING"); return
    except Exception as e:
        print(f"  none yet ({type(e).__name__}); creating")
    fs.create_online_service(C.DEV_ROLE, C.DEV_ROLE)
    waited = 0
    while getattr(fs.get_online_service_status(), "status", None) != "RUNNING":
        time.sleep(30); waited += 30
        print(f"    [{waited}s] provisioning...")
        if waited > 1800:
            raise TimeoutError("online service did not reach RUNNING")
    print("  RUNNING")


def main():
    if "SNOWFLAKE_PAT" not in os.environ:
        try:
            os.environ["SNOWFLAKE_PAT"] = C.get_pat()
        except Exception:
            print("WARNING: SNOWFLAKE_PAT not set; online reads/REST will fail.")
    session = create_snowpark_session()
    session.sql(f"USE ROLE {C.DEV_ROLE}").collect()
    session.sql(f"USE WAREHOUSE {C.WAREHOUSE}").collect()
    session.sql(f"USE SCHEMA {C.DATABASE}.{C.FEATURE_STORE_SCHEMA}").collect()

    fs = FeatureStore(session=session, database=C.DATABASE, name=C.FEATURE_STORE_SCHEMA,
                      default_warehouse=C.WAREHOUSE, creation_mode=CreationMode.CREATE_IF_NOT_EXIST)
    account = Entity(name=C.ENTITY_NAME, join_keys=[C.ENTITY_JOIN_KEY], desc="Sender account")
    fs.register_entity(account)

    ensure_online_service(fs)

    # ACCOUNT_PROFILE online (Postgres)
    profile_df = session.sql(f"""
        SELECT {C.ENTITY_JOIN_KEY}, HIST_TXN_COUNT, HIST_AVG_AMOUNT, HIST_STD_AMOUNT,
               HIST_MAX_AMOUNT, HIST_DISTINCT_RECEIVERS, HIST_DISTINCT_RECEIVER_BANKS,
               HIST_DISTINCT_COUNTRIES, HIST_FOREIGN_CCY_SHARE, HIST_HIGH_RISK_SHARE,
               LAST_ACTIVITY_TS
        FROM {C.TBL_ACCOUNT_HISTORY}""")
    profile_fv = FeatureView(
        name=C.FV_PROFILE, entities=[account], feature_df=profile_df,
        timestamp_col="LAST_ACTIVITY_TS", refresh_freq="1 minute",
        online_config=OnlineConfig(enable=True, target_lag="10s", store_type=OnlineStoreType.POSTGRES),
        desc="Per-account profile served online (Postgres).")
    fs.register_feature_view(profile_fv, version=C.FV_VERSION, overwrite=True)

    # TRANSACTION_EVENTS stream source + ACCOUNT_VELOCITY stream FV (continuous aggs)
    event_schema = StructType([
        StructField(C.ENTITY_JOIN_KEY, StringType()),
        StructField("EVENT_TS", TimestampType(TimestampTimeZone.NTZ)),
        StructField("AMOUNT_PAID", DoubleType()),
        StructField("RECEIVER_ACCOUNT_ID", StringType()),
        StructField("RECEIVER_BANK", StringType()),
        StructField("IS_CROSS_CURRENCY", LongType()),
        StructField("IS_HIGH_RISK_FORMAT", LongType()),
    ])
    stream = StreamSource(name=C.STREAM_SOURCE, schema=event_schema,
                          desc="Real-time payment events for velocity features")
    fs.register_stream_source(stream)

    backfill = session.table(f"{C.PROD_DATABASE}.{C.CURATED_SCHEMA}.TXN_EVENTS").select(
        F.col(C.ENTITY_JOIN_KEY), F.col("EVENT_TS"),
        F.col("AMOUNT_PAID").cast("double").alias("AMOUNT_PAID"),
        F.col("RECEIVER_ACCOUNT_ID"), F.col("RECEIVER_BANK"),
        F.col("IS_CROSS_CURRENCY").cast("bigint").alias("IS_CROSS_CURRENCY"),
        F.col("IS_HIGH_RISK_FORMAT").cast("bigint").alias("IS_HIGH_RISK_FORMAT"))
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
        name=C.FV_VELOCITY, entities=[account],
        stream_config=StreamConfig(stream_source=stream, transformation_fn=passthrough, backfill_df=backfill),
        timestamp_col="EVENT_TS", refresh_freq="1 minute", feature_granularity="1 hour",
        features=velocity_features,
        online_config=OnlineConfig(enable=True, store_type=OnlineStoreType.POSTGRES),
        feature_aggregation_method=FeatureAggregationMethod.CONTINUOUS,
        desc="Real-time per-account velocity (continuous aggregation).")
    fs.register_feature_view(velocity_fv, version=C.FV_VERSION, overwrite=True)
    print("Online path ready (Postgres). Remember to tear down to stop cost.")
    session.close()


if __name__ == "__main__":
    main()
