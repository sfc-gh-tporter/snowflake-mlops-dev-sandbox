"""Demo (LIVE) - Register the real-time feature view TXN_RISK_SIGNALS.

This is the "register one feature view live" moment. A real-time feature view
runs a Python compute_fn at QUERY time (not precomputed). It combines per-request
transaction context (RequestSource) with the stored ACCOUNT_PROFILE to derive
signals that only exist at the moment of the payment - the canonical example
being "current amount vs. this account's historical average".

It is a teaching artifact shown via read_feature_view; it is NOT a model input
(the model was trained on profile + velocity), so registering it live needs no
retrain and no infra provisioning - it is instant.

Run: SNOWFLAKE_CONNECTION_NAME=<conn> .venv/bin/python demo/register_realtime_fv.py
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from snowpark_session import create_snowpark_session

from snowflake.ml.feature_store import (
    FeatureStore, CreationMode, FeatureView, RealtimeConfig, RequestSource,
)
from snowflake.snowpark.types import (
    StructType, StructField, DoubleType, LongType, StringType,
)


def compute_txn_risk(request_df: pd.DataFrame, profile_df: pd.DataFrame) -> pd.DataFrame:
    """Request-time risk signals. Rows align positionally by entity key.

    request_df : current transaction context (RequestSource columns)
    profile_df : latest ACCOUNT_PROFILE values for the same accounts
    """
    amount = request_df["AMOUNT_PAID"].astype(float).reset_index(drop=True)
    pay_ccy = request_df["PAYMENT_CURRENCY"].astype(str).reset_index(drop=True)
    rcv_ccy = request_df["RECEIVING_CURRENCY"].astype(str).reset_index(drop=True)

    avg = profile_df["HIST_AVG_AMOUNT"].fillna(0.0).reset_index(drop=True)
    std = profile_df["HIST_STD_AMOUNT"].fillna(0.0).reset_index(drop=True)

    ratio = amount / avg.replace(0.0, float("nan"))
    ratio = ratio.fillna(1.0)
    zscore = (amount - avg) / std.replace(0.0, float("nan"))
    zscore = zscore.fillna(0.0)

    return pd.DataFrame({
        "AMOUNT_TO_AVG_RATIO": ratio,
        "AMOUNT_ZSCORE": zscore,
        "IS_GT_3X_AVG": (ratio > 3.0).astype(int),
        "IS_CROSS_CURRENCY": (pay_ccy != rcv_ccy).astype(int),
    })


def main():
    session = create_snowpark_session()
    session.sql(f"USE WAREHOUSE {C.WAREHOUSE}").collect()
    session.sql(f"USE SCHEMA {C.DATABASE}.{C.FEATURE_STORE_SCHEMA}").collect()
    fs = FeatureStore(session=session, database=C.DATABASE,
                      name=C.FEATURE_STORE_SCHEMA, default_warehouse=C.WAREHOUSE,
                      creation_mode=CreationMode.FAIL_IF_NOT_EXIST)

    reg_profile = fs.get_feature_view(C.FV_PROFILE, C.FV_VERSION)

    request_source = RequestSource(schema=StructType([
        StructField("AMOUNT_PAID", DoubleType()),
        StructField("PAYMENT_CURRENCY", StringType()),
        StructField("RECEIVING_CURRENCY", StringType()),
    ]))

    rtfv = FeatureView(
        name=C.FV_REALTIME,
        entities=[fs.get_entity(C.ENTITY_NAME)],
        realtime_config=RealtimeConfig(
            compute_fn=compute_txn_risk,
            sources=[request_source, reg_profile],
            output_schema=StructType([
                StructField("AMOUNT_TO_AVG_RATIO", DoubleType()),
                StructField("AMOUNT_ZSCORE", DoubleType()),
                StructField("IS_GT_3X_AVG", LongType()),
                StructField("IS_CROSS_CURRENCY", LongType()),
            ]),
        ),
        desc="Request-time risk signals: current amount vs. account history",
    )
    reg_rtfv = fs.register_feature_view(rtfv, C.FV_VERSION, overwrite=True)
    print(f"Registered real-time FV {reg_rtfv.name}/{reg_rtfv.version}")

    # --- Demonstrate request-time computation ------------------------------
    sample = session.table(C.TBL_ACCOUNT_HISTORY).select(
        C.ENTITY_JOIN_KEY, "HIST_AVG_AMOUNT").limit(2).collect()
    keys = [[r[C.ENTITY_JOIN_KEY]] for r in sample]
    # Score a deliberately large, cross-currency amount to show the ratio jump.
    ctx = pd.DataFrame({
        "AMOUNT_PAID": [float(sample[0]["HIST_AVG_AMOUNT"] or 0) * 8 + 1, 50.0],
        "PAYMENT_CURRENCY": ["US Dollar", "US Dollar"],
        "RECEIVING_CURRENCY": ["Bitcoin", "US Dollar"],
    })
    print("Reading real-time FV (request-time compute):")
    out = fs.read_feature_view(reg_rtfv, keys=keys, request_context=ctx)
    print(out.to_pandas().to_string(index=False) if hasattr(out, "to_pandas") else out)
    print("\nLive FV registration complete.")
    session.close()


if __name__ == "__main__":
    main()
