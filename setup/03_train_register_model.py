"""Setup 03 - Generate point-in-time training set, train XGBoost, register model.

- Pulls a downsampled spine (all positives + sampled negatives, split preserved)
  and joins profile + velocity features via the FRAUD_FEATURES feature group with
  point-in-time correctness (spine_timestamp_col=EVENT_TS).
- Trains an imbalance-aware XGBoost classifier (scale_pos_weight) on the TRAIN
  split, evaluates on TEST (chronologically later). Reports PR-AUC + recall@1%
  (NOT accuracy - the base rate is ~0.1%).
- Registers the model to the Snowflake Model Registry with an explicit signature.

Run: SNOWFLAKE_CONNECTION_NAME=<conn> .venv/bin/python setup/03_train_register_model.py
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from snowpark_session import create_snowpark_session

from snowflake.ml.feature_store import FeatureStore, CreationMode
from snowflake.ml.registry import Registry

# Negatives to retain (all positives are always kept). Keeps the point-in-time
# join + local training tractable while preserving chronological splits.
NEG_KEEP = 400_000

# Columns that must never be model features.
EXCLUDE = {
    C.ENTITY_JOIN_KEY, "EVENT_TS", "LAST_ACTIVITY_TS", "SPLIT", "IS_LAUNDERING",
    "RECEIVING_CURRENCY", "PAYMENT_CURRENCY", "TO_COUNTRY", "HOME_COUNTRY",
}
CATEGORICAL = ["PAYMENT_FORMAT", "ENTITY_TYPE"]


def recall_at_k(y_true, y_score, k_frac=0.01):
    n = max(1, int(len(y_score) * k_frac))
    idx = np.argsort(y_score)[::-1][:n]
    caught = y_true[idx].sum()
    total = y_true.sum()
    return caught / total if total else 0.0


def main():
    session = create_snowpark_session()
    session.sql(f"USE WAREHOUSE {C.WAREHOUSE}").collect()
    session.sql(f"USE SCHEMA {C.DATABASE}.{C.FEATURE_STORE_SCHEMA}").collect()
    fs = FeatureStore(session=session, database=C.DATABASE,
                      name=C.FEATURE_STORE_SCHEMA, default_warehouse=C.WAREHOUSE,
                      creation_mode=CreationMode.FAIL_IF_NOT_EXIST)

    # --- Downsampled spine (all positives + sampled negatives) -------------
    print("Building downsampled spine")
    session.sql(f"""
        CREATE OR REPLACE TABLE {C.DATABASE}.{C.CURATED_SCHEMA}.TXN_SPINE_SAMPLE AS
        SELECT * FROM {C.TBL_TXN_SPINE} WHERE IS_LAUNDERING = 1
        UNION ALL
        SELECT * FROM (
            SELECT * FROM {C.TBL_TXN_SPINE} WHERE IS_LAUNDERING = 0
        ) SAMPLE ({NEG_KEEP} ROWS)
    """).collect()
    spine = session.table(f"{C.DATABASE}.{C.CURATED_SCHEMA}.TXN_SPINE_SAMPLE")
    print(f"  spine sample rows: {spine.count():,}")

    # --- Point-in-time training set via feature group ----------------------
    print("Generating training set (point-in-time join via FRAUD_FEATURES)")
    fg = fs.get_feature_group(C.FEATURE_GROUP, C.FV_VERSION.lower())
    training_df = fs.generate_training_set(
        spine_df=spine,
        feature_group=fg,
        spine_timestamp_col="EVENT_TS",
        spine_label_cols=["IS_LAUNDERING"],
    )
    pdf = training_df.to_pandas()
    print(f"  training frame: {pdf.shape}")

    # --- Assemble feature matrix -------------------------------------------
    pdf.columns = [c.upper() for c in pdf.columns]
    keep_split = pdf["SPLIT"].values
    y = pdf["IS_LAUNDERING"].astype(int).values

    feat_cols = [c for c in pdf.columns if c not in EXCLUDE]
    # One-hot the low-cardinality categoricals; numeric-coerce the rest.
    cat_present = [c for c in CATEGORICAL if c in pdf.columns]
    num_cols = [c for c in feat_cols if c not in cat_present]
    X_num = pdf[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X = pd.concat([X_num, pd.get_dummies(pdf[cat_present].astype(str),
                                         prefix=cat_present)], axis=1)
    X.columns = [str(c).upper().replace(" ", "_") for c in X.columns]
    print(f"  feature count: {X.shape[1]} (numeric={len(num_cols)}, "
          f"cat-onehot={X.shape[1]-len(num_cols)})")

    tr = keep_split == "TRAIN"
    te = keep_split == "TEST"
    pos, neg = int(y[tr].sum()), int((~y[tr].astype(bool)).sum())
    spw = neg / max(1, pos)
    print(f"  train pos={pos} neg={neg} scale_pos_weight={spw:.1f}; "
          f"test n={te.sum():,} pos={int(y[te].sum())}")

    # --- Train gradient-boosted trees (sklearn HGB; portable, no libomp dep) -
    # Imbalance handled via sample_weight (positives weighted by neg/pos ratio).
    # MONOTONIC CONSTRAINTS: domain knowledge says more velocity / dispersion /
    # amount / cross-border activity should never DECREASE fraud risk. Enforcing
    # this removes counterintuitive boundaries and makes the model behave
    # sensibly for a probing audience (and makes a fan-out burst raise the score).
    MONO_POS = {
        "AMOUNT_PAID", "IS_CROSS_CURRENCY", "IS_CROSS_BORDER", "IS_HIGH_RISK_FORMAT",
        "AMOUNT_TO_AVG_RATIO", "TXN_COUNT_1H", "TXN_COUNT_24H", "AMT_SUM_24H",
        "AMT_SUM_48H", "DISTINCT_BANKS_24H", "DISTINCT_RECEIVERS_24H",
        "CROSS_CCY_CNT_24H", "HIGH_RISK_CNT_24H",
    }
    monotonic_cst = [1 if c in MONO_POS else 0 for c in X.columns]
    print(f"  monotonic +1 features: {sum(monotonic_cst)} of {X.shape[1]}")

    sw = np.where(y[tr] == 1, spw, 1.0)
    model = HistGradientBoostingClassifier(
        max_iter=300, max_depth=6, learning_rate=0.1,
        l2_regularization=1.0, min_samples_leaf=50,
        monotonic_cst=monotonic_cst, random_state=42,
    )
    model.fit(X[tr], y[tr], sample_weight=sw)

    # --- Evaluate (PR-AUC + recall@1%, NOT accuracy) -----------------------
    proba = model.predict_proba(X[te])[:, 1]
    pr_auc = average_precision_score(y[te], proba)
    roc = roc_auc_score(y[te], proba)
    r_at_1 = recall_at_k(y[te], proba, 0.01)
    base = y[te].mean()
    print("\n=== TEST METRICS (chronologically later split) ===")
    print(f"  base rate (prevalence in sample): {base:.4%}")
    print(f"  PR-AUC:        {pr_auc:.4f}  (random baseline = base rate)")
    print(f"  ROC-AUC:       {roc:.4f}")
    print(f"  Recall@1%:     {r_at_1:.4f}  (share of fraud caught in top 1% scored)")
    print("  NOTE: negatives were downsampled, so absolute precision is optimistic")
    print("        vs. true ~0.1% prevalence; ranking metrics remain informative.")

    # --- Local demo-scenario sanity check (before the expensive deploy) -----
    # Confirm a new-account fan-out burst scores HIGH and a quiet baseline LOW.
    def scenario(**ov):
        rowv = {c: 0.0 for c in X.columns}
        if "PAYMENT_FORMAT_ACH" in rowv:
            rowv["PAYMENT_FORMAT_ACH"] = 1.0
        rowv.update(ov)
        xx = pd.DataFrame([[rowv[c] for c in X.columns]], columns=X.columns)
        return float(model.predict_proba(xx)[:, 1][0])

    baseline = scenario(AMOUNT_PAID=1500, TXN_COUNT_1H=1, TXN_COUNT_24H=1,
                        DISTINCT_BANKS_24H=1, AMT_SUM_24H=1500, AMT_SUM_48H=1500)
    fanout = scenario(AMOUNT_PAID=9000, AMOUNT_TO_AVG_RATIO=5,
                      IS_CROSS_CURRENCY=1, IS_CROSS_BORDER=1,
                      TXN_COUNT_1H=40, TXN_COUNT_24H=40, AMT_SUM_24H=300000,
                      AMT_SUM_48H=300000, DISTINCT_BANKS_24H=35,
                      DISTINCT_RECEIVERS_24H=40, CROSS_CCY_CNT_24H=28,
                      HIGH_RISK_CNT_24H=40)
    print("\n=== DEMO SCENARIO CHECK (local) ===")
    print(f"  quiet single txn (new acct):   P(fraud)={baseline:.4f}")
    print(f"  new-account fan-out burst:     P(fraud)={fanout:.4f}")

    # --- Register to Model Registry ----------------------------------------
    print("\nRegistering model to registry")
    reg = Registry(session=session, database_name=C.DATABASE,
                   schema_name=C.FEATURE_STORE_SCHEMA)
    sample_input = X[te].head(20)
    mv = reg.log_model(
        model=model,
        model_name=C.MODEL_NAME,
        version_name=C.MODEL_VERSION,
        sample_input_data=sample_input,
        comment="AML fraud gradient-boosting (profile+velocity, imbalance-aware). Demo-grade.",
        options={"relax_version": True},
    )
    print(f"  registered {C.MODEL_NAME}/{C.MODEL_VERSION}")
    print("  feature order:", list(X.columns))
    print("\nSetup 03 complete.")
    session.close()


if __name__ == "__main__":
    main()
