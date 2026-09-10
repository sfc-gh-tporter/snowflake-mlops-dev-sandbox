"""Phase 5 - Train in the dev sandbox: feature store -> model -> experiment + dev registry.

- Builds a downsampled training set from the dev ABT (all positives + sampled
  negatives), joining ACCOUNT_PROFILE features from the feature store.
- Trains an imbalance-aware, monotonic HistGradientBoosting classifier.
- Logs params + metrics to a Snowflake ML Experiment (dev EXPERIMENTS schema).
- Registers the model version to the DEV model registry (ML_FRAUD_DEV_SANDBOX.ML).

Everything runs as ML_DEV_ROLE - the data scientist can do all of this in dev,
but cannot touch prod.

  SNOWFLAKE_CONNECTION_NAME=demo156_keypair ML_ENV=dev .venv/bin/python setup/03_train_register.py [--version V1]
"""

import argparse
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
from snowflake.ml.experiment import ExperimentTracking

EXPERIMENT = "AML_FRAUD_TRAINING"
EXCLUDE = {C.ENTITY_JOIN_KEY, "EVENT_TS", "SPLIT", "IS_LAUNDERING",
           "PAYMENT_CURRENCY", "RECEIVING_CURRENCY", "TO_COUNTRY"}
CATEGORICAL = ["PAYMENT_FORMAT"]
MONO_POS = {"AMOUNT_PAID", "IS_CROSS_CURRENCY", "IS_CROSS_BORDER",
            "IS_HIGH_RISK_FORMAT", "AMOUNT_TO_AVG_RATIO"}


def recall_at_k(y_true, y_score, k_frac=0.01):
    n = max(1, int(len(y_score) * k_frac))
    idx = np.argsort(y_score)[::-1][:n]
    return y_true[idx].sum() / y_true.sum() if y_true.sum() else 0.0


def build_training_frame(session, fs, abt):
    # Downsample negatives (keep all positives), preserving SPLIT.
    sample = f"{C.DATABASE}.{C.CURATED_SCHEMA}.FRAUD_ABT_SAMPLE"
    session.sql(f"""
        CREATE OR REPLACE TABLE {sample} AS
        SELECT * FROM {abt} WHERE IS_LAUNDERING = 1
        UNION ALL
        SELECT * FROM (SELECT * FROM {abt} WHERE IS_LAUNDERING = 0) SAMPLE ({C.TRAIN_NEG_SAMPLE} ROWS)
    """).collect()
    spine = session.table(sample)
    profile_fv = fs.get_feature_view(C.FV_PROFILE, C.FV_VERSION)
    training_df = fs.generate_training_set(
        spine_df=spine, features=[profile_fv], spine_label_cols=["IS_LAUNDERING"])
    pdf = training_df.to_pandas()
    pdf.columns = [c.upper() for c in pdf.columns]
    return pdf


def assemble_matrix(pdf):
    feat_cols = [c for c in pdf.columns if c not in EXCLUDE]
    cat = [c for c in CATEGORICAL if c in pdf.columns]
    num = [c for c in feat_cols if c not in cat]
    X_num = pdf[num].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X = pd.concat([X_num, pd.get_dummies(pdf[cat].astype(str), prefix=cat)], axis=1)
    X.columns = [str(c).upper().replace(" ", "_") for c in X.columns]
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="V1")
    args = ap.parse_args()

    session = create_snowpark_session()
    session.sql(f"USE ROLE {C.DEV_ROLE}").collect()
    session.sql(f"USE WAREHOUSE {C.WAREHOUSE}").collect()
    session.sql(f"USE SCHEMA {C.DATABASE}.{C.FEATURE_STORE_SCHEMA}").collect()
    fs = FeatureStore(session=session, database=C.DATABASE, name=C.FEATURE_STORE_SCHEMA,
                      default_warehouse=C.WAREHOUSE, creation_mode=CreationMode.FAIL_IF_NOT_EXIST)

    print("Building training set from feature store (ABT + ACCOUNT_PROFILE)")
    pdf = build_training_frame(session, fs, C.abt_fqn("dev"))
    y = pdf["IS_LAUNDERING"].astype(int).values
    split = pdf["SPLIT"].values
    X = assemble_matrix(pdf)
    print(f"  training frame {pdf.shape}; features {X.shape[1]}")

    tr, te = split == "TRAIN", split == "TEST"
    pos, neg = int(y[tr].sum()), int((~y[tr].astype(bool)).sum())
    spw = neg / max(1, pos)
    monotonic_cst = [1 if c in MONO_POS else 0 for c in X.columns]
    params = dict(max_iter=300, max_depth=6, learning_rate=0.1,
                  l2_regularization=1.0, min_samples_leaf=50, random_state=42)
    print(f"  train pos={pos} neg={neg} spw={spw:.1f}; monotonic+1={sum(monotonic_cst)}")

    model = HistGradientBoostingClassifier(monotonic_cst=monotonic_cst, **params)
    model.fit(X[tr], y[tr], sample_weight=np.where(y[tr] == 1, spw, 1.0))

    proba = model.predict_proba(X[te])[:, 1]
    metrics = {
        "pr_auc": float(average_precision_score(y[te], proba)),
        "roc_auc": float(roc_auc_score(y[te], proba)),
        "recall_at_1pct": float(recall_at_k(y[te], proba, 0.01)),
        "test_base_rate": float(y[te].mean()),
    }
    print("  TEST metrics:", {k: round(v, 4) for k, v in metrics.items()})

    # --- Experiment tracking (dev EXPERIMENTS schema) ----------------------
    print("Logging experiment run")
    exp = ExperimentTracking(session=session, database_name=C.DEV_DATABASE,
                             schema_name=C.EXPERIMENTS_SCHEMA)
    exp.set_experiment(EXPERIMENT)
    run_name = f"train_{args.version}"
    try:
        exp.delete_run(run_name)  # idempotent re-runs
    except Exception:
        pass
    with exp.start_run(run_name):
        exp.log_params({**params, "scale_pos_weight": round(spw, 2),
                        "n_features": X.shape[1], "neg_sample": C.TRAIN_NEG_SAMPLE,
                        "monotonic_pos": sum(monotonic_cst)})
        exp.log_metrics(metrics)

    # --- Register to DEV registry ------------------------------------------
    print(f"Registering to dev registry {C.DEV_DATABASE}.{C.REGISTRY_SCHEMA}")
    reg = Registry(session=session, database_name=C.DEV_DATABASE, schema_name=C.REGISTRY_SCHEMA)
    mv = reg.log_model(
        model=model, model_name=C.MODEL_NAME, version_name=args.version,
        sample_input_data=X[te].head(20),
        comment="AML fraud GBM (profile + request-context, monotonic, imbalance-aware).",
        metrics=metrics)
    reg.get_model(C.MODEL_NAME).default = args.version
    print(f"  registered {C.MODEL_NAME}/{args.version} in dev; set as default")
    print("  feature order:", list(X.columns))
    session.close()


if __name__ == "__main__":
    main()
