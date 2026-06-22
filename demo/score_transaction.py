"""Demo - Score a transaction in real time (the payoff).

Flow:
  1. Read the full feature vector for an account from the FRAUD_FEATURES feature
     group (profile + velocity) via the online store - one round trip.
  2. Overlay the current transaction's request-context fields.
  3. Align columns to the model's signature and POST to the real-time inference
     endpoint (dataframe_split). Print the fraud probability + round-trip latency.

  --scenario fraud : first fire a fan-out velocity burst (stream_events.py),
                     then score the same account so the team sees the score jump.

Usage:
  export SNOWFLAKE_PAT=...
  .venv/bin/python demo/score_transaction.py --account <ACCOUNT_ID> [--amount 250000]
  .venv/bin/python demo/score_transaction.py --account <ACCOUNT_ID> --scenario fraud
"""

import argparse
import json
import os
import subprocess
import sys
import time

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from snowpark_session import create_snowpark_session

from snowflake.ml.feature_store import FeatureStore, CreationMode
from snowflake.ml.registry import Registry


def get_inference_url(session):
    rows = session.sql(
        f"SHOW ENDPOINTS IN SERVICE {C.DATABASE}.{C.FEATURE_STORE_SCHEMA}.{C.INFERENCE_SERVICE}"
    ).collect()
    for r in rows:
        d = r.as_dict()
        u = d.get("ingress_url") or ""
        if u and "snowflakecomputing" in u:
            return f"https://{u}"
    raise RuntimeError("Ingress URL not provisioned yet; retry in a minute")


def model_feature_spec(reg):
    """Return [(name, dtype_str)] in signature order for the predict method."""
    mv = reg.get_model(C.MODEL_NAME).version(C.MODEL_VERSION)
    fns = mv.show_functions()
    fn = next((f for f in fns if f["target_method"].lower() == "predict"), fns[0])
    out = []
    for s in fn["signature"].inputs:
        dt = str(getattr(s, "_dtype", "")).upper()
        out.append((str(s.name), dt))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--account", required=True)
    ap.add_argument("--amount", type=float, default=None,
                    help="txn amount (default: 2500 normal / 9000 fraud)")
    ap.add_argument("--scenario", choices=["normal", "fraud"], default="normal")
    ap.add_argument("--format", dest="fmt", default=None,
                    help="payment format (default Wire for fraud, ACH for normal)")
    args = ap.parse_args()
    if args.amount is None:
        args.amount = 9000.0 if args.scenario == "fraud" else 2500.0

    pat = C.get_pat()
    session = create_snowpark_session()
    session.sql(f"USE WAREHOUSE {C.WAREHOUSE}").collect()
    session.sql(f"USE SCHEMA {C.DATABASE}.{C.FEATURE_STORE_SCHEMA}").collect()
    fs = FeatureStore(session=session, database=C.DATABASE,
                      name=C.FEATURE_STORE_SCHEMA, default_warehouse=C.WAREHOUSE,
                      creation_mode=CreationMode.FAIL_IF_NOT_EXIST)
    reg = Registry(session=session, database_name=C.DATABASE,
                   schema_name=C.FEATURE_STORE_SCHEMA)

    if args.scenario == "fraud":
        print("=== Firing fan-out velocity burst, then scoring ===")
        subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__),
                        "stream_events.py"), "--mode", "fraud", "--account",
                        args.account, "--count", "40", "--rate", "20"], check=False)
        print("waiting 3s for continuous aggregation to settle...")
        time.sleep(3)

    # 1. Read feature group online (profile + velocity)
    fg = fs.get_feature_group(C.FEATURE_GROUP, C.FV_VERSION.lower())
    feats = fs.read_feature_group(fg, keys=[[args.account]])
    feats.columns = [c.upper() for c in feats.columns]

    # Profile (slow-moving batch FV) fallback: if the online profile row isn't
    # served, read it from its source table so the model gets full features.
    if "HIST_TXN_COUNT" not in feats.columns or pd.isna(feats["HIST_TXN_COUNT"].iloc[0]):
        prof = session.sql(
            f"SELECT * FROM {C.TBL_ACCOUNT_HISTORY} WHERE {C.ENTITY_JOIN_KEY}='{args.account}'"
        ).to_pandas()
        if len(prof):
            for c in prof.columns:
                feats[c.upper()] = prof[c].iloc[0]
            print("(profile served from source table - batch online sync lagging)")

    print("\nLooked-up features (profile + velocity):")
    print(feats.to_string(index=False))

    # 2. Overlay request context for THIS transaction
    # ACH is the channel real laundering uses in this dataset, so it's the
    # default for both scenarios; the fraud signal comes from velocity/dispersion.
    fmt = args.fmt or "ACH"
    feats["AMOUNT_PAID"] = args.amount
    feats["IS_CROSS_CURRENCY"] = 1 if args.scenario == "fraud" else 0
    feats["IS_CROSS_BORDER"] = 1 if args.scenario == "fraud" else 0
    feats["IS_HIGH_RISK_FORMAT"] = 1 if fmt in ("Wire", "Bitcoin", "Cash") else 0
    avg = float(feats.get("HIST_AVG_AMOUNT", pd.Series([0])).iloc[0] or 0)
    feats["AMOUNT_TO_AVG_RATIO"] = args.amount / avg if avg else 1.0
    # Set the payment-format one-hot the model was trained on.
    fmt_col = "PAYMENT_FORMAT_" + fmt.upper().replace(" ", "_")

    # 3. Align to model signature, coercing each value to its declared dtype
    #    (INT8 flags/one-hots, DOUBLE numerics) and set the active format one-hot.
    spec = model_feature_spec(reg)
    order = [name for name, _ in spec]

    def coerce(v, dt):
        try:
            f = float(v)
            if f != f:  # NaN
                f = 0.0
        except (TypeError, ValueError):
            f = 0.0
        if "BOOL" in dt:
            return bool(int(f))
        if "INT" in dt:
            return int(f)
        return f

    dtype_of = dict(spec)
    row = {name: (coerce(feats[name].iloc[0], dt) if name in feats.columns
                  else (0 if ("INT" in dt or "BOOL" in dt) else 0.0))
           for name, dt in spec}
    if fmt_col in order:
        row[fmt_col] = coerce(1, dtype_of[fmt_col])
    payload = {"dataframe_split": {"index": [0], "columns": order,
                                   "data": [[row[c] for c in order]]}}

    url = get_inference_url(session) + "/predict-proba"
    t0 = time.time()
    resp = requests.post(url, headers={"Authorization": f'Snowflake Token="{pat}"',
                                       "Content-Type": "application/json"},
                         json=payload, timeout=30)
    dt = (time.time() - t0) * 1000
    print(f"\nInference HTTP {resp.status_code}  round-trip {dt:.0f} ms")
    try:
        body = resp.json()
        print("Raw response:", body)
        # predict_proba returns class probabilities; extract P(fraud)=class 1.
        rec = body["data"][0][1]
        proba = rec.get("output_feature_1", rec.get("1"))
        if proba is not None:
            print(f"\n>>> FRAUD PROBABILITY: {float(proba):.4f}  (account {args.account}, "
                  f"{args.scenario} scenario)")
    except Exception as e:
        print(resp.text, e)
    session.close()


if __name__ == "__main__":
    main()
