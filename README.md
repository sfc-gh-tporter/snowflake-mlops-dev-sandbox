# Real-Time Feature Store - Fraud / AML Transaction Monitoring Demo

An end-to-end ML demo on Snowflake's **Online Feature Store (Postgres-backed, preview)**
and **Real-Time Inference REST API**. Every incoming payment is scored for fraud / AML
risk in milliseconds by combining three kinds of features:

| Feature type | Feature view | Example signals | Freshness |
|---|---|---|---|
| Slow-moving account profile | `ACCOUNT_PROFILE` (batch online) | lifetime avg/std amount, distinct receiver banks/countries, entity type | minutes |
| Fast-moving velocity | `ACCOUNT_VELOCITY` (stream, continuous agg) | txn count 1h/24h, spend 24h/48h, **distinct banks 24h**, cross-currency & high-risk-format counts | < 2 sec |
| Request-time signals | `TXN_RISK_SIGNALS` (real-time, registered live) | current amount vs. account average, z-score, cross-currency flag | per request |

The star is the **serving architecture** (online feature store + real-time inference),
not the model. Generic financial-services framing.

## Dataset
IBM AML **HI-Small** (`HI-Small_Trans.csv`, `HI-Small_accounts.csv`, `HI-Small_Patterns.txt`)
- 5,078,345 transactions across 518,581 accounts at thousands of banks (multi-bank ecosystem).
- 0.10% laundering (extreme imbalance - see RUNBOOK for how to talk about metrics).
- Sep 1-18 2022; the Sep-11+ "all-laundering tail" is trimmed (see RUNBOOK "Marco quirk").

## Object inventory (all in `FRAUD_RT_DEMO`)
- `RAW.TRANSACTIONS`, `RAW.ACCOUNTS`, `RAW.LAUNDERING_PATTERNS` (reserved for a future graph chapter)
- `CURATED.BANK_DIM`, `ACCOUNT_DIM`, `TXN_EVENTS`, `ACCOUNT_HISTORY`, `TXN_SPINE`
- `FEATURE_STORE`: entity `ACCOUNT`, FVs `ACCOUNT_PROFILE` / `ACCOUNT_VELOCITY` / `TXN_RISK_SIGNALS`,
  feature group `FRAUD_FEATURES`, Postgres online service, model `AML_FRAUD_GBM`, service `AML_FRAUD_RT_SERVICE`

## Prerequisites
1. Python env: `.venv` (Python 3.10) with `snowflake-ml-python>=1.41`, scikit-learn, etc. (already created).
2. **Connection (build steps):** `SNOWFLAKE_CONNECTION_NAME=demo156_keypair` — key-pair auth, non-interactive (no browser prompts). All `setup/*` and the FV-registration script use this.
3. **PAT (live demo steps - REQUIRED):** the Online Feature Store read path and REST ingest/query/inference endpoints all require a Programmatic Access Token. Key-pair does NOT satisfy the online service; the SDK explicitly needs `SNOWFLAKE_PAT`.
   ```
   export SNOWFLAKE_PAT="<token>"
   ```
   Create in Snowsight: *Profile -> Settings -> Authentication -> Programmatic access tokens*.

**Deployed inference endpoint:** `https://ey4lo-sfsenorthamerica-demo156.snowflakecomputing.app/predict`

## Build order (pre-demo setup)
```
export SNOWFLAKE_CONNECTION_NAME=demo156_keypair
.venv/bin/python setup/01_load_data.py            # load + curate (idempotent)
.venv/bin/python setup/02_feature_store.py        # FS + Postgres online service + FVs (slow: provisions PG)
.venv/bin/python setup/03_train_register_model.py # point-in-time training set + HistGradientBoosting + registry
.venv/bin/python setup/04_deploy_inference_service.py  # SPCS real-time inference endpoint
```

## Live demo
```
export SNOWFLAKE_CONNECTION_NAME=demo156_keypair
export SNOWFLAKE_PAT="<token>"      # REQUIRED for all online reads + REST below
.venv/bin/python demo/register_realtime_fv.py                       # register the real-time FV live
.venv/bin/python demo/stream_events.py --mode fraud --account <ID>  # fan-out velocity burst
.venv/bin/python demo/query_features.py --account <ID>              # prove <2s freshness
.venv/bin/python demo/score_transaction.py --account <ID> --scenario fraud  # score + latency
```

See **RUNBOOK.md** for the presenter script, how to talk about the model, and the typology cheat-sheet.
