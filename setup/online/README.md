# Optional: Real-time / Online path (Postgres-backed Online Feature Store)

**This path is OPTIONAL and is NOT part of the default batch demo.**

It stands up the Postgres-backed Online Feature Store (stream ingest, continuous
time-windowed aggregations, REST query/ingest) plus an SPCS real-time inference
service. It runs **24/7 (Postgres HA)** and therefore **accrues cost even when
idle** on a demo account. Only enable it when you specifically want to show the
real-time serving story, and tear it down afterwards.

## Enable
```
SNOWFLAKE_CONNECTION_NAME=demo156_keypair ML_ENV=dev SNOWFLAKE_PAT=... \
  .venv/bin/python setup/online/setup_online.py
```
Requirements:
- `SNOWFLAKE_PAT` env var (online reads + REST hard-require a PAT; key-pair JWT
  does not satisfy the online service).
- The feature-store schema must be **owned** by the running role
  (`create_online_service` requires schema OWNERSHIP). Grant if needed:
  `GRANT OWNERSHIP ON SCHEMA <db>.FEATURE_STORE TO ROLE ML_DEV_ROLE COPY CURRENT GRANTS;`

## Tear down (stop the cost)
`setup/99_reset_teardown.py --yes` drops the online service (and the whole
sandbox), with an orphaned-Postgres-instance safety net. See that script.

## What it creates
- `ACCOUNT_PROFILE` online (OnlineStoreType.POSTGRES) - latest profile per account.
- `TRANSACTION_EVENTS` stream source + `ACCOUNT_VELOCITY` stream FV (CONTINUOUS
  time-windowed aggregations) backfilled from prod `CURATED.TXN_EVENTS`.
- The Postgres online service (ingest + query REST endpoints).

For the real-time inference SPCS service and the REST demo notebook, see the
original real-time build (git history) - those are compatible once the online
service is running.
