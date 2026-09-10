"""The data scientist's selective transform: prod data -> analytical base table.

This is the DS's FIRST action. Rather than building feature views straight off
raw prod tables, they select/derive a clean, labeled analytical base table (ABT)
from production data. Feature views and training then read the ABT.

Env-aware and promotable:
  * Source data is ALWAYS production (ML_FRAUD_PRODUCTION.CURATED) - the single
    source of truth. Dev reads it read-only.
  * Target ABT is env-specific: dev writes ML_FRAUD_DEV_SANDBOX.CURATED.FRAUD_ABT;
    on promotion the SAME transform runs in prod writing the prod ABT.

Run standalone to (re)build the ABT for the current ML_ENV:
  SNOWFLAKE_CONNECTION_NAME=<your-connection> ML_ENV=dev  .venv/bin/python transforms/base_features.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C


def abt_select_sql() -> str:
    """The canonical selection/derivation applied to prod events (single source
    of truth for both dev and prod ABT builds). Split thresholds are injected."""
    src = C.PROD_DATABASE
    return f"""
        SELECT
            e.{C.ENTITY_JOIN_KEY},
            e.EVENT_TS,
            e.AMOUNT_PAID,
            e.PAYMENT_CURRENCY,
            e.RECEIVING_CURRENCY,
            e.PAYMENT_FORMAT,
            e.IS_CROSS_CURRENCY,
            e.IS_CROSS_BORDER,
            e.IS_HIGH_RISK_FORMAT,
            e.TO_COUNTRY,
            e.AMOUNT_PAID / NULLIF(h.HIST_AVG_AMOUNT, 0) AS AMOUNT_TO_AVG_RATIO,
            e.IS_LAUNDERING,
            CASE
              WHEN e.EVENT_TS <= '{{p_train}}'::TIMESTAMP_NTZ THEN 'TRAIN'
              WHEN e.EVENT_TS <= '{{p_val}}'::TIMESTAMP_NTZ   THEN 'VAL'
              ELSE 'TEST'
            END AS SPLIT
        FROM {src}.{C.CURATED_SCHEMA}.TXN_EVENTS e
        LEFT JOIN {src}.{C.CURATED_SCHEMA}.ACCOUNT_HISTORY h
               ON e.{C.ENTITY_JOIN_KEY} = h.{C.ENTITY_JOIN_KEY}
    """


def build_abt(session, env: str = None) -> str:
    """Build/refresh the analytical base table for `env` (dev or prod).

    Reads prod events + history, applies the selective transform, writes the
    env-specific FRAUD_ABT table. Returns the fully-qualified ABT name.
    """
    env = (env or C.ML_ENV).lower()
    src = C.PROD_DATABASE
    target = C.abt_fqn(env)

    # Chronological split thresholds (numeric ORDER BY via epoch seconds).
    row = session.sql(f"""
        SELECT
          TO_TIMESTAMP_NTZ(PERCENTILE_CONT({C.SPLIT_TRAIN_FRAC})
              WITHIN GROUP (ORDER BY DATE_PART(EPOCH_SECOND, EVENT_TS))) AS P_TRAIN,
          TO_TIMESTAMP_NTZ(PERCENTILE_CONT({C.SPLIT_TRAIN_FRAC + C.SPLIT_VAL_FRAC})
              WITHIN GROUP (ORDER BY DATE_PART(EPOCH_SECOND, EVENT_TS))) AS P_VAL
        FROM {src}.{C.CURATED_SCHEMA}.TXN_EVENTS
    """).collect()[0]
    p_train, p_val = row["P_TRAIN"], row["P_VAL"]

    select_sql = abt_select_sql().replace("{p_train}", str(p_train)).replace("{p_val}", str(p_val))
    session.sql(f"CREATE OR REPLACE TABLE {target} AS {select_sql}").collect()
    return target


def main():
    from snowpark_session import create_snowpark_session
    env = C.ML_ENV
    role = C.DEV_ROLE if env == "dev" else "ACCOUNTADMIN"  # prod build via deploy in CI
    s = create_snowpark_session()
    s.sql(f"USE ROLE {role}").collect()
    s.sql(f"USE WAREHOUSE {C.WAREHOUSE}").collect()
    print(f"Building ABT for env={env} as role={role} (source=prod, target={C.abt_fqn(env)})")
    target = build_abt(s, env)

    r = s.sql(f"""SELECT COUNT(*) N, SUM(IS_LAUNDERING) POS,
              COUNT(DISTINCT SPLIT) NSPLIT FROM {target}""").collect()[0]
    by = s.sql(f"""SELECT SPLIT, COUNT(*) N, SUM(IS_LAUNDERING) POS
              FROM {target} GROUP BY SPLIT ORDER BY MIN(EVENT_TS)""").collect()
    print(f"ABT {target}: {r['N']:,} rows, {r['POS']:,} positives "
          f"({100*r['POS']/r['N']:.3f}%), {r['NSPLIT']} splits")
    for x in by:
        print(f"  {x['SPLIT']:<6} n={x['N']:,} pos={x['POS']:,}")
    s.close()


if __name__ == "__main__":
    main()
