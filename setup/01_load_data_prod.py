"""Phase 2 - Load IBM AML HI-Small into ML_FRAUD_PRODUCTION (the prod data).

Produces the CLEAN PRODUCTION data a data scientist reads from:
  RAW.TRANSACTIONS, RAW.ACCOUNTS, RAW.LAUNDERING_PATTERNS
  CURATED.BANK_DIM, CURATED.ACCOUNT_DIM
  CURATED.TXN_EVENTS       - sender-side enriched events (clean Sep 1-10 window)
  CURATED.ACCOUNT_HISTORY  - per-account lifetime aggregates (profile source)

The LABELED analytical base table (spine + request context + split) is NOT built
here - that is the data scientist's selective transform (transforms/base_features.py,
Phase 3), authored in the dev sandbox reading this prod data.

Run as an admin, targeting prod:
  SNOWFLAKE_CONNECTION_NAME=<your-connection> ML_ENV=prod .venv/bin/python setup/01_load_data_prod.py
"""

import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from snowpark_session import create_snowpark_session

DB = C.PROD_DATABASE  # this loader always targets prod, regardless of ML_ENV


def run(session, sql, label=None):
    if label:
        print(f"  - {label}")
    return session.sql(sql).collect()


def main():
    session = create_snowpark_session()
    run(session, "USE ROLE ACCOUNTADMIN")
    run(session, f"USE WAREHOUSE {C.WAREHOUSE}")
    print("Loading prod data into", DB)

    run(session, f"CREATE SCHEMA IF NOT EXISTS {DB}.{C.RAW_SCHEMA}")
    run(session, f"CREATE SCHEMA IF NOT EXISTS {DB}.{C.CURATED_SCHEMA}")
    run(session, f"CREATE STAGE IF NOT EXISTS {C.STAGE_RAW}", "stage")
    run(session, f"USE SCHEMA {DB}.{C.RAW_SCHEMA}")

    # --- 1. Landing tables --------------------------------------------------
    run(session, f"""
        CREATE TABLE IF NOT EXISTS {C.TBL_TRANS} (
            TXN_TIMESTAMP STRING, FROM_BANK STRING, FROM_ACCOUNT STRING,
            TO_BANK STRING, TO_ACCOUNT STRING, AMOUNT_RECEIVED FLOAT,
            RECEIVING_CURRENCY STRING, AMOUNT_PAID FLOAT, PAYMENT_CURRENCY STRING,
            PAYMENT_FORMAT STRING, IS_LAUNDERING NUMBER(1))""", "table TRANSACTIONS")
    run(session, f"""
        CREATE TABLE IF NOT EXISTS {C.TBL_ACCOUNTS} (
            BANK_NAME STRING, BANK_ID STRING, ACCOUNT_NUMBER STRING,
            ENTITY_ID STRING, ENTITY_NAME STRING)""", "table ACCOUNTS")

    csv_fmt = ("TYPE=CSV SKIP_HEADER=1 FIELD_OPTIONALLY_ENCLOSED_BY='\"' "
               "EMPTY_FIELD_AS_NULL=TRUE")

    # --- 2. PUT + COPY (idempotent) ----------------------------------------
    expected = {C.TBL_TRANS: 5_000_000, C.TBL_ACCOUNTS: 500_000}
    for path, table in ((C.FILE_TRANS, C.TBL_TRANS), (C.FILE_ACCOUNTS, C.TBL_ACCOUNTS)):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Source file not found: {path}")
        already = run(session, f"SELECT COUNT(*) C FROM {table}")[0]["C"]
        if already >= expected[table]:
            print(f"  - {os.path.basename(path)} already loaded ({already:,}) - skip")
            continue
        print(f"  - uploading {os.path.basename(path)} ...")
        session.file.put(f"file://{path}", f"@{C.STAGE_RAW}", auto_compress=True, overwrite=True)
        fname = os.path.basename(path)
        run(session, f"""COPY INTO {table} FROM @{C.STAGE_RAW}/{fname}.gz
            FILE_FORMAT=({csv_fmt}) ON_ERROR=ABORT_STATEMENT""", f"COPY {fname}")

    # --- 3. Patterns -> RAW.LAUNDERING_PATTERNS ----------------------------
    print("  - parsing patterns file")
    pat_rows, pid, ptype = [], 0, None
    with open(C.FILE_PATTERNS) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("BEGIN LAUNDERING ATTEMPT"):
                pid += 1
                m = re.search(r"-\s*([A-Za-z\- ]+?)(?::|$)", line)
                ptype = m.group(1).strip().upper() if m else "UNKNOWN"
            elif line.startswith("END LAUNDERING ATTEMPT"):
                ptype = None
            elif ptype and "," in line:
                p = line.split(",")
                if len(p) >= 11:
                    pat_rows.append({"PATTERN_ID": pid, "PATTERN_TYPE": ptype,
                                     "EVENT_TS": p[0], "FROM_ACCOUNT_ID": f"{p[1]}_{p[2]}",
                                     "TO_ACCOUNT_ID": f"{p[3]}_{p[4]}",
                                     "AMOUNT_PAID": float(p[7]), "PAYMENT_CURRENCY": p[8]})
    session.write_pandas(pd.DataFrame(pat_rows), "LAUNDERING_PATTERNS",
                         database=DB, schema=C.RAW_SCHEMA, auto_create_table=True, overwrite=True)
    print(f"    parsed {len(pat_rows)} pattern txns across {pid} subgraphs")

    # --- 4. Curated dims ----------------------------------------------------
    run(session, f"USE SCHEMA {DB}.{C.CURATED_SCHEMA}")
    run(session, f"""
        CREATE OR REPLACE TABLE {DB}.{C.CURATED_SCHEMA}.BANK_DIM AS
        SELECT BANK_ID, ANY_VALUE(BANK_NAME) AS BANK_NAME,
            CASE WHEN ANY_VALUE(BANK_NAME) RLIKE '^.+ Bank #[0-9]+$'
                 THEN UPPER(REGEXP_SUBSTR(ANY_VALUE(BANK_NAME),'^(.*) Bank #[0-9]+$',1,1,'e',1))
                 ELSE 'US' END AS BANK_COUNTRY_RAW
        FROM {C.TBL_ACCOUNTS} GROUP BY BANK_ID""", "BANK_DIM")
    run(session, f"""UPDATE {DB}.{C.CURATED_SCHEMA}.BANK_DIM
        SET BANK_COUNTRY_RAW='CRYPTO' WHERE BANK_COUNTRY_RAW='CRYTPO'""")
    run(session, f"""
        CREATE OR REPLACE TABLE {DB}.{C.CURATED_SCHEMA}.BANK_DIM AS
        SELECT BANK_ID, BANK_NAME, COALESCE(NULLIF(BANK_COUNTRY_RAW,''),'US') AS BANK_COUNTRY
        FROM {DB}.{C.CURATED_SCHEMA}.BANK_DIM""")
    run(session, f"""
        CREATE OR REPLACE TABLE {DB}.{C.CURATED_SCHEMA}.ACCOUNT_DIM AS
        SELECT a.BANK_ID || '_' || a.ACCOUNT_NUMBER AS {C.ENTITY_JOIN_KEY},
            a.BANK_ID, b.BANK_COUNTRY, a.ENTITY_ID,
            TRIM(REGEXP_REPLACE(a.ENTITY_NAME, '\\\\s*#[0-9]+$','')) AS ENTITY_TYPE
        FROM {C.TBL_ACCOUNTS} a
        LEFT JOIN {DB}.{C.CURATED_SCHEMA}.BANK_DIM b ON a.BANK_ID=b.BANK_ID""", "ACCOUNT_DIM")

    # --- 5. Clean sender-side events (Sep 1-10 window) ---------------------
    run(session, f"""
        CREATE OR REPLACE TABLE {DB}.{C.CURATED_SCHEMA}.TXN_EVENTS AS
        SELECT
            t.FROM_BANK || '_' || t.FROM_ACCOUNT AS {C.ENTITY_JOIN_KEY},
            t.TO_BANK   || '_' || t.TO_ACCOUNT   AS RECEIVER_ACCOUNT_ID,
            t.TO_BANK AS RECEIVER_BANK,
            TO_TIMESTAMP_NTZ(t.TXN_TIMESTAMP,'YYYY/MM/DD HH24:MI') AS EVENT_TS,
            t.AMOUNT_PAID, t.PAYMENT_CURRENCY, t.RECEIVING_CURRENCY, t.PAYMENT_FORMAT,
            IFF(t.RECEIVING_CURRENCY <> t.PAYMENT_CURRENCY,1,0) AS IS_CROSS_CURRENCY,
            IFF(t.FROM_BANK <> t.TO_BANK,1,0) AS IS_CROSS_BANK,
            IFF(t.PAYMENT_FORMAT IN ('Bitcoin','Wire','Cash'),1,0) AS IS_HIGH_RISK_FORMAT,
            sb.BANK_COUNTRY AS FROM_COUNTRY, rb.BANK_COUNTRY AS TO_COUNTRY,
            IFF(COALESCE(sb.BANK_COUNTRY,'US') <> COALESCE(rb.BANK_COUNTRY,'US'),1,0) AS IS_CROSS_BORDER,
            t.IS_LAUNDERING
        FROM {C.TBL_TRANS} t
        LEFT JOIN {DB}.{C.CURATED_SCHEMA}.BANK_DIM sb ON t.FROM_BANK=sb.BANK_ID
        LEFT JOIN {DB}.{C.CURATED_SCHEMA}.BANK_DIM rb ON t.TO_BANK=rb.BANK_ID
        WHERE TO_TIMESTAMP_NTZ(t.TXN_TIMESTAMP,'YYYY/MM/DD HH24:MI') < '{C.TAIL_CUTOFF_DATE}'::TIMESTAMP_NTZ
        """, "TXN_EVENTS (clean window)")

    # --- 6. Account history (profile source) -------------------------------
    run(session, f"""
        CREATE OR REPLACE TABLE {C.TBL_ACCOUNT_HISTORY} AS
        SELECT e.{C.ENTITY_JOIN_KEY},
            COUNT(*) AS HIST_TXN_COUNT, AVG(e.AMOUNT_PAID) AS HIST_AVG_AMOUNT,
            COALESCE(STDDEV(e.AMOUNT_PAID),0) AS HIST_STD_AMOUNT, MAX(e.AMOUNT_PAID) AS HIST_MAX_AMOUNT,
            COUNT(DISTINCT e.RECEIVER_ACCOUNT_ID) AS HIST_DISTINCT_RECEIVERS,
            COUNT(DISTINCT e.RECEIVER_BANK) AS HIST_DISTINCT_RECEIVER_BANKS,
            COUNT(DISTINCT e.TO_COUNTRY) AS HIST_DISTINCT_COUNTRIES,
            AVG(e.IS_CROSS_CURRENCY) AS HIST_FOREIGN_CCY_SHARE,
            AVG(e.IS_HIGH_RISK_FORMAT) AS HIST_HIGH_RISK_SHARE,
            MAX(e.EVENT_TS) AS LAST_ACTIVITY_TS,
            ANY_VALUE(d.ENTITY_TYPE) AS ENTITY_TYPE, ANY_VALUE(d.BANK_COUNTRY) AS HOME_COUNTRY
        FROM {DB}.{C.CURATED_SCHEMA}.TXN_EVENTS e
        LEFT JOIN {DB}.{C.CURATED_SCHEMA}.ACCOUNT_DIM d ON e.{C.ENTITY_JOIN_KEY}=d.{C.ENTITY_JOIN_KEY}
        GROUP BY e.{C.ENTITY_JOIN_KEY}""", "ACCOUNT_HISTORY")

    # --- 7. Verify ----------------------------------------------------------
    print("\nVerification:")
    n_t = run(session, f"SELECT COUNT(*) C FROM {C.TBL_TRANS}")[0]["C"]
    n_a = run(session, f"SELECT COUNT(*) C FROM {C.TBL_ACCOUNTS}")[0]["C"]
    ev = run(session, f"""SELECT COUNT(*) N, SUM(IS_LAUNDERING) POS,
             MIN(EVENT_TS) MN, MAX(EVENT_TS) MX FROM {DB}.{C.CURATED_SCHEMA}.TXN_EVENTS""")[0]
    n_h = run(session, f"SELECT COUNT(*) C FROM {C.TBL_ACCOUNT_HISTORY}")[0]["C"]
    print(f"  RAW.TRANSACTIONS: {n_t:,} (expect ~5,078,345)")
    print(f"  RAW.ACCOUNTS:     {n_a:,} (expect ~518,581)")
    print(f"  TXN_EVENTS: {ev['N']:,}  positives: {ev['POS']:,} ({100*ev['POS']/ev['N']:.3f}%)")
    print(f"  TXN_EVENTS range: {ev['MN']} -> {ev['MX']} (max must be < {C.TAIL_CUTOFF_DATE})")
    print(f"  ACCOUNT_HISTORY accounts: {n_h:,}")
    assert str(ev["MX"]) < C.TAIL_CUTOFF_DATE, "Tail not trimmed!"
    print("\nProd data load complete.")
    session.close()


if __name__ == "__main__":
    main()
