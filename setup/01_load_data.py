"""Setup 01 - Load IBM AML HI-Small into FRAUD_RT_DEMO and build curated tables.

Produces:
  RAW.TRANSACTIONS          - raw transactions (duplicate `Account` headers split
                              into FROM_ACCOUNT / TO_ACCOUNT)
  RAW.ACCOUNTS              - account/entity/bank dimension
  RAW.LAUNDERING_PATTERNS   - parsed typology subgraphs (reserved, Appendix A)
  CURATED.BANK_DIM          - bank_id -> name, country
  CURATED.ACCOUNT_DIM       - composite ACCOUNT_ID -> bank/country/entity_type
  CURATED.TXN_EVENTS        - sender-side enriched events (clean Sep 1-10 window)
                              source for the velocity FV + stream schema + spine
  CURATED.ACCOUNT_HISTORY   - per-account lifetime aggregates (profile FV source)
  CURATED.TXN_SPINE         - labeled spine w/ request-context features + split

Key handling:
  * ACCOUNT_ID is composite (BANK_ID || '_' || ACCOUNT_NUMBER) - account numbers
    are NOT globally unique across banks.
  * Country parsed from bank name '<Country> Bank #N' (~61%); named banks -> US.
  * "Marco quirk": rows on/after TAIL_CUTOFF_DATE (Sep 11) are the trailing
    all-laundering tail; trimmed before building features/spine.
  * Laundering_type / patterns NEVER flow into feature or spine tables.

Run:  SNOWFLAKE_CONNECTION_NAME=<conn> .venv/bin/python setup/01_load_data.py
"""

import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from snowpark_session import create_snowpark_session


def run(session, sql, label=None):
    if label:
        print(f"  - {label}")
    return session.sql(sql).collect()


def main():
    session = create_snowpark_session()
    print("Connected. Building", C.DATABASE)

    # --- 0. Database / schemas / stage / warehouse --------------------------
    run(session, f"USE WAREHOUSE {C.WAREHOUSE}")
    run(session, f"CREATE DATABASE IF NOT EXISTS {C.DATABASE}", "database")
    for sch in (C.RAW_SCHEMA, C.CURATED_SCHEMA, C.FEATURE_STORE_SCHEMA):
        run(session, f"CREATE SCHEMA IF NOT EXISTS {C.DATABASE}.{sch}", f"schema {sch}")
    run(session, f"CREATE STAGE IF NOT EXISTS {C.STAGE_RAW}", "stage")
    run(session, f"USE SCHEMA {C.DATABASE}.{C.RAW_SCHEMA}")

    # --- 1. Landing tables (explicit cols; dup `Account` -> FROM/TO) --------
    run(session, f"""
        CREATE TABLE IF NOT EXISTS {C.TBL_TRANS} (
            TXN_TIMESTAMP      STRING,
            FROM_BANK          STRING,
            FROM_ACCOUNT       STRING,
            TO_BANK            STRING,
            TO_ACCOUNT         STRING,
            AMOUNT_RECEIVED    FLOAT,
            RECEIVING_CURRENCY STRING,
            AMOUNT_PAID        FLOAT,
            PAYMENT_CURRENCY   STRING,
            PAYMENT_FORMAT     STRING,
            IS_LAUNDERING      NUMBER(1)
        )""", "table TRANSACTIONS")
    run(session, f"""
        CREATE TABLE IF NOT EXISTS {C.TBL_ACCOUNTS} (
            BANK_NAME      STRING,
            BANK_ID        STRING,
            ACCOUNT_NUMBER STRING,
            ENTITY_ID      STRING,
            ENTITY_NAME    STRING
        )""", "table ACCOUNTS")

    csv_fmt = ("TYPE=CSV SKIP_HEADER=1 FIELD_OPTIONALLY_ENCLOSED_BY='\"' "
               "EMPTY_FIELD_AS_NULL=TRUE")

    # --- 2. PUT + COPY the two CSVs (idempotent: skip if already loaded) ----
    expected = {C.TBL_TRANS: 5_000_000, C.TBL_ACCOUNTS: 500_000}
    for path, table in ((C.FILE_TRANS, C.TBL_TRANS), (C.FILE_ACCOUNTS, C.TBL_ACCOUNTS)):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Source file not found: {path}")
        already = run(session, f"SELECT COUNT(*) C FROM {table}")[0]["C"]
        if already >= expected[table]:
            print(f"  - {os.path.basename(path)} already loaded ({already:,} rows) - skipping upload")
            continue
        print(f"  - uploading {os.path.basename(path)} ...")
        session.file.put(f"file://{path}", f"@{C.STAGE_RAW}",
                         auto_compress=True, overwrite=True)
        fname = os.path.basename(path)
        run(session, f"""
            COPY INTO {table}
            FROM @{C.STAGE_RAW}/{fname}.gz
            FILE_FORMAT=({csv_fmt})
            ON_ERROR=ABORT_STATEMENT""", f"COPY {fname}")

    # --- 3. Parse patterns.txt locally -> RAW.LAUNDERING_PATTERNS ----------
    print("  - parsing patterns file")
    pat_rows = []
    pid = 0
    ptype = None
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
                    pat_rows.append({
                        "PATTERN_ID": pid,
                        "PATTERN_TYPE": ptype,
                        "EVENT_TS": p[0],
                        "FROM_ACCOUNT_ID": f"{p[1]}_{p[2]}",
                        "TO_ACCOUNT_ID": f"{p[3]}_{p[4]}",
                        "AMOUNT_PAID": float(p[7]),
                        "PAYMENT_CURRENCY": p[8],
                    })
    pat_df = pd.DataFrame(pat_rows)
    session.write_pandas(pat_df, "LAUNDERING_PATTERNS",
                         database=C.DATABASE, schema=C.RAW_SCHEMA,
                         auto_create_table=True, overwrite=True)
    print(f"    parsed {len(pat_df)} pattern transactions across {pid} subgraphs")

    # --- 4. Curated dimensions ---------------------------------------------
    run(session, f"USE SCHEMA {C.DATABASE}.{C.CURATED_SCHEMA}")

    # Bank dim: country from '<Country> Bank #N', 'Crytpo'->CRYPTO, else US.
    run(session, f"""
        CREATE OR REPLACE TABLE {C.DATABASE}.{C.CURATED_SCHEMA}.BANK_DIM AS
        SELECT
            BANK_ID,
            ANY_VALUE(BANK_NAME) AS BANK_NAME,
            CASE
              WHEN ANY_VALUE(BANK_NAME) RLIKE '^.+ Bank #[0-9]+$'
                THEN UPPER(REGEXP_SUBSTR(ANY_VALUE(BANK_NAME),
                                         '^(.*) Bank #[0-9]+$', 1, 1, 'e', 1))
              ELSE 'US'
            END AS BANK_COUNTRY_RAW
        FROM {C.TBL_ACCOUNTS}
        GROUP BY BANK_ID""", "BANK_DIM")
    run(session, f"""
        UPDATE {C.DATABASE}.{C.CURATED_SCHEMA}.BANK_DIM
        SET BANK_COUNTRY_RAW='CRYPTO' WHERE BANK_COUNTRY_RAW='CRYTPO'""")
    run(session, f"""
        CREATE OR REPLACE TABLE {C.DATABASE}.{C.CURATED_SCHEMA}.BANK_DIM AS
        SELECT BANK_ID, BANK_NAME,
               COALESCE(NULLIF(BANK_COUNTRY_RAW,''),'US') AS BANK_COUNTRY
        FROM {C.DATABASE}.{C.CURATED_SCHEMA}.BANK_DIM""")

    # Account dim: composite id, entity type from '<Type> #NNNN'.
    run(session, f"""
        CREATE OR REPLACE TABLE {C.DATABASE}.{C.CURATED_SCHEMA}.ACCOUNT_DIM AS
        SELECT
            a.BANK_ID || '_' || a.ACCOUNT_NUMBER AS {C.ENTITY_JOIN_KEY},
            a.BANK_ID,
            b.BANK_COUNTRY,
            a.ENTITY_ID,
            TRIM(REGEXP_REPLACE(a.ENTITY_NAME, '\\\\s*#[0-9]+$', '')) AS ENTITY_TYPE
        FROM {C.TBL_ACCOUNTS} a
        LEFT JOIN {C.DATABASE}.{C.CURATED_SCHEMA}.BANK_DIM b ON a.BANK_ID=b.BANK_ID
        """, "ACCOUNT_DIM")

    # --- 5. Sender-side enriched events (clean window only) ----------------
    run(session, f"""
        CREATE OR REPLACE TABLE {C.DATABASE}.{C.CURATED_SCHEMA}.TXN_EVENTS AS
        SELECT
            t.FROM_BANK || '_' || t.FROM_ACCOUNT AS {C.ENTITY_JOIN_KEY},
            t.TO_BANK   || '_' || t.TO_ACCOUNT   AS RECEIVER_ACCOUNT_ID,
            t.TO_BANK                             AS RECEIVER_BANK,
            TO_TIMESTAMP_NTZ(t.TXN_TIMESTAMP, 'YYYY/MM/DD HH24:MI') AS EVENT_TS,
            t.AMOUNT_PAID,
            t.PAYMENT_CURRENCY,
            t.RECEIVING_CURRENCY,
            t.PAYMENT_FORMAT,
            IFF(t.RECEIVING_CURRENCY <> t.PAYMENT_CURRENCY, 1, 0) AS IS_CROSS_CURRENCY,
            IFF(t.FROM_BANK <> t.TO_BANK, 1, 0)                   AS IS_CROSS_BANK,
            IFF(t.PAYMENT_FORMAT IN ('Bitcoin','Wire','Cash'), 1, 0) AS IS_HIGH_RISK_FORMAT,
            sb.BANK_COUNTRY AS FROM_COUNTRY,
            rb.BANK_COUNTRY AS TO_COUNTRY,
            IFF(COALESCE(sb.BANK_COUNTRY,'US') <> COALESCE(rb.BANK_COUNTRY,'US'), 1, 0)
                AS IS_CROSS_BORDER,
            t.IS_LAUNDERING
        FROM {C.TBL_TRANS} t
        LEFT JOIN {C.DATABASE}.{C.CURATED_SCHEMA}.BANK_DIM sb ON t.FROM_BANK=sb.BANK_ID
        LEFT JOIN {C.DATABASE}.{C.CURATED_SCHEMA}.BANK_DIM rb ON t.TO_BANK=rb.BANK_ID
        WHERE TO_TIMESTAMP_NTZ(t.TXN_TIMESTAMP,'YYYY/MM/DD HH24:MI')
              < '{C.TAIL_CUTOFF_DATE}'::TIMESTAMP_NTZ
        """, "TXN_EVENTS (clean window)")

    # --- 6. Account history (profile FV source) ----------------------------
    run(session, f"""
        CREATE OR REPLACE TABLE {C.TBL_ACCOUNT_HISTORY} AS
        SELECT
            e.{C.ENTITY_JOIN_KEY},
            COUNT(*)                                  AS HIST_TXN_COUNT,
            AVG(e.AMOUNT_PAID)                        AS HIST_AVG_AMOUNT,
            COALESCE(STDDEV(e.AMOUNT_PAID),0)         AS HIST_STD_AMOUNT,
            MAX(e.AMOUNT_PAID)                        AS HIST_MAX_AMOUNT,
            COUNT(DISTINCT e.RECEIVER_ACCOUNT_ID)     AS HIST_DISTINCT_RECEIVERS,
            COUNT(DISTINCT e.RECEIVER_BANK)           AS HIST_DISTINCT_RECEIVER_BANKS,
            COUNT(DISTINCT e.TO_COUNTRY)              AS HIST_DISTINCT_COUNTRIES,
            AVG(e.IS_CROSS_CURRENCY)                  AS HIST_FOREIGN_CCY_SHARE,
            AVG(e.IS_HIGH_RISK_FORMAT)                AS HIST_HIGH_RISK_SHARE,
            MAX(e.EVENT_TS)                           AS LAST_ACTIVITY_TS,
            ANY_VALUE(d.ENTITY_TYPE)                  AS ENTITY_TYPE,
            ANY_VALUE(d.BANK_COUNTRY)                 AS HOME_COUNTRY
        FROM {C.DATABASE}.{C.CURATED_SCHEMA}.TXN_EVENTS e
        LEFT JOIN {C.DATABASE}.{C.CURATED_SCHEMA}.ACCOUNT_DIM d
               ON e.{C.ENTITY_JOIN_KEY}=d.{C.ENTITY_JOIN_KEY}
        GROUP BY e.{C.ENTITY_JOIN_KEY}""", "ACCOUNT_HISTORY")

    # --- 7. Labeled spine w/ request-context features + chronological split -
    # PERCENTILE_CONT needs a numeric ORDER BY -> use epoch seconds, convert back.
    epochs = run(session, f"""
        SELECT
          TO_TIMESTAMP_NTZ(PERCENTILE_CONT({C.SPLIT_TRAIN_FRAC})
              WITHIN GROUP (ORDER BY DATE_PART(EPOCH_SECOND, EVENT_TS))) AS P_TRAIN,
          TO_TIMESTAMP_NTZ(PERCENTILE_CONT({C.SPLIT_TRAIN_FRAC + C.SPLIT_VAL_FRAC})
              WITHIN GROUP (ORDER BY DATE_PART(EPOCH_SECOND, EVENT_TS))) AS P_VAL
        FROM {C.DATABASE}.{C.CURATED_SCHEMA}.TXN_EVENTS""")
    p_train, p_val = epochs[0]["P_TRAIN"], epochs[0]["P_VAL"]
    print(f"    split thresholds: train<= {p_train}  val<= {p_val}")

    run(session, f"""
        CREATE OR REPLACE TABLE {C.TBL_TXN_SPINE} AS
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
              WHEN e.EVENT_TS <= '{p_train}'::TIMESTAMP_NTZ THEN 'TRAIN'
              WHEN e.EVENT_TS <= '{p_val}'::TIMESTAMP_NTZ   THEN 'VAL'
              ELSE 'TEST'
            END AS SPLIT
        FROM {C.DATABASE}.{C.CURATED_SCHEMA}.TXN_EVENTS e
        LEFT JOIN {C.TBL_ACCOUNT_HISTORY} h
               ON e.{C.ENTITY_JOIN_KEY}=h.{C.ENTITY_JOIN_KEY}""", "TXN_SPINE")

    # --- 8. Verification ----------------------------------------------------
    print("\nVerification:")
    n_trans = run(session, f"SELECT COUNT(*) C FROM {C.TBL_TRANS}")[0]["C"]
    n_acct = run(session, f"SELECT COUNT(*) C FROM {C.TBL_ACCOUNTS}")[0]["C"]
    spine = run(session, f"""
        SELECT COUNT(*) N, SUM(IS_LAUNDERING) POS,
               MIN(EVENT_TS) MN, MAX(EVENT_TS) MX
        FROM {C.TBL_TXN_SPINE}""")[0]
    split = run(session, f"""
        SELECT SPLIT, COUNT(*) N, SUM(IS_LAUNDERING) POS
        FROM {C.TBL_TXN_SPINE} GROUP BY SPLIT ORDER BY MIN(EVENT_TS)""")
    print(f"  RAW.TRANSACTIONS rows: {n_trans:,} (expect 5,078,345)")
    print(f"  RAW.ACCOUNTS rows:     {n_acct:,} (expect 518,581)")
    print(f"  SPINE rows: {spine['N']:,}  positives: {spine['POS']:,}  "
          f"rate: {100*spine['POS']/spine['N']:.3f}%")
    print(f"  SPINE EVENT_TS range: {spine['MN']} -> {spine['MX']}  "
          f"(max must be < {C.TAIL_CUTOFF_DATE})")
    for r in split:
        print(f"    {r['SPLIT']:<6} n={r['N']:,} pos={r['POS']:,}")
    assert str(spine["MX"]) < C.TAIL_CUTOFF_DATE, "Tail not trimmed!"
    print("\nSetup 01 complete.")
    session.close()


if __name__ == "__main__":
    main()
