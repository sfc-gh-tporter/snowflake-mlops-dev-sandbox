"""Demo - Stream transaction events to the Online Feature Store Ingest API.

Pushes events (matching the TRANSACTION_EVENTS stream schema) over the REST
Ingest API. Continuous aggregation in the ACCOUNT_VELOCITY feature view updates
within ~2 seconds, so query_features.py will show counters move.

Modes:
  normal : steady mix of ordinary payments across random accounts
  fraud  : a fan-out burst - ONE sender pushes many high-value, cross-currency,
           high-risk-format payments to many different banks (the single-account
           shadow of a fan-out / structuring laundering typology)

Usage:
  export SNOWFLAKE_PAT=...
  .venv/bin/python demo/stream_events.py --mode fraud --account <ACCOUNT_ID> --count 40
  .venv/bin/python demo/stream_events.py --mode normal --count 200 --rate 20
"""

import argparse
import os
import random
import sys
import time
from datetime import datetime, timedelta

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from snowpark_session import create_snowpark_session

from snowflake.ml.feature_store import FeatureStore, CreationMode, online_service

CURRENCIES = ["US Dollar", "Euro", "Yuan", "UK Pound", "Yen", "Rupee", "Bitcoin"]
HIGH_RISK = ["Bitcoin", "Wire", "Cash"]


def get_ingest_url(fs):
    status = fs.get_online_service_status()
    return online_service.endpoint_url(status, "ingest")


def random_account():
    return f"{random.randint(1, 30000):05d}_{random.randint(0, 9_000_000_000):010X}"


def make_event(account_id, fraud=False, seq=0):
    # Unique sub-second timestamp per event so the Ingest API (which dedups on
    # entity key + timestamp) counts each one distinctly.
    now = (datetime.utcnow() + timedelta(milliseconds=seq)).strftime("%Y-%m-%d %H:%M:%S.%f")
    if fraud:
        amount = round(random.uniform(50_000, 900_000), 2)
        cross = 1 if random.random() < 0.7 else 0
        high_risk = 1
    else:
        amount = round(random.uniform(20, 8_000), 2)
        cross = 1 if random.random() < 0.05 else 0
        high_risk = 1 if random.random() < 0.1 else 0
    return {
        C.ENTITY_JOIN_KEY: account_id,
        "EVENT_TS": now,
        "AMOUNT_PAID": amount,
        "RECEIVER_ACCOUNT_ID": random_account(),
        "RECEIVER_BANK": f"{random.randint(1, 250000)}",  # many distinct banks
        "IS_CROSS_CURRENCY": cross,
        "IS_HIGH_RISK_FORMAT": high_risk,
    }


def post_batch(ingest_url, pat, records):
    resp = requests.post(
        f"{ingest_url}/api/v1/ingest",
        headers={"Authorization": f'Snowflake Token="{pat}"',
                 "Content-Type": "application/json"},
        json={"dry_run": False, "records": {C.STREAM_SOURCE: records}},
        timeout=30,
    )
    return resp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["normal", "fraud"], default="normal")
    ap.add_argument("--account", default=None, help="sender account for fraud burst")
    ap.add_argument("--count", type=int, default=50)
    ap.add_argument("--rate", type=float, default=10, help="events per second")
    ap.add_argument("--batch", type=int, default=10)
    args = ap.parse_args()

    pat = C.get_pat()
    fs = create_fs()
    ingest_url = get_ingest_url(fs)
    print(f"Ingest endpoint: {ingest_url}")

    account = args.account or random_account()
    if args.mode == "fraud":
        print(f"FRAUD fan-out burst from sender {account}: "
              f"{args.count} high-value cross-border payments to many banks")
    else:
        print(f"NORMAL traffic: {args.count} events across random accounts")

    sent = 0
    buf = []
    for i in range(args.count):
        acct = account if args.mode == "fraud" else random_account()
        buf.append(make_event(acct, fraud=(args.mode == "fraud"), seq=i))
        if len(buf) >= args.batch:
            r = post_batch(ingest_url, pat, buf)
            sent += len(buf)
            print(f"  sent {sent}/{args.count}  http={r.status_code}")
            buf = []
            time.sleep(len(buf) / args.rate if args.rate else 0)
        time.sleep(1.0 / args.rate if args.rate else 0)
    if buf:
        r = post_batch(ingest_url, pat, buf)
        sent += len(buf)
        print(f"  sent {sent}/{args.count}  http={r.status_code}")
    print(f"Done. Target account for scoring: {account}")


def create_fs():
    session = create_snowpark_session()
    session.sql(f"USE WAREHOUSE {C.WAREHOUSE}").collect()
    session.sql(f"USE SCHEMA {C.DATABASE}.{C.FEATURE_STORE_SCHEMA}").collect()
    return FeatureStore(session=session, database=C.DATABASE,
                        name=C.FEATURE_STORE_SCHEMA, default_warehouse=C.WAREHOUSE,
                        creation_mode=CreationMode.FAIL_IF_NOT_EXIST)


if __name__ == "__main__":
    main()
