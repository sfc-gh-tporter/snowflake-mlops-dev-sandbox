"""Demo - Query online features via the REST Query API (freshness check).

Run this right after stream_events.py to show the ACCOUNT_VELOCITY counters move
within ~2 seconds of ingestion.

Usage:
  export SNOWFLAKE_PAT=...
  .venv/bin/python demo/query_features.py --account <ACCOUNT_ID>
"""

import argparse
import os
import sys

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from snowpark_session import create_snowpark_session

from snowflake.ml.feature_store import FeatureStore, CreationMode, online_service


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--account", required=True)
    ap.add_argument("--view", default=C.FV_VELOCITY,
                    help="feature view to query (default ACCOUNT_VELOCITY)")
    args = ap.parse_args()

    pat = C.get_pat()
    session = create_snowpark_session()
    session.sql(f"USE WAREHOUSE {C.WAREHOUSE}").collect()
    session.sql(f"USE SCHEMA {C.DATABASE}.{C.FEATURE_STORE_SCHEMA}").collect()
    fs = FeatureStore(session=session, database=C.DATABASE,
                      name=C.FEATURE_STORE_SCHEMA, default_warehouse=C.WAREHOUSE,
                      creation_mode=CreationMode.FAIL_IF_NOT_EXIST)
    status = fs.get_online_service_status()
    query_url = online_service.endpoint_url(status, "query")

    resp = requests.post(
        f"{query_url}/api/v1/query",
        headers={"Authorization": f'Snowflake Token="{pat}"',
                 "Content-Type": "application/json"},
        json={
            "name": args.view,
            "version": C.FV_VERSION,
            "object_type": "feature_view",
            "request_rows": [{"entity": {C.ENTITY_JOIN_KEY: args.account}}],
        },
        timeout=30,
    )
    print(f"HTTP {resp.status_code}")
    print(resp.json())
    session.close()


if __name__ == "__main__":
    main()
