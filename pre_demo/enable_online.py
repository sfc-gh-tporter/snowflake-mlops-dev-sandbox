"""OPTIONAL online branch - enable + verify the Postgres online feature store.

RUN THE DAY BEFORE a demo that will show real-time serving. The online service
provisions slowly (minutes to ~an hour on some accounts) and bills 24/7 while it
exists, so never stand it up live - warm it ahead, verify, then leave it running.
Tear it down afterwards with reset/teardown.py.

Guarded: prints a cost warning and only provisions with --yes.

  SNOWFLAKE_CONNECTION_NAME=demo156_keypair ML_ENV=dev SNOWFLAKE_PAT=... \
    .venv/bin/python pre_demo/enable_online.py --yes
"""

import argparse
import os
import runpy
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from snowpark_session import create_snowpark_session

SETUP_ONLINE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "setup", "online", "setup_online.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="actually provision the online service")
    ap.add_argument("--account", default="012719_8019E5AE0", help="account for the smoke read")
    args = ap.parse_args()

    print("=" * 70)
    print("OPTIONAL ONLINE FEATURE STORE - enable + verify")
    print("WARNING: the Postgres online service bills 24/7 and can take up to")
    print("~an hour to provision. Tear it down with reset/teardown.py afterwards.")
    print("=" * 70)
    if not args.yes:
        print("\nDry run - pass --yes to provision. Nothing changed.")
        return

    # Ensure the SDK online-read PAT is available.
    if "SNOWFLAKE_PAT" not in os.environ:
        try:
            os.environ["SNOWFLAKE_PAT"] = C.get_pat()
        except Exception:
            print("WARNING: no SNOWFLAKE_PAT - online read/REST will fail.")

    print("\n[1] Provisioning online feature store (setup/online/setup_online.py)")
    runpy.run_path(SETUP_ONLINE, run_name="__main__")

    print("\n[2] Verify: poll online service + smoke read")
    from snowflake.ml.feature_store import FeatureStore, CreationMode, StoreType
    s = create_snowpark_session()
    s.sql(f"USE ROLE {C.DEV_ROLE}").collect()
    s.sql(f"USE SCHEMA {C.DATABASE}.{C.FEATURE_STORE_SCHEMA}").collect()
    fs = FeatureStore(session=s, database=C.DATABASE, name=C.FEATURE_STORE_SCHEMA,
                      default_warehouse=C.WAREHOUSE, creation_mode=CreationMode.FAIL_IF_NOT_EXIST)
    waited = 0
    while getattr(fs.get_online_service_status(), "status", None) != "RUNNING":
        time.sleep(30); waited += 30
        print(f"    [{waited}s] provisioning...")
        if waited > 3600:
            print("    still not RUNNING after 1h; check the account"); break
    try:
        fv = fs.get_feature_view(C.FV_PROFILE, C.FV_VERSION)
        t0 = time.time()
        df = fs.read_feature_view(fv, keys=[[args.account]], store_type=StoreType.ONLINE)
        print(f"    smoke read OK in {(time.time()-t0)*1000:.0f} ms: {df.shape if hasattr(df,'shape') else 'ok'}")
    except Exception as e:
        print(f"    smoke read failed: {type(e).__name__}: {str(e)[:140]}")
    print("\nOnline feature store READY. Remember to tear it down after the demo.")
    s.close()


if __name__ == "__main__":
    main()
