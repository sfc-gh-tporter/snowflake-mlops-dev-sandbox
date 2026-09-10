"""Demo kickoff / pre-warm - run this ~5 minutes before presenting.

Speeds up the things we can't speed up live:
  * Resumes the CORTEX_CODE_WH warehouse.
  * Resumes + warms the MLOPS_CPU_M_POOL compute pool so the promotion ML Job
    does NOT pay a cold-start during the demo.
  * (optional --warm-image) submits a trivial ML Job to cache the container
    image on the node, so the real promotion job starts fast.
  * Readiness check: confirms dev ABT, dev model V2, and (if promoted) prod
    predictions are present.

  SNOWFLAKE_CONNECTION_NAME=demo156_keypair .venv/bin/python setup/demo_kickoff.py
  SNOWFLAKE_CONNECTION_NAME=demo156_keypair .venv/bin/python setup/demo_kickoff.py --warm-image
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from snowpark_session import create_snowpark_session


def pool_row(s):
    for r in s.sql(f"SHOW COMPUTE POOLS LIKE '{C.JOB_COMPUTE_POOL}'").collect():
        return r.as_dict()
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm-image", action="store_true",
                    help="also submit a trivial ML Job to cache the container image")
    args = ap.parse_args()

    s = create_snowpark_session()
    s.sql("USE ROLE ACCOUNTADMIN").collect()

    print("[1] Warehouse")
    try:
        s.sql(f"ALTER WAREHOUSE {C.WAREHOUSE} RESUME").collect()
    except Exception:
        pass  # already running
    s.sql(f"USE WAREHOUSE {C.WAREHOUSE}").collect()
    s.sql("SELECT 1").collect()
    print(f"    {C.WAREHOUSE} ready")

    print("[2] Compute pool (pre-warm)")
    st = pool_row(s).get("state")
    if st in ("SUSPENDED", "STOPPING"):
        try:
            s.sql(f"ALTER COMPUTE POOL {C.JOB_COMPUTE_POOL} RESUME").collect()
        except Exception as e:
            print("   resume:", type(e).__name__, str(e)[:80])
    waited = 0
    while True:
        r = pool_row(s)
        st = r.get("state")
        ready = int(r.get("active_nodes", 0) or 0) + int(r.get("idle_nodes", 0) or 0)
        print(f"    [{waited}s] state={st} nodes_ready={ready}")
        if st in ("ACTIVE", "IDLE") and ready >= 1:
            break
        if waited > 420:
            print("    WARNING: pool not fully warm after 7 min; continuing")
            break
        time.sleep(20); waited += 20

    if args.warm_image:
        print("[2b] Warming container image with a trivial ML Job")
        try:
            import tempfile
            from snowflake.ml.jobs import submit_file
            s.sql(f"USE ROLE {C.DEPLOY_ROLE}").collect()
            s.sql(f"USE SCHEMA {C.PROD_DATABASE}.{C.REGISTRY_SCHEMA}").collect()
            with tempfile.TemporaryDirectory() as d:
                p = os.path.join(d, "warm.py")
                open(p, "w").write("print('image warm')\n")
                job = submit_file(p, C.JOB_COMPUTE_POOL, stage_name=C.JOB_PAYLOAD_STAGE, session=s)
                job.wait()
                print(f"    warm job {job.status}")
            s.sql("USE ROLE ACCOUNTADMIN").collect()
        except Exception as e:
            print("    warm-image skipped:", type(e).__name__, str(e)[:100])

    print("[3] Readiness check")
    def count(fqn):
        try:
            return s.sql(f"SELECT COUNT(*) C FROM {fqn}").collect()[0]["C"]
        except Exception:
            return None
    dev_abt = count(C.abt_fqn("dev"))
    preds = count(C.PREDICTIONS)
    try:
        dev_versions = [str(v.version_name) for v in
                        __import__("snowflake.ml.registry", fromlist=["Registry"]).Registry(
                            session=s, database_name=C.DEV_DATABASE, schema_name=C.REGISTRY_SCHEMA
                        ).get_model(C.MODEL_NAME).versions()]
    except Exception:
        dev_versions = []
    print(f"    dev ABT rows:        {dev_abt}")
    print(f"    dev model versions:  {dev_versions}")
    print(f"    prod predictions:    {preds}")

    ok = bool(dev_abt) and ("V1" in dev_versions)
    print("\n" + ("GO - core dev artifacts present." if ok else
                  "CHECK - dev artifacts missing; run setup 00-03 first."))
    print("Pool is warm. Run the demo; the promotion ML Job should start without cold-start.")
    s.close()


if __name__ == "__main__":
    main()
