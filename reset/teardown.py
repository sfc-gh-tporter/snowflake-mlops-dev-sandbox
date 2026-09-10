"""Teardown / reset for the MLOps dev/prod demo.

Removes EVERYTHING this demo created across BOTH databases plus the account-level
roles, the OIDC service user, and the auth policy. If the optional online path
was enabled, its Postgres online service is dropped first (24/7 HA cost) with an
orphaned-instance safety net.

Dry-run by default; only tears down with --yes. The shared compute pool
(MLOPS_CPU_M_POOL) is left intact unless --drop-pool is passed.

  SNOWFLAKE_CONNECTION_NAME=<your-connection> .venv/bin/python setup/99_reset_teardown.py          # dry run
  SNOWFLAKE_CONNECTION_NAME=<your-connection> .venv/bin/python setup/99_reset_teardown.py --yes    # execute
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from snowpark_session import create_snowpark_session

DRY = True


def step(desc, fn):
    if DRY:
        print(f"  [dry-run] would: {desc}")
        return
    try:
        fn()
        print(f"  [ok]   {desc}")
    except Exception as e:
        print(f"  [skip] {desc}  ->  {type(e).__name__}: {str(e)[:150]}")


def sql(session, statement):
    return lambda: session.sql(statement).collect()


def drop_online_if_present(session, db):
    """Best-effort: drop the Postgres online service in a DB's feature store."""
    def fn():
        from snowflake.ml.feature_store import FeatureStore, CreationMode
        fs = FeatureStore(session=session, database=db, name=C.FEATURE_STORE_SCHEMA,
                          default_warehouse=C.WAREHOUSE, creation_mode=CreationMode.FAIL_IF_NOT_EXIST)
        # remove feature views first, then the online service (if any)
        for name in (C.FV_VELOCITY, C.FV_REALTIME, "ACCOUNT_RISK", C.FV_PROFILE):
            try:
                fs.delete_feature_view(fs.get_feature_view(name, C.FV_VERSION))
            except Exception:
                pass
        try:
            fs.delete_stream_source(C.STREAM_SOURCE)
        except Exception:
            pass
        fs.drop_online_service()
    step(f"drop online service in {db} (if present)", fn)


def main():
    global DRY
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--drop-pool", action="store_true")
    args = ap.parse_args()
    DRY = not args.yes

    print("=" * 72)
    print("MLOPS DEV/PROD DEMO -- TEARDOWN", "(DRY RUN)" if DRY else "(EXECUTING)")
    print("=" * 72)

    s = create_snowpark_session()
    step("USE ROLE ACCOUNTADMIN", sql(s, "USE ROLE ACCOUNTADMIN"))
    step(f"USE WAREHOUSE {C.WAREHOUSE}", sql(s, f"USE WAREHOUSE {C.WAREHOUSE}"))

    print("\n[1] Optional online services (Postgres HA -- drop first if present)")
    if not DRY:
        for db in (C.DEV_DATABASE, C.PROD_DATABASE):
            drop_online_if_present(s, db)
    else:
        print(f"  [dry-run] would: drop online services in {C.DEV_DATABASE}, {C.PROD_DATABASE} (if any)")

    print("\n[2] Batch task + model registries")
    step(f"DROP TASK {C.BATCH_TASK}", sql(s, f"DROP TASK IF EXISTS {C.BATCH_TASK}"))
    for db in (C.DEV_DATABASE, C.PROD_DATABASE):
        step(f"DROP MODEL {db}.{C.REGISTRY_SCHEMA}.{C.MODEL_NAME}",
             sql(s, f"DROP MODEL IF EXISTS {db}.{C.REGISTRY_SCHEMA}.{C.MODEL_NAME}"))

    print("\n[3] Account-level identity: service user, auth policy")
    # detach policy before dropping user/policy
    step(f"UNSET AUTHENTICATION POLICY on {C.DEPLOY_USER}",
         sql(s, f"ALTER USER IF EXISTS {C.DEPLOY_USER} UNSET AUTHENTICATION POLICY"))
    step(f"DROP USER {C.DEPLOY_USER}", sql(s, f"DROP USER IF EXISTS {C.DEPLOY_USER}"))
    step(f"DROP AUTHENTICATION POLICY {C.AUTH_POLICY}",
         sql(s, f"DROP AUTHENTICATION POLICY IF EXISTS {C.PROD_DATABASE}.{C.REGISTRY_SCHEMA}.{C.AUTH_POLICY}"))

    print("\n[4] Databases (cascade: schemas, tables, ABT, feature store, experiments)")
    for db in (C.DEV_DATABASE, C.PROD_DATABASE):
        step(f"DROP DATABASE {db} CASCADE", sql(s, f"DROP DATABASE IF EXISTS {db} CASCADE"))

    print("\n[5] Roles")
    for role in (C.DEV_ROLE, C.DEPLOY_ROLE):
        step(f"DROP ROLE {role}", sql(s, f"DROP ROLE IF EXISTS {role}"))

    print("\n[6] Compute pool")
    if args.drop_pool:
        step(f"DROP COMPUTE POOL {C.INFERENCE_COMPUTE_POOL}",
             sql(s, f"DROP COMPUTE POOL IF EXISTS {C.INFERENCE_COMPUTE_POOL}"))
    else:
        print(f"  [keep] {C.INFERENCE_COMPUTE_POOL} is shared/pre-existing -- not dropped.")

    print("\n[7] Safety net -- orphaned Postgres instance check")
    def sweep():
        rows = s.sql("SHOW POSTGRES INSTANCES").collect()
        orphans = [r["name"] for r in rows if str(r["name"]).upper().startswith("FS_RUNTIME_PG")]
        if not orphans:
            print(f"      {len(rows)} instances, no FS_RUNTIME_PG* orphans."); return
        for n in orphans:
            print(f"      dropping orphan {n}")
            s.sql(f'DROP POSTGRES INSTANCE IF EXISTS "{n}"').collect()
    step("SHOW POSTGRES INSTANCES + drop FS_RUNTIME_PG* orphans", sweep)

    print("\n" + "=" * 72)
    print("DRY RUN complete -- nothing changed. Re-run with --yes." if DRY
          else "Teardown complete. Both databases, roles, service user, and policy removed.")
    print("=" * 72)
    s.close()


if __name__ == "__main__":
    main()
