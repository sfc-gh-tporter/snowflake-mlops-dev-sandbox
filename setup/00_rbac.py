"""Phase 1 - RBAC + topology + keyless OIDC service identity.

Creates the two-database MLOps topology and the roles that enforce the core
governance boundary: a data scientist (ML_DEV_ROLE) can read all prod data and
do anything in the dev sandbox, but CANNOT write prod. Only the service account
(SVC_ML_DEPLOY, authenticating via GitHub OIDC / Workload Identity Federation)
can deploy to prod.

Run as an admin (script elevates to ACCOUNTADMIN):
  SNOWFLAKE_CONNECTION_NAME=<your-connection> .venv/bin/python setup/00_rbac.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from snowpark_session import create_snowpark_session


def run(session, sql, label=None, expect_fail=False):
    tag = label or sql[:70].replace("\n", " ")
    try:
        session.sql(sql).collect()
        if expect_fail:
            print(f"  [WARN] expected failure but SUCCEEDED: {tag}")
            return True
        print(f"  [ok]   {tag}")
        return True
    except Exception as e:
        if expect_fail:
            print(f"  [ok, blocked] {tag}  ->  {type(e).__name__}")
            return False
        print(f"  [skip] {tag}  ->  {type(e).__name__}: {str(e)[:120]}")
        return False


def main():
    s = create_snowpark_session()
    run(s, "USE ROLE ACCOUNTADMIN")

    print("\n[1] Databases + schemas")
    run(s, f"CREATE DATABASE IF NOT EXISTS {C.PROD_DATABASE}")
    run(s, f"CREATE DATABASE IF NOT EXISTS {C.DEV_DATABASE}")
    for sch in (C.RAW_SCHEMA, C.CURATED_SCHEMA, C.FEATURE_STORE_SCHEMA,
                C.REGISTRY_SCHEMA, C.ANALYTICS_SCHEMA):
        run(s, f"CREATE SCHEMA IF NOT EXISTS {C.PROD_DATABASE}.{sch}")
    for sch in (C.CURATED_SCHEMA, C.FEATURE_STORE_SCHEMA,
                C.REGISTRY_SCHEMA, C.EXPERIMENTS_SCHEMA):
        run(s, f"CREATE SCHEMA IF NOT EXISTS {C.DEV_DATABASE}.{sch}")
    # Payload stage for ML Jobs (promotion runs as a server-side job).
    run(s, f"CREATE STAGE IF NOT EXISTS {C.JOB_PAYLOAD_STAGE}")

    print("\n[2] Roles")
    run(s, f"CREATE ROLE IF NOT EXISTS {C.DEV_ROLE}")
    run(s, f"CREATE ROLE IF NOT EXISTS {C.DEPLOY_ROLE}")
    # Role hierarchy: admins manage both; the dev role goes to the human user.
    run(s, f"GRANT ROLE {C.DEV_ROLE} TO ROLE SYSADMIN")
    run(s, f"GRANT ROLE {C.DEPLOY_ROLE} TO ROLE SYSADMIN")
    run(s, f"GRANT ROLE {C.DEV_ROLE} TO USER TPORTER")
    # Deploy role inherits dev role so it can load dev-owned models during promotion.
    # (This does NOT give the dev role deploy rights - the boundary is one-way.)
    run(s, f"GRANT ROLE {C.DEV_ROLE} TO ROLE {C.DEPLOY_ROLE}")
    # NOTE: ML_DEPLOY_SVC is deliberately NOT granted to the dev workflow.

    print("\n[3] Warehouse + task privileges")
    for role in (C.DEV_ROLE, C.DEPLOY_ROLE):
        run(s, f"GRANT USAGE ON WAREHOUSE {C.WAREHOUSE} TO ROLE {role}")
    run(s, f"GRANT EXECUTE TASK ON ACCOUNT TO ROLE {C.DEPLOY_ROLE}")
    run(s, f"GRANT EXECUTE MANAGED TASK ON ACCOUNT TO ROLE {C.DEPLOY_ROLE}")
    # ML Jobs: deploy role runs the promotion as a server-side job on the pool.
    run(s, f"GRANT USAGE ON COMPUTE POOL {C.JOB_COMPUTE_POOL} TO ROLE {C.DEPLOY_ROLE}")
    run(s, f"GRANT USAGE ON INTEGRATION {C.PYPI_EAI} TO ROLE {C.DEPLOY_ROLE}")

    print("\n[4] DEV role: full CRUD in the dev sandbox")
    run(s, f"GRANT ALL ON DATABASE {C.DEV_DATABASE} TO ROLE {C.DEV_ROLE}")
    run(s, f"GRANT ALL ON ALL SCHEMAS IN DATABASE {C.DEV_DATABASE} TO ROLE {C.DEV_ROLE}")
    run(s, f"GRANT ALL ON FUTURE SCHEMAS IN DATABASE {C.DEV_DATABASE} TO ROLE {C.DEV_ROLE}")

    print("\n[5] DEV role: read-only on prod data (RAW + CURATED)")
    run(s, f"GRANT USAGE ON DATABASE {C.PROD_DATABASE} TO ROLE {C.DEV_ROLE}")
    for sch in (C.RAW_SCHEMA, C.CURATED_SCHEMA):
        run(s, f"GRANT USAGE ON SCHEMA {C.PROD_DATABASE}.{sch} TO ROLE {C.DEV_ROLE}")
        for obj in ("TABLES", "VIEWS", "DYNAMIC TABLES"):
            run(s, f"GRANT SELECT ON ALL {obj} IN SCHEMA {C.PROD_DATABASE}.{sch} TO ROLE {C.DEV_ROLE}")
            run(s, f"GRANT SELECT ON FUTURE {obj} IN SCHEMA {C.PROD_DATABASE}.{sch} TO ROLE {C.DEV_ROLE}")

    print("\n[6] DEPLOY role: full write on prod + read dev registry (to pull the model)")
    run(s, f"GRANT ALL ON DATABASE {C.PROD_DATABASE} TO ROLE {C.DEPLOY_ROLE}")
    run(s, f"GRANT ALL ON ALL SCHEMAS IN DATABASE {C.PROD_DATABASE} TO ROLE {C.DEPLOY_ROLE}")
    run(s, f"GRANT ALL ON FUTURE SCHEMAS IN DATABASE {C.PROD_DATABASE} TO ROLE {C.DEPLOY_ROLE}")
    # Table-level SELECT on prod data (managed feature-view DTs run as this role).
    for sch in (C.RAW_SCHEMA, C.CURATED_SCHEMA):
        for obj in ("TABLES", "VIEWS", "DYNAMIC TABLES"):
            run(s, f"GRANT SELECT ON ALL {obj} IN SCHEMA {C.PROD_DATABASE}.{sch} TO ROLE {C.DEPLOY_ROLE}")
            run(s, f"GRANT SELECT ON FUTURE {obj} IN SCHEMA {C.PROD_DATABASE}.{sch} TO ROLE {C.DEPLOY_ROLE}")
    run(s, f"GRANT USAGE ON DATABASE {C.DEV_DATABASE} TO ROLE {C.DEPLOY_ROLE}")
    run(s, f"GRANT USAGE ON SCHEMA {C.DEV_DATABASE}.{C.REGISTRY_SCHEMA} TO ROLE {C.DEPLOY_ROLE}")
    run(s, f"GRANT READ, WRITE ON STAGE {C.JOB_PAYLOAD_STAGE} TO ROLE {C.DEPLOY_ROLE}")
    for obj in ("MODELS", "TABLES", "VIEWS"):
        run(s, f"GRANT SELECT ON ALL {obj} IN SCHEMA {C.DEV_DATABASE}.{C.REGISTRY_SCHEMA} TO ROLE {C.DEPLOY_ROLE}") if obj != "MODELS" else \
            run(s, f"GRANT USAGE ON ALL MODELS IN SCHEMA {C.DEV_DATABASE}.{C.REGISTRY_SCHEMA} TO ROLE {C.DEPLOY_ROLE}")
        run(s, f"GRANT SELECT ON FUTURE {obj} IN SCHEMA {C.DEV_DATABASE}.{C.REGISTRY_SCHEMA} TO ROLE {C.DEPLOY_ROLE}") if obj != "MODELS" else \
            run(s, f"GRANT USAGE ON FUTURE MODELS IN SCHEMA {C.DEV_DATABASE}.{C.REGISTRY_SCHEMA} TO ROLE {C.DEPLOY_ROLE}")
    # dev also needs to read prod data to build features; deploy runs prod transform reading prod (already has ALL on prod)

    print("\n[7] Keyless OIDC service user (GitHub Actions / WIF)")
    created = run(s, f"""CREATE USER IF NOT EXISTS {C.DEPLOY_USER}
        TYPE = SERVICE
        DEFAULT_ROLE = {C.DEPLOY_ROLE}
        WORKLOAD_IDENTITY = (
            TYPE = OIDC
            ISSUER = '{C.OIDC_ISSUER}'
            SUBJECT = '{C.OIDC_SUBJECT}'
        )
        COMMENT = 'GitHub Actions OIDC deploy identity; only role that writes prod'""",
        label="CREATE USER SVC_ML_DEPLOY (OIDC)")
    # If the user already existed, ensure WIF is set correctly.
    run(s, f"""ALTER USER {C.DEPLOY_USER} SET WORKLOAD_IDENTITY = (
            TYPE = OIDC ISSUER = '{C.OIDC_ISSUER}' SUBJECT = '{C.OIDC_SUBJECT}')""",
        label="ALTER USER set WORKLOAD_IDENTITY (idempotent)")
    run(s, f"GRANT ROLE {C.DEPLOY_ROLE} TO USER {C.DEPLOY_USER}")

    print("\n[8] Authentication policy: lock the service user to OIDC/WIF")
    pol = f"{C.PROD_DATABASE}.{C.REGISTRY_SCHEMA}.{C.AUTH_POLICY}"
    run(s, f"""CREATE OR REPLACE AUTHENTICATION POLICY {pol}
        WORKLOAD_IDENTITY_POLICY = (ALLOWED_PROVIDERS = (OIDC))
        COMMENT = 'Restrict SVC_ML_DEPLOY to GitHub OIDC workload identity'""")
    run(s, f"ALTER USER {C.DEPLOY_USER} SET AUTHENTICATION POLICY {pol}")

    print("\n[9] Verify the governance boundary as ML_DEV_ROLE")
    run(s, f"USE ROLE {C.DEV_ROLE}")
    run(s, f"USE WAREHOUSE {C.WAREHOUSE}")
    run(s, f"CREATE OR REPLACE TABLE {C.DEV_DATABASE}.{C.CURATED_SCHEMA}._RBAC_TEST (x INT)",
        label="dev write (expect OK)")
    run(s, f"DROP TABLE IF EXISTS {C.DEV_DATABASE}.{C.CURATED_SCHEMA}._RBAC_TEST")
    blocked = run(s, f"CREATE TABLE {C.PROD_DATABASE}.{C.FEATURE_STORE_SCHEMA}._RBAC_TEST (x INT)",
                  label="prod write (expect BLOCKED)", expect_fail=True)
    if blocked:  # it unexpectedly succeeded; clean up
        run(s, f"USE ROLE ACCOUNTADMIN")
        run(s, f"DROP TABLE IF EXISTS {C.PROD_DATABASE}.{C.FEATURE_STORE_SCHEMA}._RBAC_TEST")

    run(s, "USE ROLE ACCOUNTADMIN")
    print("\nRBAC + topology complete.")
    s.close()


if __name__ == "__main__":
    main()
