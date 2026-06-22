"""Setup 04 - Deploy the registered model as a real-time inference REST service.

Creates an SPCS-backed HTTP endpoint (ingress enabled) on MLOPS_CPU_M_POOL from
the registered AML fraud model. The demo's score_transaction.py reads the full
feature vector from the feature group and POSTs it here (dataframe_split).

Run: SNOWFLAKE_CONNECTION_NAME=<conn> .venv/bin/python setup/04_deploy_inference_service.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from snowpark_session import create_snowpark_session

from snowflake.ml.registry import Registry


def main():
    session = create_snowpark_session()
    session.sql(f"USE WAREHOUSE {C.WAREHOUSE}").collect()
    session.sql(f"USE SCHEMA {C.DATABASE}.{C.FEATURE_STORE_SCHEMA}").collect()

    # Ensure the compute pool is resumed.
    print(f"Resuming compute pool {C.INFERENCE_COMPUTE_POOL}")
    try:
        session.sql(f"ALTER COMPUTE POOL {C.INFERENCE_COMPUTE_POOL} RESUME").collect()
    except Exception as e:
        print(f"  (resume note: {e})")

    reg = Registry(session=session, database_name=C.DATABASE,
                   schema_name=C.FEATURE_STORE_SCHEMA)
    mv = reg.get_model(C.MODEL_NAME).version(C.MODEL_VERSION)

    # Idempotency: skip if the service already lists for this model version.
    existing = []
    try:
        existing = [s.get("name", s) if isinstance(s, dict) else s
                    for s in mv.list_services().to_dict("records")]
    except Exception:
        pass

    # Drop any existing service so we (re)deploy the current model version.
    fqsvc = f"{C.DATABASE}.{C.FEATURE_STORE_SCHEMA}.{C.INFERENCE_SERVICE}"
    try:
        session.sql(f"DROP SERVICE IF EXISTS {fqsvc}").collect()
        print(f"  dropped existing {C.INFERENCE_SERVICE} (if any)")
    except Exception as e:
        print(f"  (drop note: {e})")

    print(f"Creating inference service {C.INFERENCE_SERVICE} on {C.MODEL_NAME}/{C.MODEL_VERSION} (CPU, ~10 min build)")
    mv.create_service(
        service_name=C.INFERENCE_SERVICE,
        service_compute_pool=C.INFERENCE_COMPUTE_POOL,
        ingress_enabled=True,
        gpu_requests=None,
        max_instances=1,
    )

    # --- Endpoint URL ------------------------------------------------------
    print("\nResolving public endpoint (waiting for ingress provisioning)...")
    url = None
    for _ in range(40):
        rows = session.sql(
            f"SHOW ENDPOINTS IN SERVICE {C.DATABASE}.{C.FEATURE_STORE_SCHEMA}.{C.INFERENCE_SERVICE}"
        ).collect()
        for r in rows:
            d = r.as_dict()
            u = d.get("ingress_url") or ""
            if u and "snowflakecomputing" in u:
                url = u
                break
        if url:
            break
        time.sleep(30)

    if url:
        print(f"  inference endpoint: https://{url}/predict")
        print("  (call with header  Authorization: Snowflake Token=\"$SNOWFLAKE_PAT\")")
    else:
        print("  endpoint not ready yet; re-check with SHOW ENDPOINTS IN SERVICE ...")

    print("\nSetup 04 complete.")
    session.close()


if __name__ == "__main__":
    main()
