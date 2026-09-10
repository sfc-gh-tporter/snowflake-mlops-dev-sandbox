"""Submit the promotion as a Snowflake ML Job (server-side, on a compute pool).

This is what the GitHub Actions gate runs. Instead of executing the promotion on
the GitHub runner, it submits an ML Job that runs mlops/promote_model.py on the
Snowflake compute pool - "operationalizing" the workload rather than running a
notebook/script. The job runs under the submitting role (ML_DEPLOY_SVC).

Only a minimal payload (config, transforms, the promote script) is uploaded - NOT
the repo (no .venv / source-data / pat/). scikit-learn is pinned so the job runtime
matches the version the model was trained with (clean mv.load()).

  # local (as an admin holding ML_DEPLOY_SVC):
  SNOWFLAKE_CONNECTION_NAME=<your-connection> .venv/bin/python mlops/submit_promote_job.py --dev-version V2
  # CI: authenticated as SVC_ML_DEPLOY via OIDC (role ML_DEPLOY_SVC)
"""

import argparse
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from snowpark_session import create_snowpark_session

from snowflake.ml.jobs import submit_directory

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_payload(tmp):
    """Copy only the files the promotion needs into a clean payload dir."""
    for f in ("config.py", "snowpark_session.py"):
        shutil.copy(os.path.join(ROOT, f), os.path.join(tmp, f))
    shutil.copytree(os.path.join(ROOT, "transforms"), os.path.join(tmp, "transforms"),
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(os.path.join(ROOT, "demo", "promote_model.py"),
                os.path.join(tmp, "promote_model.py"))
    # Pin the training-critical dependency so the job runtime matches.
    with open(os.path.join(tmp, "requirements.txt"), "w") as fh:
        fh.write("scikit-learn==1.7.2\n")
    return tmp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-version", default="V2")
    ap.add_argument("--no-wait", action="store_true")
    args = ap.parse_args()

    session = create_snowpark_session()
    # Ensure the job owner role is the service/deploy role (governance).
    try:
        session.sql(f"USE ROLE {C.DEPLOY_ROLE}").collect()
    except Exception:
        pass
    session.sql(f"USE DATABASE {C.PROD_DATABASE}").collect()
    session.sql(f"USE SCHEMA {C.REGISTRY_SCHEMA}").collect()

    with tempfile.TemporaryDirectory() as tmp:
        payload = build_payload(tmp)
        print(f"Submitting promotion ML Job to {C.JOB_COMPUTE_POOL} "
              f"(promote {C.MODEL_NAME}/{args.dev_version})")
        job = submit_directory(
            payload, C.JOB_COMPUTE_POOL,
            entrypoint="promote_model.py",
            stage_name=C.JOB_PAYLOAD_STAGE,
            args=["--dev-version", args.dev_version],
            external_access_integrations=[C.PYPI_EAI],
            session=session,
        )
        print(f"  job id: {job.id}  status: {job.status}")
        if args.no_wait:
            return
        print("  waiting for job to finish (compute pool cold start can take minutes)...")
        job.wait()
        print(f"  final status: {job.status}")
        print("\n--- job logs (tail) ---")
        logs = job.get_logs()
        print("\n".join(logs.splitlines()[-40:]))
        if job.status != "DONE":
            sys.exit(1)
    session.close()


if __name__ == "__main__":
    main()
