# MLOps on Snowflake — Dev Sandbox Inside a Production Account

A demo of how a data-science team operates a **dev ML sandbox inside a production
Snowflake account**: reading production data, iterating on features and models with
experiment tracking, and promoting to production through a **keyless, gated CI/CD
pipeline** — where a data scientist can *never* deploy to prod directly.

Fraud / anti-money-laundering (IBM AML HI-Small, synthetic) is the relatable backdrop,
but the story is the **ML process**, not the model.

## The governance boundary (the point of the demo)
- **Dev can read all prod data** — no masking, no cohort restriction. The data isn't sensitive; the message is about *process*.
- **The enforced line is write/deploy to prod.** A data scientist (`ML_DEV_ROLE`) can do anything in the dev sandbox and read prod, but **cannot** write the prod feature store, registry, or predictions.
- **Only the service account (`SVC_ML_DEPLOY`) deploys to prod**, and only through GitHub Actions using **OIDC / Workload Identity Federation** — no key pair, no stored secret. A GitHub `production` environment adds a human approval gate on top of RBAC.

## Topology (one account, two databases)
```
ML_FRAUD_PRODUCTION          ML_FRAUD_DEV_SANDBOX
  RAW / CURATED (prod data)    CURATED    (dev ABT)
  CURATED.FRAUD_ABT (prod)     FEATURE_STORE (dev FVs)
  FEATURE_STORE (prod FVs)     ML         (dev registry)
  ML (prod registry)           EXPERIMENTS (tracking)
  ANALYTICS.PREDICTIONS
```

Roles: `ML_DEV_ROLE` (data scientist, read prod / full dev), `ML_DEPLOY_SVC` (only role
that writes prod), `SVC_ML_DEPLOY` (OIDC service user for CI).

## The lineage the demo tells
```
prod RAW ─▶ selective transform ─▶ ABT ─▶ feature views ─▶ train ─▶ experiment ─▶ (Git gate) ─▶ prod
```

## Repository layout (organized by *when you run it*)
```
config.py, snowpark_session.py, requirements.txt   shared
transforms/base_features.py                          shared module (imported)
setup/     one-time build (done once): 00_rbac, 01_load_data_prod,
           02_feature_store, 03_train_register, online/ (optional)
pre_demo/  kickoff.py            run ~5 min before presenting (pre-warm + readiness)
           enable_online.py      OPTIONAL: enable the online store the day before (costs 24/7)
demo/      what you run LIVE, in order:
             01_explore_and_prep.ipynb    Act 2 - DS notebook
             02_add_feature_retrain.ipynb  Act 4 - retrain V2 (notebook)
             03_submit_promote_job.py      Act 5 - promote (also run by GitHub)
             promote_model.py              ML Job payload (not run directly)
             optional_online_realtime.ipynb  OPTIONAL Act 6 - real-time serving
reset/     teardown.py           full teardown of both DBs + roles + service user
```

## Files
| Path | What |
|------|------|
| `config.py` | Env-aware config (`ML_ENV=dev\|prod`) — DBs, schemas, roles, names |
| `setup/00_rbac.py` | Two DBs, roles, OIDC service user + auth policy, grants; verifies the boundary |
| `setup/01_load_data_prod.py` | Loads IBM AML data into `ML_FRAUD_PRODUCTION` (clean prod data) |
| `transforms/base_features.py` | The DS's **selective transform**: prod → `FRAUD_ABT` (env-aware, promotable) |
| `demo/01_explore_and_prep.ipynb` | DS first action: explore prod, build the dev ABT |
| `setup/02_feature_store.py` | Env-aware **batch** feature store over the ABT (online path is optional) |
| `setup/03_train_register.py` | Dev training + experiment tracking + dev registry |
| `demo/02_add_feature_retrain.ipynb` | The loop (notebook): add a feature view → retrain V2 → compare |
| `demo/03_submit_promote_job.py` | Submits the promotion as a server-side ML Job (run by GitHub or locally) |
| `demo/promote_model.py` | Service-account promotion payload: dev → prod registry + batch scoring task |
| `pre_demo/kickoff.py` | Pre-warm the compute pool + warehouse + readiness check |
| `.github/workflows/deploy-model.yml` | Keyless (OIDC) promotion gate with `production` approval |
| `setup/online/` | **Optional** Postgres online / real-time path (24/7 cost) — not the demo |
| `pre_demo/enable_online.py` | **Optional**: enable + verify the online store (run the day before) |
| `demo/optional_online_realtime.ipynb` | **Optional** Act 6: real-time serving on the online store |
| `reset/teardown.py` | Full teardown of both DBs, roles, service user, policy |

## Fork & run this yourself
This repo is a self-contained reference — fork it and stand it up in your own Snowflake
account. Nothing here is a secret (keyless OIDC, `.gitignore` covers keys/data), and the
deploy workflow triggers on `workflow_dispatch` only, so it's safe to make public.

Three values are account/repo-specific; everything else is created by `setup/`:

1. **Repo identity** — `config.py` reads `GITHUB_REPO` from the environment. In GitHub
   Actions it's auto-detected (`GITHUB_REPOSITORY`). Running `setup/` locally, set it so the
   OIDC trust points at *your* fork: `export GITHUB_REPO=your-org/your-repo`.
2. **Snowflake account** — set repo **variable** `SNOWFLAKE_ACCOUNT` (Settings → Secrets and
   variables → Actions → Variables). It's an identifier, not a credential.
3. **Warehouse** — set repo **variable** `SNOWFLAKE_WAREHOUSE` (and `WAREHOUSE`/env locally).
   The live Workspace notebooks (`demo/*.ipynb`) also pin `CORTEX_CODE_WH` in their first
   setup cell — change that to your warehouse when you open them.

Then:
- **In Snowflake:** run `setup/00_rbac.py` (creates both DBs, roles, and the `SVC_ML_DEPLOY`
  OIDC user bound to your repo), then the rest of the setup order below.
- **In GitHub:** create a `production` **environment** and add a required reviewer — this is
  the human approval gate on every deploy.
- **Workspace link (optional):** link a Snowsight Workspace to your fork to view/pull the code.
  A private fork needs a **read-only** GitHub PAT stored as a Snowflake secret; a public fork
  needs none. You won't commit from the Workspace, so read-only scope is enough.

## Setup order
```bash
# 1. RBAC + topology (as admin)
SNOWFLAKE_CONNECTION_NAME=<your-connection> ML_ENV=dev  python setup/00_rbac.py
# 2. Prod data
SNOWFLAKE_CONNECTION_NAME=<your-connection> ML_ENV=prod python setup/01_load_data_prod.py
# 3. DS transform -> dev ABT (as ML_DEV_ROLE)
SNOWFLAKE_CONNECTION_NAME=<your-connection> ML_ENV=dev  python transforms/base_features.py
# 4. Dev batch feature store
SNOWFLAKE_CONNECTION_NAME=<your-connection> ML_ENV=dev  python setup/02_feature_store.py
# 5. Train + track + register (dev)
SNOWFLAKE_CONNECTION_NAME=<your-connection> ML_ENV=dev  python setup/03_train_register.py --version V1
```

## The MLOps demo
- **Retrain (dev):** open `demo/02_add_feature_retrain.ipynb` in a Snowsight Workspace and
  run all cells — adds the `ACCOUNT_RISK` feature view, retrains V2, logs the experiment run,
  and compares V1 → V2.
- **Promote (dev → prod):** trigger the **GitHub Actions** workflow ("Promote model to
  production") and approve the `production` environment. It submits an **ML Job** that runs
  the promotion server-side as `ML_DEPLOY_SVC`. Local fallback (off-stage):
  `SNOWFLAKE_CONNECTION_NAME=<your-connection> python demo/03_submit_promote_job.py --dev-version V2`

Batch predictions land in `ML_FRAUD_PRODUCTION.ANALYTICS.PREDICTIONS`, refreshed by the
scheduled task `SCORE_BATCH_TASK` (created suspended; resume to enable daily scoring).

## Teardown (stop all cost)
```bash
SNOWFLAKE_CONNECTION_NAME=<your-connection> python reset/teardown.py         # dry run
SNOWFLAKE_CONNECTION_NAME=<your-connection> python reset/teardown.py --yes   # execute
```

See `RUNBOOK.md` for the presenter talk track. The Snowflake real-time / online serving
build (Postgres online store + REST + SPCS) lives in git history and under `setup/online/`.
