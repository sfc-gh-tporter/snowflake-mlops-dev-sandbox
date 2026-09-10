"""Central configuration for the MLOps dev/prod demo (fraud/AML backdrop).

Env-aware: set ML_ENV=dev|prod to resolve the database, schemas, and roles for
the two-database topology:

  ML_FRAUD_PRODUCTION   - prod data, prod feature store, prod model registry
  ML_FRAUD_DEV_SANDBOX  - dev feature store, dev registry, experiments

The dev sandbox reads prod data read-only; only the service account can write
prod. All object names live here so every script imports from one place.
"""

import os

# --- Environment -------------------------------------------------------------
ML_ENV = os.environ.get("ML_ENV", "dev").strip().lower()
if ML_ENV not in ("dev", "prod"):
    raise ValueError(f"ML_ENV must be 'dev' or 'prod', got {ML_ENV!r}")

# --- Databases (two-DB topology, one account) --------------------------------
PROD_DATABASE = "ML_FRAUD_PRODUCTION"
DEV_DATABASE = "ML_FRAUD_DEV_SANDBOX"
DATABASE = PROD_DATABASE if ML_ENV == "prod" else DEV_DATABASE

# --- Schemas -----------------------------------------------------------------
RAW_SCHEMA = "RAW"                 # prod-only; dev reads prod RAW read-only
CURATED_SCHEMA = "CURATED"         # holds the analytical base table (ABT)
FEATURE_STORE_SCHEMA = "FEATURE_STORE"
REGISTRY_SCHEMA = "ML"             # model registry lives here
ANALYTICS_SCHEMA = "ANALYTICS"     # batch predictions (prod)
EXPERIMENTS_SCHEMA = "EXPERIMENTS" # experiment tracking (dev)

WAREHOUSE = os.environ.get("SNOWFLAKE_WAREHOUSE", "CORTEX_CODE_WH")
INFERENCE_COMPUTE_POOL = "MLOPS_CPU_M_POOL"  # optional online path only

# --- Roles / keyless CI-CD identity ------------------------------------------
DEV_ROLE = "ML_DEV_ROLE"           # data scientist: full CRUD in dev, read prod
DEPLOY_ROLE = "ML_DEPLOY_SVC"      # only role that can write prod
DEPLOY_USER = "SVC_ML_DEPLOY"      # OIDC/WIF service user used by GitHub Actions
AUTH_POLICY = "ML_DEPLOY_WIF_POLICY"

# Repo that owns the keyless deploy identity. Auto-detects the fork in CI
# (GitHub sets GITHUB_REPOSITORY); set GITHUB_REPO locally before running
# setup/00_rbac.py so the OIDC trust points at your own fork.
GITHUB_REPO = (
    os.environ.get("GITHUB_REPO")
    or os.environ.get("GITHUB_REPOSITORY")  # auto-set by GitHub Actions
    or "sfc-gh-tporter/snowflake-mlops-dev-sandbox"
)
GITHUB_DEPLOY_ENV = "production"
OIDC_ISSUER = "https://token.actions.githubusercontent.com"
OIDC_SUBJECT = f"repo:{GITHUB_REPO}:environment:{GITHUB_DEPLOY_ENV}"

# --- Entity / features -------------------------------------------------------
ENTITY_NAME = "ACCOUNT"
ENTITY_JOIN_KEY = "ACCOUNT_ID"
FEATURE_GROUP = "FRAUD_FEATURES"
FV_VERSION = "V1"
FV_PROFILE = "ACCOUNT_PROFILE"        # batch FV (default path)
FV_VELOCITY = "ACCOUNT_VELOCITY"      # stream FV (optional online path)
FV_REALTIME = "TXN_RISK_SIGNALS"      # real-time FV (optional online path)
STREAM_SOURCE = "TRANSACTION_EVENTS"  # optional online path

# --- Analytical base table (the DS's selective transform output) -------------
ABT_TABLE = "FRAUD_ABT"


def abt_fqn(env: str = ML_ENV) -> str:
    """Fully-qualified ABT name for the given env (dev or prod)."""
    db = PROD_DATABASE if env == "prod" else DEV_DATABASE
    return f"{db}.{CURATED_SCHEMA}.{ABT_TABLE}"


ABT = abt_fqn(ML_ENV)

# --- Model / registry / batch scoring ----------------------------------------
MODEL_NAME = "AML_FRAUD_GBM"
REGISTRY_DB = DATABASE                       # dev or prod registry
PREDICTIONS = f"{PROD_DATABASE}.{ANALYTICS_SCHEMA}.PREDICTIONS"
BATCH_TASK = f"{PROD_DATABASE}.{ANALYTICS_SCHEMA}.SCORE_BATCH_TASK"
INFERENCE_SERVICE = "AML_FRAUD_RT_SERVICE"   # optional online path

# --- ML Jobs (promotion runs as a server-side job on a compute pool) ---------
JOB_COMPUTE_POOL = INFERENCE_COMPUTE_POOL
JOB_PAYLOAD_STAGE = f"{PROD_DATABASE}.{REGISTRY_SCHEMA}.JOB_PAYLOAD"
PYPI_EAI = "MLOPS_PYPI_ACCESS_INTEGRATION"

# --- Prod source tables (always in PROD; dev reads these) --------------------
TBL_TRANS = f"{PROD_DATABASE}.{RAW_SCHEMA}.TRANSACTIONS"
TBL_ACCOUNTS = f"{PROD_DATABASE}.{RAW_SCHEMA}.ACCOUNTS"
TBL_PATTERNS = f"{PROD_DATABASE}.{RAW_SCHEMA}.LAUNDERING_PATTERNS"
TBL_ACCOUNT_HISTORY = f"{PROD_DATABASE}.{CURATED_SCHEMA}.ACCOUNT_HISTORY"
TBL_TXN_SPINE = f"{PROD_DATABASE}.{CURATED_SCHEMA}.TXN_SPINE"
STAGE_RAW = f"{PROD_DATABASE}.{RAW_SCHEMA}.RAW_STAGE"

# --- Source data files -------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _resolve(name: str) -> str:
    in_srcdir = os.path.join(_PROJECT_ROOT, "source-data", name)
    in_root = os.path.join(_PROJECT_ROOT, name)
    return in_srcdir if os.path.exists(in_srcdir) else in_root


FILE_TRANS = _resolve("HI-Small_Trans.csv")
FILE_ACCOUNTS = _resolve("HI-Small_accounts.csv")
FILE_PATTERNS = _resolve("HI-Small_Patterns.txt")

# --- Data-handling constants -------------------------------------------------
TAIL_CUTOFF_DATE = "2022-09-11"   # trim trailing all-laundering tail
HIGH_RISK_FORMATS = ["Bitcoin", "Wire", "Cash"]
TRAIN_NEG_SAMPLE = 400_000        # negatives sampled for training (all positives kept)
SPLIT_TRAIN_FRAC = 0.60
SPLIT_VAL_FRAC = 0.20


# --- Optional REST auth (online path only) -----------------------------------
def get_pat() -> str:
    """PAT for the OPTIONAL online REST path. Not used by the batch demo."""
    pat = os.environ.get("SNOWFLAKE_PAT")
    if pat:
        return pat.strip()
    import glob
    matches = glob.glob(os.path.join(_PROJECT_ROOT, "pat", "*token-secret.txt"))
    if matches:
        with open(matches[0]) as fh:
            tok = fh.read().strip()
        if tok:
            return tok
    raise RuntimeError("No PAT found (only needed for the optional online path).")
