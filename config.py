"""Central configuration for the Real-Time Feature Store Fraud-Detection demo.

All object names, file paths, and tunables live here so every setup/ and demo/
script imports from one place. No secrets are stored in this file - the REST
auth token is read from the SNOWFLAKE_PAT environment variable at runtime.
"""

import os

# --- Snowflake objects -------------------------------------------------------
DATABASE = "FRAUD_RT_DEMO"
RAW_SCHEMA = "RAW"
CURATED_SCHEMA = "CURATED"
FEATURE_STORE_SCHEMA = "FEATURE_STORE"
WAREHOUSE = "CORTEX_CODE_WH"

# Compute pool reused for the real-time inference SPCS service.
INFERENCE_COMPUTE_POOL = "MLOPS_CPU_M_POOL"

# Feature Store producer/consumer database roles (created in setup 02).
FS_PRODUCER_ROLE = "FS_PRODUCER_ROLE"
FS_CONSUMER_ROLE = "FS_CONSUMER_ROLE"

# --- Entity / feature views --------------------------------------------------
ENTITY_NAME = "ACCOUNT"
ENTITY_JOIN_KEY = "ACCOUNT_ID"

FV_PROFILE = "ACCOUNT_PROFILE"        # batch online FV (slow-moving profile)
FV_VELOCITY = "ACCOUNT_VELOCITY"      # stream FV (continuous time-windowed aggs)
FV_REALTIME = "TXN_RISK_SIGNALS"      # real-time FV registered live in the demo
FEATURE_GROUP = "FRAUD_FEATURES"      # bundle for training + single-call serving
FV_VERSION = "V1"

# Stream source name (must match the records POSTed to the Ingest API).
STREAM_SOURCE = "TRANSACTION_EVENTS"

# --- Model / service ---------------------------------------------------------
MODEL_NAME = "AML_FRAUD_GBM"
MODEL_VERSION = "V2"
INFERENCE_SERVICE = "AML_FRAUD_RT_SERVICE"

# --- Raw / curated tables ----------------------------------------------------
TBL_TRANS = f"{DATABASE}.{RAW_SCHEMA}.TRANSACTIONS"
TBL_ACCOUNTS = f"{DATABASE}.{RAW_SCHEMA}.ACCOUNTS"
TBL_PATTERNS = f"{DATABASE}.{RAW_SCHEMA}.LAUNDERING_PATTERNS"   # reserved (Appendix A)
TBL_ACCOUNT_HISTORY = f"{DATABASE}.{CURATED_SCHEMA}.ACCOUNT_HISTORY"
TBL_TXN_SPINE = f"{DATABASE}.{CURATED_SCHEMA}.TXN_SPINE"
STAGE_RAW = f"{DATABASE}.{RAW_SCHEMA}.RAW_STAGE"

# --- Source data files (currently in project root) ---------------------------
# The 3 IBM AML HI-Small files. Loader reads from here.
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _resolve(name: str) -> str:
    """Prefer source-data/ if the file was moved there, else project root."""
    in_srcdir = os.path.join(_PROJECT_ROOT, "source-data", name)
    in_root = os.path.join(_PROJECT_ROOT, name)
    return in_srcdir if os.path.exists(in_srcdir) else in_root


FILE_TRANS = _resolve("HI-Small_Trans.csv")
FILE_ACCOUNTS = _resolve("HI-Small_accounts.csv")
FILE_PATTERNS = _resolve("HI-Small_Patterns.txt")

# --- Data handling constants -------------------------------------------------
# "Marco quirk": transactions on/after this date are the trailing all-laundering
# tail (final hops of in-flight chains after normal-traffic generation stopped).
# Trim them so a chronological split can't leak the date as a predictor.
TAIL_CUTOFF_DATE = "2022-09-11"

# High-risk payment formats (used to derive IS_HIGH_RISK_FORMAT).
HIGH_RISK_FORMATS = ["Bitcoin", "Wire", "Cash"]

# Negative downsampling cap for local XGBoost training (all positives kept).
TRAIN_NEG_SAMPLE = 800_000

# Chronological split fractions (applied within the clean Sep 1-10 window).
SPLIT_TRAIN_FRAC = 0.60
SPLIT_VAL_FRAC = 0.20  # remainder -> test


# --- REST auth ---------------------------------------------------------------
def get_pat() -> str:
    """Return the Programmatic Access Token for REST Ingest/Query/inference.

    Resolution order:
      1. SNOWFLAKE_PAT environment variable
      2. A token file under ./pat/ matching *token-secret.txt (gitignored)
    """
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
    raise RuntimeError(
        "No PAT found. Set SNOWFLAKE_PAT, or place the token file in ./pat/ "
        "(e.g. pat/RT_FEATURE_STORE_DEMO_PAT-token-secret.txt)."
    )


# The Online Feature Store SDK reads the PAT from the SNOWFLAKE_PAT environment
# variable (for online feature reads). Populate it from the token file on import
# so SDK online reads and the REST scripts both work without manual export.
if "SNOWFLAKE_PAT" not in os.environ:
    try:
        os.environ["SNOWFLAKE_PAT"] = get_pat()
    except Exception:
        pass
