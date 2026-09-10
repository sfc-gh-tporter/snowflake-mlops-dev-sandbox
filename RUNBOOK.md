# RUNBOOK — MLOps on Snowflake Demo (Presenter Guide)

**The story you're telling:** how a data-science team runs a **dev ML sandbox inside a
production Snowflake account** — reading production data, iterating on features and models
with experiment tracking, and promoting to production through a **keyless, gated CI/CD
pipeline** where a data scientist can *never* deploy to prod directly. Fraud / AML is the
relatable backdrop; the subject is **Snowflake ML Ops**.

Audience framing: generic financial-services fraud/AML monitoring. Do not name the customer
in anything on screen.

---

## 0. Screen setup (three screens)

| Screen | Purpose |
|--------|---------|
| **Snowsight Workspace** (git-linked to the repo) | Primary. Open the dev notebook; browse Feature Store, Experiments, Model Registry, and the PREDICTIONS table in Snowsight. |
| **Cortex Code (desktop)** | Show this build conversation + run the pre-built setup / live retrain / kickoff / reset from a local terminal. |
| **GitHub** (repo + Actions) | Trigger the promotion workflow and approve the `production` environment (the human gate). |

**Where things run — legend used below:**
- **[TERM]** = local terminal in Cortex Code (`.venv` + `snow`/python). Runs the `.py` scripts.
- **[WS]** = Snowsight Workspace notebook (git-linked repo).
- **[UI]** = Snowsight browse (AI & ML / Data).
- **[GH]** = GitHub (Actions tab).

All [TERM] commands assume:
```bash
cd "RT Feature Store - Fraud Detection"
export SNOWFLAKE_CONNECTION_NAME=<your-connection>
```

> **During the live demo you never touch a terminal or Cortex Code.** Every live step is
> [WS] (Workspace notebook), [UI] (Snowsight), or [GH] (GitHub). [TERM] is used only for
> **pre-demo kickoff** and **post-demo reset** — off-stage. (Cortex Code is up only to *show*
> this build conversation.)

---

## 1. Pre-demo kickoff (run ~5 minutes before)

The only thing we can't speed up live is the ML Job's compute-pool cold start. Pre-warm it:

**[TERM]**
```bash
ML_ENV=dev .venv/bin/python pre_demo/kickoff.py            # resume + warm pool, readiness check
# optional, if the container image may be cold:
ML_ENV=dev .venv/bin/python pre_demo/kickoff.py --warm-image
```
Expect: pool `state=ACTIVE/IDLE nodes_ready=1`, dev ABT ~5,077,237 rows, dev model versions
`['V1','V2']`, and a **GO**. Now the promotion job starts without a multi-minute cold start.

> **Two separate GitHub connections — don't conflate them:**
> 1. **Workspace ↔ GitHub (git link):** lets Snowsight pull the repo. Configured on the
>    Snowflake side with an API integration + a **Snowflake SECRET holding a GitHub PAT**.
>    This is your connectivity task; it has nothing to do with the pipeline.
> 2. **GitHub Actions → Snowflake (promotion pipeline):** **keyless (OIDC/WIF)** — no key,
>    no password, no credential secret. The only value it needs is `SNOWFLAKE_ACCOUNT`, the
>    account identifier, stored as a GitHub **variable** (not a credential). Plus the
>    `production` environment for the approval gate.

---

## 2. What's pre-built vs. live

**Pre-built before the demo** (already done — the environment is standing):
`setup/00_rbac.py` → `setup/01_load_data_prod.py` → `transforms/base_features.py`
→ `setup/02_feature_store.py` → `setup/03_train_register.py --version V1`.

**Run live during the demo:** the retrain loop (V2) and the promotion (via GitHub).

If you ever need to rebuild from scratch, run the five pre-build steps in order (README has them).

---

## 3. Order of operations (the live demo)

### Act 1 — The sandbox boundary (~2 min) [UI or TERM]
Show the two databases and roles, then prove the boundary. In a Snowsight SQL worksheet:
```sql
USE ROLE ML_DEV_ROLE;
SELECT COUNT(*) FROM ML_FRAUD_PRODUCTION.CURATED.TXN_EVENTS;   -- works: dev reads prod
CREATE TABLE ML_FRAUD_PRODUCTION.FEATURE_STORE.X (a INT);      -- FAILS: dev cannot write prod
```
**Say:** dev has full *read* of prod (no masking needed — the point is process, not data). The
enforced line is *deploy*. Only the service account can write prod, and only through CI.

### Act 2 — The data scientist's first move (~3 min) [WS]
Open `demo/01_explore_and_prep.ipynb`. Run the EDA cells (fraud rate by channel), then the
cell that calls `build_abt(session, "dev")`.
**Say:** the DS doesn't build features off raw prod — they apply a **selective transform** into
a clean, labeled **analytical base table (ABT)** in their sandbox. The *same* transform is
promoted to prod later (dev/prod parity).

### Act 3 — Feature store + training + experiment tracking (~3 min) [UI]
- **Feature Store** (AI & ML → Features): show `ACCOUNT_PROFILE` (+ `ACCOUNT_RISK` after Act 4).
- **Experiments** (AI & ML → Experiments → `AML_FRAUD_TRAINING`): show the `TRAIN_V1` run's
  params + metrics.
- **Model Registry** (AI & ML → Models): show `AML_FRAUD_GBM` in the **dev** registry.
**Say:** training reads the ABT + feature store, logs every run's params/metrics for
reproducibility, and registers the model to the *dev* registry.

### Act 4 — The MLOps loop: add a feature, retrain, compare (~3 min) [WS]
Open `demo/02_add_feature_retrain.ipynb` in the Workspace and **run all cells**.
It registers a **new** `ACCOUNT_RISK` feature view (spend volatility + receiver fan-out),
retrains **V2**, logs a second experiment run, and prints the V1→V2 comparison (V2 wins).
**Say:** this is the iteration loop — new signal → retrain → tracked comparison → V2 becomes the
candidate (dev default). Refresh Experiments in [UI] to show `TRAIN_V1` vs `TRAIN_V2`.

### Act 5 — The promotion gate (~4 min) [GH]
This is the punchline. A data scientist **cannot** push V2 to prod. Go to **GitHub → Actions →
"Promote model to production" → Run workflow** (input `V2`).
- GitHub mints an **OIDC token** — no key/secret stored.
- The `production` environment requires **your approval** (show the approval click).
- The workflow authenticates as `SVC_ML_DEPLOY` and **submits an ML Job** that runs the
  promotion **server-side** on the compute pool: builds the prod ABT with the same transform,
  registers the prod feature store, promotes the model into the **prod** registry, and creates
  the batch scoring task.

Then in [UI]/[TERM] show the result:
```sql
SELECT COUNT(*) n,
       ROUND(AVG(IFF(LABEL=1,FRAUD_SCORE,NULL)),3) avg_fraud,
       ROUND(AVG(IFF(LABEL=0,FRAUD_SCORE,NULL)),3) avg_normal
FROM ML_FRAUD_PRODUCTION.ANALYTICS.PREDICTIONS;   -- ~0.91 vs ~0.13
```

> Demo fallback: if GitHub connectivity/approval isn't ready, run the same promotion locally
> from [TERM] — it exercises identical logic (runs the ML Job under `ML_DEPLOY_SVC`):
> `ML_ENV=dev .venv/bin/python demo/03_submit_promote_job.py --dev-version V2`

---

## 4. Pitfalls & fixes

| Symptom | Cause | Fix |
|---------|-------|-----|
| Promotion ML Job takes minutes to start | Compute-pool / image cold start | Run `demo_kickoff.py` (`--warm-image`) beforehand |
| "compute pool busy (1/1 nodes)" | `MLOPS_CPU_M_POOL` is single-node; back-to-back jobs serialize | Only run one promotion at a time; or bump pool `MAX_NODES=2` before the demo |
| GitHub Action can't auth | Missing `production` env or `SNOWFLAKE_ACCOUNT` variable (pipeline auth is keyless OIDC — no credential secret) | Add the `production` environment + a `SNOWFLAKE_ACCOUNT` **variable** in repo settings; SUBJECT in `SVC_ML_DEPLOY` must match `repo:<your-org>/<your-repo>:environment:production` |
| Model load fails "owner only" | `mv.load()` needs the owner (dev) role as *primary* | Already handled — `promote_model.py` switches to `ML_DEV_ROLE` for load, back to `ML_DEPLOY_SVC` to write |
| "version V2 already existed" | Re-promoting an existing prod version | Already handled — promotion skips re-log if the prod version exists |
| Experiment run won't resume | A run name already ended | Already handled — scripts `delete_run` before `start_run` |
| Batch task didn't run | It's created **suspended** on purpose | For a live scored run: `EXECUTE TASK ML_FRAUD_PRODUCTION.ANALYTICS.SCORE_BATCH_TASK;` or `ALTER TASK ... RESUME` |
| Online / Postgres cost worry | The optional online path (setup/online/) runs 24/7 | Not used in this demo. Never run it unless showing real-time; teardown covers it |

**Note:** batch scoring uses **no PAT** — the SNOWFLAKE_PAT is only for the optional online REST path.

---

## 5. Reset (run after the demo)

**[TERM]**
```bash
.venv/bin/python reset/teardown.py          # dry run (shows what it will drop)
.venv/bin/python reset/teardown.py --yes    # execute
```
Drops **both** databases, the roles, the OIDC service user, and the auth policy; leaves the
shared compute pool. Includes an orphaned-Postgres safety net (for the optional online path).
The demo currently stands with minimal idle cost (a few daily-lag Dynamic Tables; batch task
suspended; no Postgres online) — so you can leave it up between rehearsals if you prefer.

---

## 6. Deep-dive talking points (explain it like a data scientist)

### The dataset — IBM AML HI-Small (synthetic)
- ~5.08M transactions across ~518K accounts in a **multi-bank** ecosystem, with `IS_LAUNDERING`
  labels and a companion file of injected laundering **typologies** (fan-out, structuring, cycles).
- **Extreme class imbalance:** ~**0.089%** of transactions are laundering. This is realistic and
  it dictates everything about how we model and measure.
- **Synthetic on purpose:** the demo is about the *ML process*, not the data — so there's nothing
  sensitive, which is exactly why dev gets full read access with no masking.
- **Data hygiene:** we trim a trailing all-laundering tail ("Marco quirk") so a chronological
  train/test split can't leak the date as a predictor.

### The ML problem & why we measure it this way
- Binary classification on a ~0.1% base rate. **Accuracy is useless** (predict "never fraud" =
  99.9% accurate and catches nothing). We report **PR-AUC** and **recall@1%** (what share of
  fraud we catch in the top 1% of scored transactions — the alert budget an investigations team
  actually has).
- **Chronological split** (train on earlier, test on later) — no look-ahead leakage, mirrors
  production where you score the future.

### The model
- **HistGradientBoostingClassifier** (scikit-learn). Chosen over XGBoost deliberately: no native
  `libomp` dependency, so it's portable and runs anywhere (including the ML Job runtime).
- **Class imbalance** handled with `sample_weight` (positives weighted ~100:1).
- **Monotonic constraints** (`monotonic_cst=+1`) on risk-increasing features: more amount /
  cross-border / cross-currency / velocity / fan-out must never *decrease* the risk score. This
  removes counterintuitive decision boundaries and makes the model behave sensibly for a probing
  audience.
- The output is an **uncalibrated relative risk score**, not a calibrated probability (we
  rebalanced training). Frame it as "rank/triage risk," not "P(fraud)=x".

### The signals / features (three groups)
1. **Request-context** (per transaction, from the ABT): `AMOUNT_PAID`, `IS_CROSS_CURRENCY`,
   `IS_CROSS_BORDER`, `IS_HIGH_RISK_FORMAT`, `AMOUNT_TO_AVG_RATIO`, and one-hot `PAYMENT_FORMAT`.
2. **Account profile** (lifetime aggregates, `ACCOUNT_PROFILE` FV): average/stddev/max amount,
   distinct receivers / receiver-banks / countries, foreign-currency share, high-risk-format share.
3. **Engineered risk** (the V2 additions, `ACCOUNT_RISK` FV): `HIST_AMOUNT_CV` (spend volatility =
   std/avg) and `HIST_RECEIVER_FANOUT` (distinct receivers / txn count). **Fan-out is the classic
   money-laundering tell** — a mule spraying funds to many receivers — which is why adding it
   lifts the model.
- On "did it just learn ACH = fraud?": no. In this data ~99% of ACH is normal; the model learns
  *velocity + dispersion + new-account* patterns, and the same channel with an established profile
  scores low. (Feature importance / the V1→V2 delta backs this up.)

### Why batch (default) and when online
- **Batch** = a scheduled task scores recent events into `PREDICTIONS`. No standing infrastructure
  → cheap, and it's how most AML monitoring actually runs (periodic scoring + case queue).
- **Online** (Postgres-backed feature store + real-time REST + SPCS) exists under `setup/online/`
  for sub-second scoring, but runs 24/7 and costs money — so it's opt-in, not part of this demo.

### The Snowflake ML Ops capabilities on display
- **Feature Store** — governed, reusable feature views (Dynamic Tables) over the ABT.
- **Model Registry** — versioned models, per-env (dev vs prod) registries, default version.
- **Experiment Tracking** — params + metrics per run, side-by-side comparison, reproducibility.
- **ML Jobs** — the promotion runs as a **server-side job on a compute pool** (operationalized,
  not a notebook); dependencies pinned via the job's `requirements.txt`.
- **Keyless CI/CD (OIDC / Workload Identity Federation)** — GitHub's OIDC token is validated by
  Snowflake directly; no key pair or secret stored. The modern replacement for key-pair service
  accounts.
- **RBAC governance** — the dev/prod boundary is enforced on *writes*: dev reads prod, only the
  service account deploys, and a GitHub environment approval adds a human gate.

---

## 7. Likely questions (quick answers)
- **"Is this real-time?"** Batch by default; the online/real-time path exists and is a config
  switch, but it carries 24/7 cost so we don't run it here.
- **"Can a data scientist just deploy?"** No — RBAC blocks dev from writing prod; only the OIDC
  service account can, and only through the approved GitHub workflow.
- **"Why synthetic data / no masking?"** The demo is about the ML process; the data isn't
  sensitive, which lets us show the (more interesting) *deploy* governance instead of masking.
- **"What did adding a feature actually buy?"** V1→V2: higher PR-AUC (~+0.03) and recall@1%
  (~+0.02) — shown live in the experiment comparison.
- **"What does promotion actually move?"** The exact trained artifact (loaded from dev, logged to
  prod), plus the same transform + feature definitions, plus the batch scoring task.

---

## 8. Command & file reference

| Purpose | Command |
|---------|---------|
| Pre-warm | `ML_ENV=dev python pre_demo/kickoff.py [--warm-image]` |
| Build (fresh) | `00_rbac` (dev) → `01_load_data_prod` (prod) → `transforms/base_features` (dev) → `02_feature_store` (dev) → `03_train_register --version V1` (dev) |
| Live retrain | Open `demo/02_add_feature_retrain.ipynb` in the Workspace → Run all |
| Promote (GitHub) | Actions → "Promote model to production" → Run workflow (input `V2`) → approve |
| Promote (local fallback) | `ML_ENV=dev python demo/03_submit_promote_job.py --dev-version V2` |
| Reset | `python reset/teardown.py --yes` |

Key objects: `ML_FRAUD_PRODUCTION`, `ML_FRAUD_DEV_SANDBOX`; roles `ML_DEV_ROLE`, `ML_DEPLOY_SVC`;
service user `SVC_ML_DEPLOY` (OIDC); model `AML_FRAUD_GBM`; experiment `AML_FRAUD_TRAINING`;
predictions `ML_FRAUD_PRODUCTION.ANALYTICS.PREDICTIONS`.

---

## Appendix B — Optional online / real-time branch

**Only for showing real-time serving. It costs money (Postgres online store bills 24/7) and
provisions slowly — never stand it up live.** The core MLOps story lands fully on batch; this is
an add-on flourish.

**Where it lives (three pieces):**
- `setup/online/setup_online.py` — the build: Postgres online feature store + streaming velocity FV.
- `pre_demo/enable_online.py` — **enable + verify**, run the **day before**.
- `demo/optional_online_realtime.ipynb` — the optional **Act 6** notebook.

**Enable (day before) [TERM]:**
```bash
ML_ENV=dev SNOWFLAKE_PAT=... .venv/bin/python pre_demo/enable_online.py --yes
```
Provisions the online service, polls until `RUNNING` (can take up to ~an hour), and does a
smoke read. Leave it running until after the demo.

**Notebook prerequisites (in the notebook's Service settings) [WS]:**
- Container runtime; the online service `RUNNING`.
- Attach an **External Access Integration** + a **PAT secret**; set `PAT_SECRET` in the notebook
  to that secret's normalized path.

**Act 6 (optional) [WS]:** open `demo/optional_online_realtime.ipynb`, Run all. It shows a
**millisecond point lookup**, a live ingest burst with **sub-2s stream freshness**, and a
**real-time score** off online-served features.

**Teardown (mandatory to stop cost) [TERM]:** `python reset/teardown.py --yes` drops the online
service (with an orphaned-Postgres safety net).

> Note: this notebook is a **scaffold** built from the proven real-time patterns; rehearse it once
> after enabling the online service, since it can't be validated until the service is up. The
> batch model scores off profile/risk features; the velocity FV here demonstrates the online
> store's streaming freshness (a fuller real-time model that consumes velocity is a further step).
