# Presenter RUNBOOK - Real-Time Feature Store Fraud Demo

## The one-liner
"Every incoming payment is scored for fraud in milliseconds, by combining a slow-moving
account profile, fast-moving velocity features that update within two seconds of an event,
and request-time signals computed at the moment of the transaction - all served from
Snowflake's online feature store and a real-time inference endpoint."

---

## Demo flow (~15 min)  [VALIDATED END-TO-END]

Prereqs each run: `export SNOWFLAKE_CONNECTION_NAME=demo156_keypair` (the PAT is read
automatically from `pat/`). If the inference service auto-suspended (30 min idle), resume it:
`ALTER SERVICE FRAUD_RT_DEMO.FEATURE_STORE.AML_FRAUD_RT_SERVICE RESUME;` (needs ACCOUNTADMIN
or the granted pool usage). Use a FRESH new-account id each run so velocity starts at 0.

### 1. Tour what already exists (pre-built)
- `FRAUD_RT_DEMO.RAW` / `CURATED` - the IBM AML transactions, the accounts/bank/entity
  dimension, and the curated `TXN_SPINE`. Call out the **multi-bank** angle: 86% of
  transactions cross institutions.
- Model Registry: `AML_FRAUD_GBM/V2` - monotonic-constrained gradient boosting (ROC-AUC 0.96).
- Feature Store: the entity `ACCOUNT`, the `ACCOUNT_PROFILE` (batch online) and
  `ACCOUNT_VELOCITY` (stream) feature views, and the `FRAUD_FEATURES` feature group.
- The Postgres online service - the low-latency serving layer.

### 2. Register the real-time feature view LIVE
```
.venv/bin/python demo/register_realtime_fv.py
```
Talk track: "Some features can't be precomputed - they depend on the transaction in front
of you. This `TXN_RISK_SIGNALS` view runs a Python function at query time to compare the
current amount against this account's historical average. It registers instantly, no
infrastructure to provision."

### 3. Establish a NORMAL baseline (low risk)
```
.venv/bin/python demo/score_transaction.py --account 012719_8019E5AE0 --scenario normal
```
An established account doing one ordinary ACH payment scores a **low risk (~0.3)**. (Scores are
uncalibrated relative risk scores - see model notes.)

### 4. Ingest a fan-out burst -> show freshness < 2s
```
.venv/bin/python demo/stream_events.py --mode fraud --account NEWMULE_$(date +%s) --count 40
.venv/bin/python demo/query_features.py --account <that ACCOUNT_ID>
```
Talk track: "A brand-new account suddenly fans out to ~40 different banks - the single-account
shadow of a mule / structuring pattern. Within ~2 seconds the velocity features (txn count,
distinct banks in 24h) have already moved in the online store."

### 5. Score it (the payoff)
```
.venv/bin/python demo/score_transaction.py --account <that ACCOUNT_ID> --scenario fraud
```
The same new account now scores **~0.94 fraud risk** in **~700-950 ms** end-to-end (online
feature lookup + inference). Contrast with the 0.3 baseline: a clean, intuitive escalation
driven entirely by the live velocity features.

---

## HOW TO TALK ABOUT THE MODEL (read this before presenting)

**Lead with the right metrics.** Fraud is ~0.10% of transactions (about 1 in 981). Accuracy
is meaningless here - a model that predicts "never fraud" is 99.9% accurate and catches zero
fraud. We report **PR-AUC** and **recall@1%** (of all fraud, what share lands in the top 1%
of scored transactions - how an alert queue actually works). Never quote accuracy.

**Why the rate is so low is realistic.** Real banking fraud is rare; a 1-2% rate would be
implausibly high. The extreme imbalance is handled with `scale_pos_weight` + negative
downsampling, and honest time-based (chronological) train/test splitting.

**Scores are uncalibrated relative risk scores.** Because we train on rebalanced (downsampled +
weighted) data, the output is a *risk score*, not a calibrated probability - a normal txn sits
around 0.3, fraud around 0.9. That's how production fraud scoring usually works (rank + threshold,
not literal probability). If asked: we'd calibrate to the operating point for a real deployment.

**The model is monotonic-constrained (V2).** We enforce that more velocity / dispersion / amount /
cross-border activity can only *increase* fraud risk. This removes counterintuitive boundaries and
makes the model behave sensibly under probing (an earlier unconstrained version would lower the
score when cross-currency was set - a red flag we fixed). ROC-AUC ~0.96 on the chronological test.

**What it keys on (be ready for this).** The strong fraud signal is a sudden velocity / fan-out
spike on a sparse-or-new account, via the ACH channel (in this dataset, modeled laundering is
ACH-based structuring/layering - so ACH + many small-ish transfers to many banks is the
learned signature, not high-value wires). This maps to new-account / mule fraud.

**It's a demo-grade per-transaction model, by design.** The labels are *graph typologies*
(see cheat-sheet) - multi-account network patterns. Our model scores one transaction for one
account using that account's own profile + velocity. It catches the **single-account shadow**
of these patterns (fan-out -> spike in distinct banks; structuring -> velocity; layering ->
cross-currency) but cannot see the full multi-account graph. If asked "is this a
state-of-the-art AML model?": no - SOTA AML uses graph ML; that's a natural **next chapter**
(the dataset ships graph ground truth + a published GNN repo). The star here is the real-time
serving architecture, which is production-real.

**Training/serving honesty.** The model is trained on profile + velocity features. The
live-registered `TXN_RISK_SIGNALS` real-time FV is shown as a request-time feature-engineering
teaching artifact, not a model input - so registering it live needs no retrain.

**The "Marco quirk" (if data is questioned).** HI-Small's transactions after ~Sep 10 are an
artifact: the generator stopped producing normal background traffic while the trailing hops of
in-flight laundering chains kept completing, so those days are ~60% fraud. We trim everything
from Sep 11 onward so the date can't leak as a predictor, then split chronologically within
Sep 1-10.

---

## APPENDIX B - Typology cheat-sheet

A *typology* is a named multi-account **graph shape** launderers use to break the link between
money and crime. It is defined by relationships across accounts and time - not by any single
transaction. Our per-account model sees only the single-account *shadow* (right column).

| Typology | Shape | Single-account shadow our model can see |
|----------|-------|------------------------------------------|
| **Fan-Out** | one account -> many recipients/banks | spike in distinct receiver banks/accounts in 24h |
| **Fan-In** | many senders -> one collector | spike in inbound count/sum on the collector |
| **Cycle** | A->B->...->A, returns toward origin | currency-hopping + round-trip / reciprocity flags |
| **Scatter-Gather** | scatter to intermediaries, then gather to one | outbound dispersion then convergence; intermediary velocity |
| **Gather-Scatter** | pool funds in, then redistribute out | inbound burst followed by outbound burst |
| **Bipartite** | layer-A accounts only pay layer-B accounts | one-hop cross-bank volume mimicking commercial payments |
| **Stack** | stacked/chained layers, large FX jumps | high-risk-format + large cross-currency amounts in chain steps |
| **Random** | deliberately irregular multi-hop, small amounts | weak single-account signal (built to evade) -> needs a graph model |

Talking point: this is exactly why we engineered **distinct-bank / distinct-country dispersion**
and **cross-currency / high-risk-format** features - the strongest single-account proxies for
these shapes - and why capturing the full shapes is the job of a future graph-ML chapter.

---

## APPENDIX A - Future model chapter (graph ML)
The dataset ships `HI-Small_Patterns.txt` (8 typologies as connected subgraphs), the
`accounts.csv` entity/bank dimension, and IBM's published Multi-GNN repo + NeurIPS paper.
- Rung 1 (low effort): graph-derived tabular features (degree, fan-in/out ratios, distinct-bank
  breadth, 2-hop "money comes back") fed into the *same* gradient-boosting model - served from this *same*
  online feature store.
- Rung 2: graph algorithms / embeddings (PageRank, Louvain, node2vec) via ML Jobs.
- Rung 3: a true GNN (GraphSAGE / temporal GNN) on a GPU pool; HI/LI-Large for scale.
Even GNNs precompute embeddings offline and serve them from the feature store - so a future
graph chapter plugs into this same architecture.
