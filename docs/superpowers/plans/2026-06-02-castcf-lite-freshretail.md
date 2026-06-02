# CastCF-Lite FreshRetail Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a first executable CastCF-lite experiment on FreshRetailNet-50K that tests context-aware retrieval before implementing neural models.

**Architecture:** Convert FreshRetailNet daily rows into forecasting cases, run retrieval baselines over sampled cases, aggregate neighbor futures, and report prediction plus retrieval-quality metrics. The first version is intentionally non-neural so the data protocol and claim diagnostics can be verified quickly.

**Tech Stack:** Python 3, pandas, numpy, pyarrow, scikit-learn, pytest, PyYAML.

---

### Task 1: Case Construction

**Files:**
- Create: `castcf/data.py`
- Test: `tests/test_case_builder.py`

- [ ] **Step 1: Write failing tests**

Test that a compact retail dataframe is converted into cases with `X_past`, `Y_future`, context features, metadata, and subset flags.

- [ ] **Step 2: Run failing tests**

Run: `python3 -m pytest tests/test_case_builder.py -q`

Expected: fail because `castcf.data` is not implemented.

- [ ] **Step 3: Implement case construction**

Implement `build_daily_cases`, `load_freshretail_split`, `sample_series_ids`, and `save_cases`.

- [ ] **Step 4: Verify tests pass**

Run: `python3 -m pytest tests/test_case_builder.py -q`

Expected: pass.

### Task 2: Retrieval and Aggregation

**Files:**
- Create: `castcf/retrieval.py`
- Test: `tests/test_retrieval.py`

- [ ] **Step 1: Write failing tests**

Test shape-only retrieval, context-aware reranking, exclusion of same query, and weighted aggregation of neighbor futures.

- [ ] **Step 2: Run failing tests**

Run: `python3 -m pytest tests/test_retrieval.py -q`

Expected: fail because `castcf.retrieval` is not implemented.

- [ ] **Step 3: Implement retrieval**

Implement cosine similarity, shape candidate retrieval, context reranking, and neighbor future aggregation.

- [ ] **Step 4: Verify tests pass**

Run: `python3 -m pytest tests/test_retrieval.py -q`

Expected: pass.

### Task 3: Metrics

**Files:**
- Create: `castcf/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write failing tests**

Test MAE, MSE, `NFD@K`, and subset metric behavior.

- [ ] **Step 2: Run failing tests**

Run: `python3 -m pytest tests/test_metrics.py -q`

Expected: fail because `castcf.metrics` is not implemented.

- [ ] **Step 3: Implement metrics**

Implement prediction error metrics, neighbor future distance, and subset metric helpers.

- [ ] **Step 4: Verify tests pass**

Run: `python3 -m pytest tests/test_metrics.py -q`

Expected: pass.

### Task 4: Experiment CLI

**Files:**
- Create: `experiments/prepare_freshretail_cases.py`
- Create: `experiments/run_retrieval_baselines.py`
- Create: `configs/freshretail_castcf_lite.yaml`

- [ ] **Step 1: Implement prepare CLI**

Read train/eval parquet files, sample store-product series, build daily cases, and save a compact parquet artifact.

- [ ] **Step 2: Implement run CLI**

Load prepared cases, compare recent baseline, shape-kNN, context-kNN, and CastCF-lite, then write metrics JSON.

- [ ] **Step 3: Run a smoke experiment**

Run:

```bash
python3 experiments/prepare_freshretail_cases.py --config configs/freshretail_castcf_lite.yaml
python3 experiments/run_retrieval_baselines.py --config configs/freshretail_castcf_lite.yaml
```

Expected: exits 0 and writes `artifacts/freshretail_castcf_lite_metrics.json`.

### Task 5: Final Verification

**Files:**
- Modify: none unless verification exposes a defect.

- [ ] **Step 1: Run unit tests**

Run: `python3 -m pytest tests -q`

Expected: all tests pass.

- [ ] **Step 2: Run smoke experiment**

Run both CLI commands from Task 4 and inspect the metrics JSON.

Expected: metrics contain `recent`, `shape_knn`, `context_knn`, and `castcf_lite`.

