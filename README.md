# CastCF

CastCF is an early research prototype for context-aware collaborative retrieval in time series forecasting. The current code focuses on a non-neural CastCF-lite experiment on FreshRetailNet-50K, testing whether context-aware retrieval finds more forecast-useful historical cases than shape-only retrieval.

## Project Structure

```text
castcf/                         Core data, feature, retrieval, and metric utilities
configs/freshretail_castcf_lite.yaml
experiments/                    CLI scripts for preparing cases and running baselines
tests/                          Unit and smoke tests
freshretail_castcf_lite_experiment_results.md
context_aware_tsf_datasets.md
cast_cf方案.md
```

## Data

The FreshRetailNet-50K data is not committed to this repository. Download it locally before running the experiment:

```bash
hf download Dingdong-Inc/FreshRetailNet-50K \
  --repo-type dataset \
  --local-dir FreshRetailNet-50K
```

The default config expects:

```text
FreshRetailNet-50K/data/train.parquet
FreshRetailNet-50K/data/eval.parquet
```

## Setup

```bash
python3 -m pip install -r requirements.txt
```

## Run Tests

```bash
python3 -m pytest tests -q
```

## Run FreshRetailNet CastCF-lite Experiment

```bash
python3 experiments/prepare_freshretail_cases.py \
  --config configs/freshretail_castcf_lite.yaml

python3 -W error experiments/run_retrieval_baselines.py \
  --config configs/freshretail_castcf_lite.yaml
```

The default smoke configuration samples 500 store-product series and writes:

```text
artifacts/freshretail_cases_sample.parquet
artifacts/freshretail_castcf_lite_metrics.json
```

## Current Baselines

- `recent`: repeat the last observed daily sales value over the horizon.
- `shape_knn`: retrieve neighbors by historical sales shape only.
- `context_knn`: retrieve neighbors by context features only.
- `castcf_lite`: shape-first retrieval followed by context/meta reranking.
- `castcf_multiroute`: union shape, context, and meta candidates, then rerank.

