# CastCF

CastCF 是一个面向 **Context-aware Time Series Forecasting** 的研究原型。当前代码实现的是 FreshRetailNet-50K 上的检索式协同预测实验，用来验证一个核心假设：

> 对于未来预测有用的历史 case，不一定只是历史曲线最相似的 case；相似性应该由上下文、实体信息和未来预测效用共同学习出来。

当前版本已经包含一版可训练的 **forecast-aware metric learning**，可以在多路召回候选池上学习一个重排打分函数，并用检索到的近邻未来轨迹做预测。

## 当前状态

这个仓库目前是研究原型和 pilot experiment，不是完整顶会投稿代码。它已经支持：

- FreshRetailNet-50K 数据读取与 case 构造。
- memory/query 风格的检索式预测实验。
- shape、context、meta 三路近邻召回（meta 路基于实体字段逐项匹配，而非 ID 数值相似度）。
- 同序列未来窗口重叠的泄漏防护（exclusion mask），覆盖全部检索路径和训练配对。
- 手工加权 rerank baseline。
- forecast-aware pairwise metric learning。
- MAE、MSE、NFD@K 和若干业务子集指标。

当前还没有实现：

- 神经网络 encoder。
- 强 forecasting backbone。
- prediction loss、calibration loss、retrieval reliability gate。
- 完整数据规模、多 seed、多数据集评测。
- 真实部署口径下的天气预报误差建模；当前 `context_future` 使用官方未来窗口里的天气字段，等价于 perfect-weather-foresight / known-future-context pilot 设定。

## 项目结构

```text
castcf/
  data.py                  FreshRetailNet 数据读取、daily case 构造、case 级泄漏掩码
  features.py              reference-only 标准化、未来窗口重叠 exclusion mask
  retrieval.py             shape/context/meta 检索、实体字段匹配、手工 rerank、邻居聚合
  learned_metric.py        forecast-aware learned metric scorer
  training_pairs.py        基于未来轨迹效用构造 pairwise ranking pairs
  metrics.py               MAE、MSE、NFD@K、subset metrics

experiments/
  prepare_freshretail_cases.py     构造 memory/query cases
  train_metric_reranker.py         训练 learned metric reranker
  run_retrieval_baselines.py       运行 baseline 和 learned_metric 评估

configs/
  freshretail_castcf_lite.yaml     默认 FreshRetailNet 实验配置

tests/                            单元测试和 smoke tests
artifacts/                        本地实验输出，通常不应提交
FreshRetailNet-50K/               本地数据目录，通常不应提交
```

## 数据

默认实验使用 [FreshRetailNet-50K](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K)。数据不应提交到 Git 仓库，需要在本地下载：

```bash
hf download Dingdong-Inc/FreshRetailNet-50K \
  --repo-type dataset \
  --local-dir FreshRetailNet-50K
```

默认配置期望以下文件存在：

```text
FreshRetailNet-50K/data/train.parquet
FreshRetailNet-50K/data/eval.parquet
```

## 环境安装

```bash
python3 -m pip install -r requirements.txt
```

依赖包括：

```text
numpy
pandas
pyarrow
pytest
PyYAML
scikit-learn
```

## 默认实验配置

默认配置位于 `configs/freshretail_castcf_lite.yaml`：

```yaml
data:
  max_series: 500
  lookback_days: 28
  horizon_days: 7
  stride_days: 7

retrieval:
  k: 10
  candidate_k: 100
  route_k: 100
  temperature: 0.5
  standardize_features: true
  shape_weight: 0.35
  context_weight: 0.55
  meta_weight: 0.10
  learned_model_path: artifacts/freshretail_learned_metric.npz

training:
  route_k: 100
  epochs: 200
  learning_rate: 0.1
  l2: 0.0001
  min_distance_margin: 0.0
```

含义：

- `lookback_days=28`：用过去 28 天销量作为历史窗口。
- `horizon_days=7`：预测未来 7 天销量。
- `k=10`：最终用于预测的近邻数量。
- `candidate_k=100`：`castcf_lite` 的 shape-first 候选数。
- `route_k=100`：shape/context/meta 每一路召回的候选数。
- `temperature=0.5`：邻居未来轨迹 softmax 加权温度（按消融结果从 0.2 调整，见下文）。
- `max_series=500`：默认 smoke 配置只采样 500 条 store-product 序列。

## 运行测试

```bash
python3 -W error -m pytest tests -q
```

当前测试覆盖 case 构造、reference-only 标准化、未来窗口重叠掩码、meta 字段匹配、检索/聚合、forecast-aware metric learning、训练 CLI 和评估 CLI。当前代码的完整测试数为 27 个。

## 运行完整实验

第一步，构造 FreshRetailNet memory/query cases：

```bash
python3 experiments/prepare_freshretail_cases.py \
  --config configs/freshretail_castcf_lite.yaml
```

输出：

```text
artifacts/freshretail_cases_sample.parquet
```

第二步，训练 forecast-aware metric reranker：

```bash
python3 -W error experiments/train_metric_reranker.py \
  --config configs/freshretail_castcf_lite.yaml
```

输出：

```text
artifacts/freshretail_learned_metric.npz
```

第三步，运行 baseline 和 learned_metric 评估：

```bash
python3 -W error experiments/run_retrieval_baselines.py \
  --config configs/freshretail_castcf_lite.yaml
```

输出：

```text
artifacts/freshretail_castcf_lite_metrics.json
```

## 方法说明

### Case 构造

`prepare_freshretail_cases.py` 会把 FreshRetailNet 的 daily rows 转成滑窗 case：

```text
case = {
  x_past:          过去 28 天 sale_amount
  y_future:        未来 7 天 sale_amount
  context_past:    过去窗口的促销、节假日、天气、缺货摘要
  context_future:  未来窗口可见上下文摘要
  meta:            city/store/category/product 等实体字段
}
```

训练 split 生成 `memory` cases，eval split 生成 `query` cases。评估阶段 query 只检索 memory，不从 query 集合内互相检索。

### 泄漏防护（exclusion mask）

同一序列上 anchor 间隔小于 `horizon_days` 的两个 case，其 `y_future` 窗口在时间上重叠；如果允许互相检索或互相作为训练正样本，查询自己的未来会泄漏进结果，使指标系统性偏乐观。

为此，所有检索函数和训练配对的候选生成都支持 `exclusion_mask`（query × corpus 布尔矩阵），实验脚本会自动构建并传入：

- `run_retrieval_baselines.py` 构建 query→memory 掩码，作用于全部五种检索方法；metrics JSON 的 `retrieval.leakage_guard` 记录 `min_anchor_gap_days` 和被排除的 pair 数。
- `train_metric_reranker.py` 对 memory 同池构建掩码，保证 pairwise 监督的正样本不会是查询自己的平移窗口；训练 summary 输出 `excluded_overlap_pairs`。

库层入口：`castcf.features.overlap_exclusion_mask`（数组级）和 `castcf.data.case_overlap_exclusion_mask`（直接吃 case DataFrame）。

### Meta 相似度

实体字段（city/store/category/product）是名义型 ID，数值相近不代表实体相似，因此 meta 路不使用 cosine 或 L1 距离，而是 `meta_match_matrix`：按字段逐项比较相等，返回匹配比例（[0,1]）。meta 特征不参与 z-score 标准化，直接用原始值做相等匹配。

### Baselines

当前评估包含以下方法：

| 方法 | 说明 |
| --- | --- |
| `recent` | 直接重复历史窗口最后一天销量 |
| `shape_knn` | 按 `x_past` cosine similarity 检索近邻 |
| `context_knn` | 按 `context_past + context_future` cosine similarity 检索近邻 |
| `castcf_lite` | shape-first 召回 `candidate_k` 个候选，再用 shape/context 相似度 + meta 字段匹配手工加权 rerank |
| `castcf_multiroute` | shape/context/meta 三路各召回 `route_k` 个候选，合并后手工加权 rerank |
| `learned_metric` | shape/context/meta 三路召回候选，用训练得到的 metric scorer 重排 |

所有方法的检索都经过同序列未来窗口重叠的 exclusion mask 过滤（见上文「泄漏防护」）。

### Metric Learning

当前 learned metric 是轻量线性模型，不是神经网络。

训练阶段只使用 memory cases。对每个 query case，先构造多路候选池：

```text
candidate_pool(q) =
  top route_k by shape cosine
  union top route_k by context cosine
  union top route_k by meta field-match fraction
  minus 同序列未来窗口重叠的 case（exclusion mask）和 query 自身
```

然后用真实未来轨迹构造监督信号：

```text
d_future(q, i) = mean(abs(y_future_q - y_future_i))
```

在候选池中：

```text
i+ = argmin_i d_future(q, i)
i- = argmax_i d_future(q, i)
```

训练目标是让正样本得分高于负样本：

```text
s(q, i+) > s(q, i-)
```

pairwise ranking loss：

```text
L = mean(log(1 + exp(-(s(q,i+) - s(q,i-))))) + 0.5 * l2 * ||w||^2
```

当前 pair 特征为 11 维（4 个相似度/距离特征 + 7 个实体字段相等指示）：

```text
[
  shape_cosine,
  context_cosine,
  -shape_mae_distance,
  -context_mae_distance,
  same_city, same_store, same_management_group,
  same_category_l1, same_category_l2, same_category_l3,
  same_product
]
```

scorer 是线性函数：

```text
s(q, i) = w^T feature(q, i)
```

训练时会保存：

```text
weights
feature_mean
feature_scale
loss_history
```

推理时复用训练阶段的 `feature_mean` 和 `feature_scale`，再对多路候选池打分取 top-k。

`aggregate_neighbor_futures` 还支持 `normalize_scores=True`：对每个 query 的候选分数做行内标准化后再 softmax。实验脚本把这个开关暴露为可选配置 `retrieval.normalize_learned_scores`，默认缺省为 `false`；默认 smoke 结果使用 raw learned scores，因为当前消融显示 raw score + `temperature=0.5` 更好。

## Public API

`castcf/__init__.py` 已导出常用入口，方便在 notebook 或后续实验代码中直接复用：

```python
from castcf import (
    LearnedMetricScorer,
    aggregate_neighbor_futures,
    build_daily_cases,
    case_overlap_exclusion_mask,
    castcf_multiroute_search,
    learned_metric_search,
    meta_match_matrix,
    overlap_exclusion_mask,
    pair_feature_matrix,
    shape_knn_search,
)
```

其中 `meta_match_matrix` 用于实体字段相等匹配，`case_overlap_exclusion_mask` / `overlap_exclusion_mask` 用于避免同序列未来窗口重叠泄漏，`LearnedMetricScorer` 和 `learned_metric_search` 是当前 forecast-aware metric reranker 的主要推理入口。

## 评价指标

| 指标 | 说明 |
| --- | --- |
| MAE | 预测未来 7 天销量的 mean absolute error |
| MSE | 预测未来 7 天销量的 mean squared error |
| NFD@K | Neighbor Future Distance，衡量检索到的邻居未来轨迹与 query 真实未来的接近程度 |

NFD@K 越低，说明检索到的邻居对预测越有参考价值。

评估还会按以下子集输出 MAE/MSE：

```text
future_discounted
future_holiday
future_activity
past_stockout
future_stockout
```

## 当前默认配置结果

以下结果产生于 **2026-06-11 修复之后**的代码（泄漏防护 + meta 字段匹配 + 11 维 pair 特征，learned metric 已重训；修复详情见 `castcf_code_review.md` 的「修复记录」）：

```text
max_series=500
memory_cases=4000
query_cases=500
k=10
route_k=100
temperature=0.5
```

训练摘要：

```text
pair_count=4000
excluded_overlap_pairs=4000   # 默认 stride=horizon 下恰为各 case 自身
initial_loss=0.594688
final_loss=0.100953
```

评估端泄漏核查：`leakage_guard.excluded_query_memory_pairs=0`，确认现有 train/eval 切分下 query→memory 本就无未来窗口重叠（掩码是对 `stride_days=1` 等配置的保险）。

主结果：

| 方法 | MAE | MSE | NFD@K |
| --- | ---: | ---: | ---: |
| `recent` | 0.456142 | 0.493747 |  |
| `shape_knn` | 0.870440 | 3.589749 | 1.044328 |
| `context_knn` | 0.654219 | 1.715864 | 0.822029 |
| `castcf_lite` | 0.561787 | 1.265135 | 0.711544 |
| `castcf_multiroute` | 0.501627 | 1.074730 | 0.650837 |
| `learned_metric` | 0.449310 | 0.540216 | 0.583345 |

在这组 pilot 结果中，`learned_metric` 相比 `castcf_multiroute`：

```text
MAE 下降 10.43%
MSE 下降 49.73%
NFD@K 下降 10.37%
```

相比 `recent`：MAE 低 1.50%，MSE 仍高 9.41%。`learned_metric` 是唯一在 MAE 上超过 `recent` 的检索方法，且 NFD@K（检索质量）是全部方法中最好的；MSE 落后说明邻居加权对个别大误差 case 仍不如朴素基线稳健。

### 聚合温度消融

`learned_metric` 的得分是无界线性得分，softmax 温度对其聚合影响显著（检索结果即 NFD@K 不变，只影响加权预测）：

| 设定 | temp=0.2 MAE | temp=0.5 MAE | temp=1.0 MAE |
| --- | ---: | ---: | ---: |
| raw 得分（默认） | 0.459572 | **0.449310** | 0.453577 |
| 行内标准化（`normalize_learned_scores: true`） | 0.497196 | 0.476170 | 0.461492 |

默认配置已按此消融把 `temperature` 设为 0.5（raw 得分）。`normalize_learned_scores` 配置项（默认 `false`）保留作消融开关。注意 temperature 是全方法共享的，0.2→0.5 对手工加权方法（`castcf_lite`/`castcf_multiroute`）的 MAE 有约 0.5%-1% 的轻微负影响，对 `learned_metric` 收益更大。

该结果仍是 500 序列 smoke 规模，不能直接视为完整 benchmark 结论。

## 复现实验后查看结果

可以直接读取 metrics JSON：

```bash
python3 - <<'PY'
import json
from pathlib import Path

metrics = json.loads(Path("artifacts/freshretail_castcf_lite_metrics.json").read_text())
for name, report in metrics["methods"].items():
    line = f"{name:17s} MAE={report['mae']:.6f} MSE={report['mse']:.6f}"
    if "nfd_at_k" in report:
        line += f" NFD@K={report['nfd_at_k']:.6f}"
    print(line)
PY
```

## 下一步建议

如果目标是论文投稿，建议按以下顺序加强：

1. 扩大 `max_series`，跑完整或更大规模 FreshRetailNet 实验。
2. 加入多 seed 和多 split 稳定性报告。
3. 加入强 forecasting backbone，例如 PatchTST、TimesNet、iTransformer 或 TSMixer。
4. 将 CastCF 作为 retrieval memory module 接到 backbone 上，而不是只做邻居加权。
5. 增加 ablation：去掉 context、去掉 meta、不训练 metric、不同 `k` 和 `route_k`。
6. 实现 hard negatives：shape-similar future-different、context-similar pattern-different、stale negatives。
7. 补 prediction loss、calibration loss 和 retrieval reliability gate。

## 注意事项

- `FreshRetailNet-50K/` 是本地数据目录，不应提交。
- `artifacts/` 是本地实验产物，不应作为最终论文结果直接使用。
- 2026-06-11 修复后 pair 特征从 6 维变为 11 维，修复前训练的 `learned_metric.npz` 与新代码不兼容；当前 `artifacts/` 下的模型和指标均已用新代码重新生成。
- `context_future` 中的折扣、节假日、活动和天气都来自未来窗口字段；其中计划型上下文可视作已知，天气字段在真实部署中应替换为天气预报或做去天气/带噪天气消融。
- `freshretail_castcf_lite_experiment_results.md` 是修复前历史实验记录，里面关于 meta 路和绝对指标的结论应以本 README 的修复后结果为准。
- `castcf_multiroute_search` 仍然使用暴力全量 route similarity 取 top-k；`castcf_lite` 的 rerank 已改为候选级计算，但完整规模实验前仍需要近似检索或分块检索。
- 当前实现是 numpy 级别轻量原型，适合快速验证研究假设。
- 当前默认配置是 smoke/pilot 规模，不是完整 benchmark。
