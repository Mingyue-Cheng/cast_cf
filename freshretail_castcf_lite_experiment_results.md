# FreshRetailNet-50K 上的 CastCF-lite 初版实验结果

更新时间：2026-06-02

## 1. 实验目的

本实验是 `CastCF / CoCast` 在 FreshRetailNet-50K 上的第一版可运行原型。目标不是证明最终方法已经优于所有 baseline，而是先验证三件事：

1. FreshRetailNet-50K 能否被整理成 CastCF 需要的 forecasting case。
2. 是否能在该数据集上跑通 retrieval-based forecasting baseline。
3. context-aware retrieval 是否已经表现出比 shape-only retrieval 更强的信号。

当前版本是 **CastCF-lite**，还没有训练神经网络，也没有加入 forecast-aware ranking loss。

## 2. 数据集规模、切分与 case 构造

数据路径：

```text
FreshRetailNet-50K/data/train.parquet
FreshRetailNet-50K/data/eval.parquet
```

### 2.1 原始数据规模

FreshRetailNet-50K 是一个生鲜零售需求预测数据集。当前本地下载版本包含官方 `train` 和 `eval` 两个 split：

| Split | 行数 | store-product 序列数 | stores | products | cities | 日期范围 | 天数 |
|---|---:|---:|---:|---:|---:|---|---:|
| `train` | 4,500,000 | 50,000 | 898 | 865 | 18 | 2024-03-28 至 2024-06-25 | 90 |
| `eval` | 350,000 | 50,000 | 898 | 865 | 18 | 2024-06-26 至 2024-07-02 | 7 |

数据中包含的主要 context / event 信号如下：

| Split | discount<1 比例 | holiday 比例 | activity 比例 | stockout day 比例 |
|---|---:|---:|---:|---:|
| `train` | 51.53% | 34.44% | 37.84% | 44.27% |
| `eval` | 49.49% | 28.57% | 32.97% | 40.98% |

这些比例说明该数据集不是只有少量稀疏事件，而是有足够多的促销、节假日、活动、缺货和天气相关变化，适合检验 context-aware retrieval。

### 2.2 本次实验的 train / validation / test 切分

当前实验是第一版 smoke experiment，没有训练神经网络参数，也没有用验证集调参。因此切分方式如下：

| 角色 | 来源 | 用法 |
|---|---|---|
| Training / Memory | 官方 `train` split | 构造 retrieval memory cases，作为历史案例库 |
| Validation | 当前版本未使用 | 当前 `shape_weight/context_weight/meta_weight` 固定写在配置中，没有基于 validation 调参 |
| Test / Query | 官方 `eval` split | 构造 query cases，评估未来 7 天预测效果 |

重要说明：当前 `eval` 被视作 test set。后续如果要做 grid search 或 learned metric training，应从官方 `train` 内部再切出 validation，例如用 `train` 的最后 7 天构造 validation query，避免用官方 `eval` 反复调参造成 test contamination。

### 2.3 当前 smoke 实验配置

```text
max_series: 500
seed: 42
lookback_days: 28
horizon_days: 7
stride_days: 7
```

本次只抽样 500 条 store-product 序列，用于快速验证代码和实验闭环。抽样后，每条序列在官方 `train` 90 天内按 `lookback=28, horizon=7, stride=7` 生成 8 个 memory cases，因此共有 4,000 个 memory cases；官方 `eval` 只有 7 天，每条序列生成 1 个 query case，因此共有 500 个 query cases。

构造结果：

| Split | 用途 | Case 数 |
|---|---|---:|
| train | memory cases | 4,000 |
| eval | query cases | 500 |

case schema：

```text
X_past      = 过去 28 天 sale_amount
Y_future    = 未来 7 天 sale_amount
C_past      = past discount / holiday / activity / weather / stockout summary
C_future    = future discount / holiday / activity / weather summary
M           = city_id / store_id / category ids / product_id
```

注意：`future_stockout` 没有作为输入 context 使用，只作为 subset/诊断标签，避免直接泄漏未来缺货状态。

## 3. 实验 setting

### 3.1 运行命令

单元测试命令：

```bash
python3 -m pytest tests -q
```

验证结果：

```text
13 passed in 0.29s
```

真实数据实验命令：

```bash
python3 experiments/prepare_freshretail_cases.py --config configs/freshretail_castcf_lite.yaml
python3 -W error experiments/run_retrieval_baselines.py --config configs/freshretail_castcf_lite.yaml
```

输出文件：

```text
artifacts/freshretail_cases_sample.parquet
artifacts/freshretail_castcf_lite_metrics.json
```

### 3.2 评价指标

本实验使用两类指标。

第一类是预测误差：

| 指标 | 定义 | 越低越好 |
|---|---|---|
| MAE | `mean(abs(Y_true - Y_pred))`，在所有 query 和 7 天 horizon 上平均 | 是 |
| MSE | `mean((Y_true - Y_pred)^2)`，在所有 query 和 7 天 horizon 上平均 | 是 |

第二类是 retrieval quality：

| 指标 | 定义 | 含义 |
|---|---|---|
| `NFD@K` | query 真实未来 `Y_q` 与 Top-K 邻居真实未来 `Y_i` 的平均绝对距离 | 衡量检索到的历史案例是否真的 forecast-useful |

`NFD@K` 比 MAE/MSE 更贴近 CastCF 的核心 claim。即使某个 aggregation 暂时预测不好，只要 `NFD@K` 下降，也说明检索模块正在找到更接近当前未来的历史案例。

### 3.3 对比方法

| 方法 | 含义 |
|---|---|
| `recent` | 直接用 lookback 最后一天的销量重复 7 天，作为简单 recent baseline |
| `shape_knn` | 只用 `X_past` 的历史销量形状做 kNN retrieval |
| `context_knn` | 只用 context 表征做 kNN retrieval |
| `castcf_lite` | 先用 shape 做 coarse retrieval，再用 shape/context/meta 加权 rerank，并聚合邻居未来 |
| `castcf_multiroute` | 分别从 shape/context/meta 三路取候选，合并候选池后再用加权 score rerank |

检索配置：

```text
k: 10
candidate_k: 100
route_k: 100
temperature: 0.2
standardize_features: true
shape_weight: 0.35
context_weight: 0.55
meta_weight: 0.10
```

## 4. 实验结果

| 方法 | MAE | MSE | NFD@K |
|---|---:|---:|---:|
| `recent` | 0.456142 | 0.493747 | - |
| `shape_knn` | 0.872817 | 3.601460 | 1.044328 |
| `context_knn` | 0.652710 | 1.694710 | 0.822029 |
| `castcf_lite` | 0.586405 | 1.362452 | 0.749033 |
| `castcf_multiroute` | 0.511730 | 1.155460 | 0.660868 |

其中 `NFD@K` 表示被检索邻居的未来轨迹与 query 真实未来轨迹之间的平均距离。数值越低，说明检索到的邻居越 forecast-useful。

## 5. 结果分析

### 5.1 实验闭环已经成立

当前代码已经能跑完整实验闭环：从 FreshRetailNet-50K 构造 memory/query cases，到运行 retrieval baselines，再到输出预测误差和 retrieval quality 指标。单元测试和真实数据 smoke experiment 都能通过，说明当前代码可以作为后续 CastCF 实验的起点。

### 5.2 context retrieval 信号明显强于 shape-only retrieval

`context_knn` 仍然优于 `shape_knn`：

```text
shape_knn   MAE = 0.872817, NFD@K = 1.044328
context_knn MAE = 0.652710, NFD@K = 0.822029
```

这说明在当前 500 条序列抽样上，FreshRetailNet-50K 的 context 信号确实很强。只靠历史销量形状检索邻居，会找到未来不够接近的历史 case；而 context-based retrieval 能找到更接近 query future 的邻居。

这对 CastCF 很关键，因为它支持了项目的核心问题设定：

> 历史曲线最像的 case，不一定是当前情境下最有预测价值的 case。

从 `NFD@K` 看，`context_knn` 检索到的邻居未来轨迹明显更接近 query future。这说明 FreshRetailNet 上确实存在“context 决定哪个历史案例更可参考”的现象。

### 5.3 recent baseline 很强，整体 MAE 不是唯一重点

`recent` baseline 当前最强：

```text
recent MAE = 0.456142
```

这说明官方 `eval` 的 7 天测试窗口里，许多序列具有很强的短期平稳性。简单重复 lookback 最后一天，已经能得到较低误差。因此，后续论文实验不能只用全局 MAE/MSE 证明 CastCF，而必须拆出 context-sensitive subset，例如：

- discount 变化样本
- holiday 样本
- activity 样本
- future stockout 样本
- 极端天气或温度变化样本
- `same past, different future` probe

CastCF 的优势更可能出现在这些局部困难样本中，而不是所有普通平稳样本的平均误差上。

### 5.4 标准化后 CastCF-lite 已经明显改善

```text
castcf_lite MAE = 0.586405, NFD@K = 0.749033
```

第一版实验中 `castcf_lite` 接近 `shape_knn`，主要问题是特征没有做标准化，且 shape/context/meta 的数值尺度直接影响 cosine 与 rerank。当前代码已改成用 memory split 的统计量对 `X_past`、context、meta 分别标准化，再做检索和 rerank。

标准化后，`castcf_lite` 已经优于 `shape_knn` 和 `context_knn`：

```text
shape_knn   MAE = 0.872817, NFD@K = 1.044328
context_knn MAE = 0.652710, NFD@K = 0.822029
castcf_lite MAE = 0.586405, NFD@K = 0.749033
```

这说明 shape 与 context 的组合确实有价值，但前提是特征处理要合理。

### 5.5 当前结果的研究含义

当前结果最有价值的信号是：

```text
castcf_multiroute 的 NFD@K 进一步低于 context_knn 和 castcf_lite
```

这说明 context 本身能帮助找到更 forecast-useful 的历史邻居，而 shape/context/meta 多路候选合并比单一路径更稳。换句话说，FreshRetailNet-50K 不仅能支撑 CastCF 的核心 claim，也开始支持一个更具体的方法设计：**不要只做 shape-first retrieval，而要做多路候选召回 + forecast-aware rerank。**

### 5.6 更细的定量对比

从 `shape_knn` 到 `context_knn`，两个关键指标都有明显改善：

| 对比 | MAE 变化 | MSE 变化 | NFD@K 变化 |
|---|---:|---:|---:|
| `context_knn` vs `shape_knn` | 下降约 25.2% | 下降约 52.9% | 下降约 21.3% |
| `castcf_lite` vs `context_knn` | 下降约 10.2% | 下降约 19.6% | 下降约 8.9% |
| `castcf_multiroute` vs `castcf_lite` | 下降约 12.7% | 下降约 15.2% | 下降约 11.8% |

这说明改进不是只在最终预测聚合上偶然变好，而是在 retrieval quality 上也同步变好。`castcf_multiroute` 的 `NFD@K=0.660868`，说明它检索到的邻居未来轨迹最接近 query future。

但 `context_knn` 仍然弱于 `recent`：

| 对比 | MAE |
|---|---:|
| `recent` | 0.456142 |
| `castcf_multiroute` | 0.511730 |

这说明当前 eval 窗口中，“最近一天销量”对未来 7 天仍然是非常强的近邻信号。`castcf_multiroute` 已经把 MAE 从 `context_knn` 的 0.652710 拉到 0.511730，接近但尚未超过 `recent`。因此下一步不能只盯全局 MAE，而要看 retrieval 是否在特定 context shock 下有优势。

### 5.7 subset 结果说明

当前 metrics JSON 中已经输出了若干 subset 的 MAE/MSE。几个值得注意的观察如下。

| Subset | count | recent MAE | shape-kNN MAE | context-kNN MAE | CastCF-lite MAE | CastCF-multiroute MAE |
|---|---:|---:|---:|---:|---:|---:|
| all | 500 | 0.456142 | 0.872817 | 0.652710 | 0.586405 | 0.511730 |
| future_discounted | 369 | 0.457719 | 0.895498 | 0.683751 | 0.612605 | 0.531862 |
| future_activity | 225 | 0.414584 | 0.743823 | 0.642992 | 0.542061 | 0.481753 |
| past_stockout | 494 | 0.455783 | 0.875687 | 0.656135 | 0.585191 | 0.511720 |
| future_stockout | 436 | 0.446857 | 0.818325 | 0.591887 | 0.532444 | 0.473266 |

第一，`future_discounted` 和 `future_activity` subset 中，`castcf_multiroute` 均优于 `shape_knn`、`context_knn` 和 `castcf_lite`。这说明多路候选机制不仅改善全局指标，也改善了促销和活动相关样本。

第二，`future_stockout` subset 中，`castcf_multiroute` 的 MAE 为 0.473266，已经非常接近 `recent` 的 0.446857，并明显低于 `context_knn` 的 0.591887。这提示多路 retrieval 可能更容易找到缺货/需求受限状态下的相似未来。不过需要谨慎解释，因为 `future_stockout` 没有作为输入 context，只是 evaluation subset；如果后续要研究缺货场景，应该把它作为诊断维度，而不是预测时可用特征。

第三，`past_stockout` 的 count 是 494/500，`future_stockout` 的 count 是 436/500，覆盖率很高。这说明 stockout 在这个抽样里太普遍，直接用 `stockout vs non-stockout` 作为二分 subset 区分度不够。后续更合理的做法是按 stockout 强度分层，例如：

```text
no stockout
low stockout hours
medium stockout hours
high stockout hours
```

第四，`future_holiday` 在当前 query 中 count 是 500/500，说明 7 天 eval horizon 内每个 query 都覆盖了 holiday flag。因此这个 subset 在当前切分下没有区分能力，不能用来证明 holiday-specific improvement。后续需要构造更细的 holiday probe，例如比较：

```text
horizon 内无 holiday
horizon 中含 holiday
holiday 前后窗口
holiday 影响强的品类/城市
```

### 5.8 为什么 multiroute 有效

第一版 `castcf_lite` 的设计是：

```text
先按 shape 找 candidate_k=100
再用 shape/context/meta 固定权重 rerank
```

这会产生一个瓶颈：如果第一阶段 shape coarse retrieval 已经漏掉了真正 context-useful 的邻居，后续 rerank 无法把这些邻居找回来。

本轮代码加入了 `castcf_multiroute`：

```text
shape candidates ∪ context candidates ∪ entity/meta candidates
    -> weighted rerank
```

结果显示该设计直接改善了 MAE、MSE 和 `NFD@K`。这说明 CastCF 后续方法应坚持“多路候选召回”，再进一步把 fixed weighted rerank 替换成 learned forecast-aware reranker。

### 5.9 当前实验对论文 claim 的支持程度

当前实验能支持的 claim：

1. FreshRetailNet-50K 可以被整理成 CastCF 所需的 historical case memory。
2. context-based retrieval 比 shape-based retrieval 找到更 forecast-useful 的邻居。
3. 多路候选召回比 shape-first rerank 更适合 CastCF。
4. 只看全局 forecasting error 不足以证明 CastCF，需要加入 retrieval quality 和 context-sensitive diagnostic。

当前实验还不能支持的 claim：

1. 还不能说 CastCF 已经优于所有强 baseline，因为 `recent` 仍然更强。
2. 还不能说固定加权是最终方法，当前仍需要 learned forecast-aware similarity。
3. 不能说 holiday / stockout 子集已经证明了 context reasoning，因为当前 subset 覆盖率过高或不够细。

因此，这一版结果更适合作为 **stronger pilot evidence**：它不仅证明了数据和 retrieval 信号存在，也证明了多路候选召回是比 shape-first retrieval 更合理的 CastCF 实现路线。

## 6. 当前结论

当前实验支持两个判断：

1. **FreshRetailNet-50K 可以支撑 CastCF 研究。** 它有实体、多种 context、缺货事件和未来预测窗口，能构造 CastCF 所需的 forecasting cases。
2. **context retrieval 信号已经存在。** `context_knn` 的 `NFD@K` 低于 `shape_knn`，说明 context 能帮助找到更 forecast-useful 的历史邻居。
3. **多路候选召回是有效迭代。** `castcf_multiroute` 进一步降低 MAE、MSE 和 `NFD@K`，说明 CastCF 不应只依赖 shape-first candidate pool。

但当前结果还不能声称 CastCF 已经完成，因为 `recent` baseline 仍然更强，下一步需要 validation split、参数搜索和 learned forecast-aware similarity。

## 7. 下一步建议

优先做以下改进：

1. 从官方 train 内部切 validation query，用于调 `route_k` 和 `shape_weight/context_weight/meta_weight`。
2. 在 `castcf_multiroute` 上做 grid search，而不是只调 shape-first `castcf_lite`。
3. 增加 context-sensitive subset 指标表，单独看 discount、holiday、activity、future_stockout 等样本。
4. 构造 `same past, different future` probe，用于直接证明 shape-only retrieval 的失败场景。
5. 引入 learned forecast-aware similarity，用 pairwise future utility 替代手工加权。

最重要的研究信号是：

> 在 FreshRetailNet-50K 上，context retrieval 和多路候选召回都能让邻居未来更接近真实未来；下一版 CastCF 应该把这个信号学习进 forecast-aware retrieval metric，而不是停留在固定加权。
