# CastCF 代码 Review（`castcf/` 包）

> Review 范围：`castcf/data.py`、`features.py`、`metrics.py`、`retrieval.py`、`training_pairs.py`、`learned_metric.py`、`__init__.py`（共约 910 行），并参考了 `experiments/run_retrieval_baselines.py` 的调用方式。
> 日期：2026-06-11

## 总体评价

代码质量整体偏高：模块边界清晰（数据构造 → 特征 → 检索 → 配对监督 → 学习度量 → 评估），纯 numpy/pandas 实现无重型依赖，输入校验细致（维度、行数、参数范围都有显式检查），数值处理上有不少正确的细节（softmax 减 max 防溢出、零向量余弦保护、reference-only 标准化避免泄漏、sigmoid clip）。`tests/` 下对每个模块都有对应测试文件，工程习惯好。

下面的问题按严重程度排序：前两条是方法论层面的，可能直接影响实验结论的有效性；其余是代码质量/性能问题。

---

## 高优先级（方法论正确性）

### 1. 对原始实体 ID 做余弦/距离相似度，在语义上不成立

`meta` 向量是 7 个原始整型 ID（`city_id`、`store_id`、`product_id` 等，`data.py:142`），但检索中直接对它们算余弦相似度（`retrieval.py:81,126,173`），学习度量里还算 `-mean|qmeta - cmeta|` 距离（`learned_metric.py:77`）。

这些 ID 是**名义变量**（nominal）：store 100 和 store 101 之间没有任何"更相似"的含义，但余弦/L1 会认为它们几乎相同。实验脚本里的 `standardize_by_reference` 也无法修复这一点——z-score 之后数值相近的 ID 仍然被当成相近实体。

**影响**：meta 路召回（`multiroute`）和 meta 相似度权重（默认 0.1）注入的是噪声相关性；学习度量的 `meta_cos`/`meta_distance` 两个特征同理。如果实验显示 meta 路有正向收益，很可能是 ID 编码顺序与某种真实结构（如同城门店 ID 连续分配）偶然相关，结论不稳。

**建议**：把 meta 相似度改成**逐字段相等指示向量**（same_city、same_store、same_category_l1/l2/l3、same_product 的 0/1 向量），对它做点积或直接作为 pairwise 特征。这既符合语义，也正好适配 `pair_feature_matrix` 的特征拼接方式。

### 2. 同池检索 + stride=1 滑窗，存在邻居未来轨迹重叠泄漏

`build_daily_cases`（`data.py:139`）默认 `stride_days=1`，同一序列相邻 anchor 的 case，其 `y_future` 有 `horizon-1` 天重叠。而 `shape_knn_neighbors` / `castcf_lite_neighbors` 做同池检索时只屏蔽了对角线自身（`retrieval.py:45`），**没有屏蔽同序列的时间相邻 case**。

**影响**：查询 case 最容易召回的就是自己前后一两天的滑窗——它们的 `y_future` 与查询的 `y_future` 大部分重合，NFD@k 和聚合预测的 MAE/MSE 都会系统性偏乐观。`training_pairs.build_future_utility_pairs` 同理：正样本几乎总是同序列相邻窗口，学到的度量可能退化为"找同序列最近时间点"。

**建议**：检索和配对时增加排除规则——至少排除同 `series_id` 且 `|anchor_dt差| < horizon` 的候选；更严格可整体排除同序列。`multiroute_candidate_indices` 的 `exclude_self` 目前用 `len(qx) == len(cx)` 推断同池（`learned_metric.py:232`），这个判断也很脆弱（长度恰好相等的不同池会被误判），建议显式传 `same_pool: bool`。

### 3. `context_future` 隐含"未来天气完全已知"的假设

`_context_summary` 把 `precpt`、`avg_temperature` 等实际观测天气放进 `context_future`（`data.py:70-73`）。docstring 只声明排除了 `future_stockout`，但用实际天气而非预报天气，对真实部署是偏乐观的（折扣/节假日/活动属于计划信息，已知是合理的）。

**建议**：不一定要改代码，但应在文档/论文里显式声明这是 "perfect weather foresight" 设定，或做一组去掉未来天气的消融。

---

## 中优先级（正确性边角 + 性能）

### 4. `castcf_lite_*` 重排序中三类得分量纲不可比

shape/context/meta 三个余弦得分虽然都在 [-1,1]，但分布差异很大（标准化后的 context 余弦分布远比 shape 分散），固定权重 0.5/0.4/0.1 的线性组合实际权重由分布方差决定，调参意义打折。学习度量版本（`LearnedMetricScorer` 内部做了特征标准化）已经解决了这个问题，lite 版本可以接受，但对比实验时要意识到 lite 的权重不是真实贡献占比。

### 5. 全量 O(N²) 相似度矩阵的内存风险

`castcf_lite_neighbors` 和 `castcf_multiroute_search` 都先算完整的 `ctx_sims`/`meta_sims` N×N（或 Q×C）矩阵（`retrieval.py:80-81,171-173`），即使 rerank 只用到候选列。10 万 case 时单个矩阵 ~80GB，直接不可行。当前 lite 实验规模下没问题，但 scale up 前需要改成只对候选列计算（`learned_metric.py` 的逐 query 候选特征方式就是正确范式）。

### 6. 邻居不足时 `mode="edge"` 复制填充会扭曲聚合权重

`castcf_multiroute_search`（`retrieval.py:199-202`）和 `learned_metric_search`（`learned_metric.py:290-293`）在候选不足 k 时用边缘复制填充。下游 `aggregate_neighbor_futures` 做 softmax 加权时，被复制的邻居会拿到双倍权重；NFD@k 也会被重复项稀释。建议要么返回变长邻居列表，要么在聚合端对重复索引去重/合并权重。

### 7. `aggregate_neighbor_futures` 的 temperature 对学习度量得分不适用

`temperature=0.1` 是按余弦得分（[-1,1]）标定的；`LearnedMetricScorer.predict` 输出是无界线性得分，量纲完全不同，同一 temperature 下 softmax 可能接近 one-hot 或接近均匀。建议对学习度量得分先做归一化（如除以得分 std），或按检索器类型分别调 temperature。

### 8. `meta` 在内层循环中重复计算

`data.py:142` 的 `meta = group.iloc[0][ENTITY_COLUMNS].to_numpy(...)` 对每个滑窗重复执行，但它是序列级常量，应提到 `for future_start` 循环外。`iloc[0]` 还隐含"实体列在序列内恒定"的假设，目前对 store/product 由构造保证，其余列建议加一次断言。

---

## 低优先级（代码质量）

- **`castcf_lite_neighbors` 与 `castcf_lite_search` 约 40 行重排序逻辑完全重复**（`retrieval.py:83-95` vs `128-139`）。前者可以直接委托后者（query=corpus + 对角屏蔽），或抽出共享的 `_rerank` 辅助函数。`castcf_multiroute_search` 与 `multiroute_candidate_indices` 的三路合并逻辑也是第三处重复。
- **`metrics.nfd_at_k` / `query_nfd_at_k` 可向量化**：`np.mean(np.abs(y_corpus[nbr] - y_query[:, None, :]), axis=(1, 2))` 一行替代逐行循环；两个函数本身也几乎重复，`nfd_at_k` 可实现为 `query_nfd_at_k(y, y, nbr)`。
- **`_topk_desc` 的 `kth=np.arange(k_eff)`**（`retrieval.py:30`）：传整段 arange 等于让 argpartition 对前 k 个位置全部排好序，随后又做了一次显式 argsort，二者重复。`kth=k_eff-1` 即可。
- **`multiroute_candidate_indices` 复用 `shape_knn_search` 检索 context/meta 路**（`learned_metric.py:217-219`）：功能正确，但函数名误导（"shape"），建议改名为通用的 `cosine_knn_search`。
- **`__init__.py` 为空**：建议导出主要公共 API（`build_daily_cases`、`shape_knn_search`、`LearnedMetricScorer` 等），方便 `from castcf import ...`。
- **`learned_metric.py` 第 24-29 行 `_pair_cosine` 与 `retrieval._normalize_rows` 思路重复**，可考虑合并到一处。

---

## 做得好的地方

- `features.standardize_by_reference` 只用 reference 统计量，正确避免了 query→memory 方向的信息泄漏；零方差列回退到 1.0。
- `LearnedMetricScorer.fit_pairwise` 的 pairwise logistic 梯度推导正确（`σ(-w·Δ)` 项），有 logits clip、L2、loss history、特征标准化持久化，`save/load` 用 `allow_pickle=False`，安全。
- `build_daily_cases` 显式把 `future_stockout` 排除出 `context_future` 并在 docstring 说明，泄漏意识到位。
- 全包统一的 2D 校验、行数校验和参数范围校验，报错信息带具体 shape，可调试性好。
- `cosine_similarity_matrix` 用 `np.divide(..., where=norms > 0)` 处理零向量，`aggregate_neighbor_futures` softmax 减 max，数值细节扎实。

## 建议的行动顺序

1. 修 #2（同序列时间重叠泄漏）——它直接影响所有已有实验数字的可信度。
2. 修 #1（meta ID 改为相等指示特征）——重新跑 meta 路消融。
3. 文档化 #3（天气前视假设），补一组消融。
4. 其余按需重构（#5 在 scale up 前必须做）。
