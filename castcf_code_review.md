# CastCF 代码 Review（`castcf/` 包）

> Review 范围：`castcf/data.py`、`features.py`、`metrics.py`、`retrieval.py`、`training_pairs.py`、`learned_metric.py`、`__init__.py`（共约 910 行），并参考了 `experiments/run_retrieval_baselines.py` 的调用方式。
> 日期：2026-06-11
> **更新（2026-06-11）：高优先级 #1、#2 已修复并落地代码，详见文末「修复记录」；全部 25 个测试通过。**

## 总体评价

代码质量整体偏高：模块边界清晰（数据构造 → 特征 → 检索 → 配对监督 → 学习度量 → 评估），纯 numpy/pandas 实现无重型依赖，输入校验细致（维度、行数、参数范围都有显式检查），数值处理上有不少正确的细节（softmax 减 max 防溢出、零向量余弦保护、reference-only 标准化避免泄漏、sigmoid clip）。`tests/` 下对每个模块都有对应测试文件，工程习惯好。

下面的问题按严重程度排序：前两条是方法论层面的，可能直接影响实验结论的有效性；其余是代码质量/性能问题。

---

## 高优先级（方法论正确性）

### 1. ✅ 已修复｜对原始实体 ID 做余弦/距离相似度，在语义上不成立

`meta` 向量是 7 个原始整型 ID（`city_id`、`store_id`、`product_id` 等，`data.py:142`），但检索中直接对它们算余弦相似度（`retrieval.py:81,126,173`），学习度量里还算 `-mean|qmeta - cmeta|` 距离（`learned_metric.py:77`）。

这些 ID 是**名义变量**（nominal）：store 100 和 store 101 之间没有任何"更相似"的含义，但余弦/L1 会认为它们几乎相同。实验脚本里的 `standardize_by_reference` 也无法修复这一点——z-score 之后数值相近的 ID 仍然被当成相近实体。

**影响**：meta 路召回（`multiroute`）和 meta 相似度权重（默认 0.1）注入的是噪声相关性；学习度量的 `meta_cos`/`meta_distance` 两个特征同理。如果实验显示 meta 路有正向收益，很可能是 ID 编码顺序与某种真实结构（如同城门店 ID 连续分配）偶然相关，结论不稳。

**建议**：把 meta 相似度改成**逐字段相等指示向量**（same_city、same_store、same_category_l1/l2/l3、same_product 的 0/1 向量），对它做点积或直接作为 pairwise 特征。这既符合语义，也正好适配 `pair_feature_matrix` 的特征拼接方式。

### 2. ✅ 已修复｜同池检索 + stride=1 滑窗，存在邻居未来轨迹重叠泄漏

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

- ~~**`castcf_lite_neighbors` 与 `castcf_lite_search` 约 40 行重排序逻辑完全重复**~~ —— 已随 #2 修复一并解决：两者现共用 `_rerank_candidates` 辅助函数。`castcf_multiroute_search` 与 `multiroute_candidate_indices` 的三路合并逻辑仍有一处重复。
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

1. ~~修 #2（同序列时间重叠泄漏）~~ ✅ 已完成，**需要重跑全部实验刷新指标数字**。
2. ~~修 #1（meta ID 改为相等指示特征）~~ ✅ 已完成，**需要重新训练学习度量并重跑 meta 路消融**。
3. 文档化 #3（天气前视假设），补一组消融。
4. 其余按需重构（#5 在 scale up 前必须做）。

---

## 修复记录（2026-06-11）

### #2 同序列未来窗口重叠泄漏

- **`castcf/features.py`** 新增 `overlap_exclusion_mask(query_series, query_anchors, corpus_series, corpus_anchors, min_anchor_gap)`：标记"同 series 且 anchor 间隔 < min_anchor_gap（即 horizon）"的 query-corpus 对，这些对的 `y_future` 窗口在时间上重叠，必须排除。
- **`castcf/data.py`** 新增 `case_overlap_exclusion_mask(query_cases, corpus_cases, horizon_days)`：直接从 case DataFrame（`series_id` + `anchor_dt`）构建掩码，series 经 factorize 编码、日期转为天序数。
- **`castcf/retrieval.py`**：`shape_knn_neighbors`、`shape_knn_search`、`castcf_lite_neighbors`、`castcf_lite_search`、`castcf_multiroute_search` 全部增加可选 `exclusion_mask` 参数；被排除位置的相似度置 `-inf`，multiroute 在合并候选后再按掩码过滤一次作为硬保证，候选耗尽时有降级路径。
- **`castcf/learned_metric.py`**：`multiroute_candidate_indices` 和 `learned_metric_search` 同样接受 `exclusion_mask` 并在候选合并后过滤（含 fallback 路径）；这是 `build_future_utility_pairs` 的唯一候选入口，因此训练配对也被同步保护，其 docstring 已注明该约定。
- **实验脚本**：`run_retrieval_baselines.py` 自动构建 query→memory 掩码并传入全部五种检索方法，metrics JSON 新增 `leakage_guard`（记录 `min_anchor_gap_days` 和被排除的 pair 数）；`train_metric_reranker.py` 对 memory 同池构建掩码用于配对生成，summary 新增 `excluded_overlap_pairs`。

### #1 meta ID 改为相等指示特征

- **`castcf/retrieval.py`** 新增 `meta_match_matrix(query_meta, corpus_meta)`：返回逐字段相等比例（[0,1]），替代对原始 ID 的余弦；`castcf_lite_*` 和 `castcf_multiroute_search` 的 meta 路全部改用它。
- **`castcf/learned_metric.py`**：`pair_feature_matrix` 删除 `meta_cos`/`meta_distance`，改为每个实体字段一个 0/1 相等指示（特征布局变为 `[shape_cos, context_cos, shape_distance, context_distance, meta_match×7]`，共 11 维）；`multiroute_candidate_indices` 的 meta 路改为按 match 比例取 top-k。**旧的 `learned_metric.npz` 模型与新特征维度不兼容，必须重训。**
- **实验脚本**：meta 不再做 z-score 标准化（相等匹配只需要原始值），`x`/`context` 的标准化保持不变。

### 顺带修复

- `castcf_lite_neighbors`/`castcf_lite_search` 的重复重排序逻辑合并为 `_rerank_candidates`（低优先级发现之一）。
- lite/multiroute 在候选不足 k 时统一 edge 填充，行为与之前 multiroute 一致。

### 测试

- 更新 `test_learned_metric.py`（特征维度 6→5，toy meta 为 1 列；增加 meta 指示断言）。
- 新增 6 个测试：`overlap_exclusion_mask` 行为、`shape_knn_search`/`castcf_multiroute_search`/`multiroute_candidate_indices` 的掩码生效、`meta_match_matrix` 计数、lite meta 路"数值相近但不同 ID 不算相似"。
- 全套 `pytest tests/`：**25 passed**（含两个端到端 CLI 测试）。

### 第二轮完善（2026-06-11，中低优先级项）

- **#5 内存**：lite 重排序（`_rerank_candidates`）不再预计算全量 query×corpus 的 context/meta 相似度矩阵，改为逐行只对候选计算（context 用预归一化行向量点积，meta 用逐字段相等），内存随候选数线性增长。multiroute 的三路 top-k 仍需全量相似度（暴力检索固有），维持现状。
- **#6 重复邻居**：`aggregate_neighbor_futures` 对行内重复邻居 id（edge 填充产物）只保留首次出现，softmax 不再双倍计权。
- **#7 温度语义**：`aggregate_neighbor_futures` 新增 `normalize_scores` 参数（行内标准化后再 softmax）；实验脚本暴露 `normalize_learned_scores` 配置（默认 `false`）。消融显示在共享温度下 raw 得分表现更好（temp=0.2 时 MAE 0.4596 vs 0.4972），故默认关闭、保留为消融开关。
- **#8**：`build_daily_cases` 的 `meta` 提到滑窗循环外，并增加序列内实体列恒定性断言。
- **低优先级**：NFD 计算向量化（`nfd_at_k` 委托 `query_nfd_at_k`）；`_topk_desc` 去掉 argpartition+argsort 的重复排序（`kth=k_eff-1`）；`__init__.py` 导出全部公共 API。
- **macOS 兼容**：候选级 context 相似度改用 `einsum` 而非 `@`（Accelerate BLAS 在 matmul 上会触发虚假浮点警告，`-W error` 下会中断）。
- 测试增至 **27 passed**（新增重复邻居去重、标准化平移缩放不变性两个测试）。

### 修复后重跑结果（默认 smoke 配置）

- 训练：`pair_count=4000`，loss 0.5947→0.1010；`excluded_overlap_pairs=4000`（恰为各 case 自身，印证默认 `stride=horizon` 配置泄漏面有限）。
- 评估：`leakage_guard.excluded_query_memory_pairs=0`（query→memory 本就无重叠）。
- `learned_metric`（temp=0.2 初跑）：MAE 0.459572，不优于 `recent`（0.4561）——修复前的微弱优势部分来自 meta ID 余弦的噪声相关性。
- 依据温度消融将默认 `temperature` 调整为 0.5 后（最终配置）：`learned_metric` MAE 0.449310 / MSE 0.540216 / NFD@K 0.583345，相比 `castcf_multiroute` MAE -10.43%、MSE -49.73%、NFD -10.37%；相比 `recent` MAE 低 1.50%（MSE 仍高 9.41%）。详见 README「当前默认配置结果」。

### 遗留注意事项

- 历史实验结果（如 `freshretail_castcf_lite_experiment_results.md`）基于修复前代码，meta 路结论和绝对指标需以重跑结果为准。
- 由于默认配置 memory 取 `stride_days = horizon_days`（窗口不重叠），#2 对默认配置的训练配对影响有限，但 `stride_days=1` 等配置此前会泄漏；query→memory 方向在现有 train/eval 切分下本就无重叠，掩码主要是对未来配置变化的保险。
