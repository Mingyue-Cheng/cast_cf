我建议把这个工作具象化为一个顶会级别的主线：

**CoCast: Context-aware collaborative retrieval for time series forecasting**
中文可以叫：**情境感知的协同式时间序列预测方法**。

它的核心不是简单把协同过滤套到时间序列上，而是提出一个新的问题视角：**在时间序列预测中，真正有用的历史样本不是“过去曲线最像”的样本，而是“在当前情境下，对未来演化最有参考价值”的样本。** 这个问题非常适合做成顶会工作，因为它同时连接了 retrieval-augmented forecasting、context-aware forecasting、collaborative filtering、instance memory 和 evidence-based forecasting 几条线。

现有 retrieval-augmented time series forecasting 已经有一定基础。例如 RAFT 在预测时会从训练集中检索与当前输入模式相似的历史候选，并利用这些候选的未来值辅助预测；这说明“历史案例检索 + 未来轨迹利用”在时间序列预测中是一个成立方向。([Proceedings of Machine Learning Research](https://proceedings.mlr.press/v267/han25d.html?utm_source=chatgpt.com)) 但 RAFT 这类方法的主要问题是：它更多关注历史曲线模式相似，而没有充分回答“什么样的相似性才真正对预测有用”。另一方面，ContextFormer 等工作强调多模态情境信息对预测很重要，并尝试将 categorical、continuous、time-varying、textual context 融入预测模型。([arXiv](https://arxiv.org/abs/2410.12672?utm_source=chatgpt.com)) 还有 Context is Key 这类 benchmark 直接说明，某些预测任务必须理解文本情境才能正确预测。([Proceedings of Machine Learning Research](https://proceedings.mlr.press/v267/williams25a.html?utm_source=chatgpt.com)) 但是这些 context-aware forecasting 方法通常是把情境作为输入特征融入模型，并没有把情境用于“历史案例协同检索”。推荐系统中的 context-aware collaborative filtering 则长期关注如何基于 context similarity 利用相似情境下的历史反馈。([MDPI](https://www.mdpi.com/2078-2489/13/1/42?utm_source=chatgpt.com)) 你的工作可以正好切在三者交叉处：**用情境感知相似性，把历史时间序列案例变成可检索、可加权、可解释的协同预测证据。**

------

## 0. 当前代码进度与规划状态

更新时间：2026-06-11。

这份文档现在应作为 **CoCast / CastCF 的 code-aligned 规划文档** 使用。相较于最早的方案草案，当前代码已经不只是概念设计，也不只是固定加权的 CastCF-lite baseline；它已经实现了一版 lightweight forecast-aware metric learning。

当前代码已经完成：

1. **FreshRetailNet-50K 数据读取与 case memory 构造。** 默认数据路径为 `FreshRetailNet-50K/data/train.parquet` 和 `FreshRetailNet-50K/data/eval.parquet`，当前 smoke 配置抽样 `max_series=500`，构造 `4000` 个 memory cases 和 `500` 个 query cases。
2. **检索式预测闭环。** 已支持 `recent`、`shape_knn`、`context_knn`、`castcf_lite`、`castcf_multiroute` 和 `learned_metric`。
3. **多路候选召回。** `castcf_multiroute` 从 shape、context、meta 三路各取 `route_k=100` 个候选，合并后 rerank，避免只依赖 shape-first candidate pool。
4. **Forecast-aware pairwise metric learning。** 当前 `learned_metric` 是轻量线性 scorer，不是神经网络 encoder。它基于未来轨迹效用构造正负样本，训练 pairwise ranking loss，并保存 `weights`、`feature_mean`、`feature_scale` 和 `loss_history`。
5. **基础评估指标。** 已输出 MAE、MSE、NFD@K，以及 `future_discounted`、`future_holiday`、`future_activity`、`past_stockout`、`future_stockout` 等 subset 指标。

当前默认 smoke 结果如下：

| 方法 | MAE | MSE | NFD@K |
| --- | ---: | ---: | ---: |
| `recent` | 0.456142 | 0.493747 | - |
| `shape_knn` | 0.872817 | 3.601460 | 1.044328 |
| `context_knn` | 0.652710 | 1.694710 | 0.822029 |
| `castcf_lite` | 0.586405 | 1.362452 | 0.749033 |
| `castcf_multiroute` | 0.511730 | 1.155460 | 0.660868 |
| `learned_metric` | 0.439667 | 0.490470 | 0.564674 |

这组结果说明两件事：

1. context 和 meta 参与检索后，NFD@K 明显低于 shape-only retrieval，说明 FreshRetailNet-50K 上确实存在“历史曲线相似不等于未来有用”的信号。
2. `learned_metric` 已经把“未来预测效用监督相似性”从方案推进到了可运行代码，当前下一步不再是“引入 learned forecast-aware similarity”，而是要把它扩展到更强、更严格的论文级实验。

当前还没有完成：

1. 神经网络 multi-view encoder。
2. 与 PatchTST、TimesNet、iTransformer、TSMixer 等 strong forecasting backbone 的融合。
3. base forecast + collaborative residual correction。
4. retrieval reliability gate。
5. hard negatives 的系统构造。
6. full-scale FreshRetailNet、多 seed、多 split、多数据集评测。
7. counterfactual context test 和 same-past-different-future probe。

因此，当前阶段定位应从 **concept proposal** 更新为 **verified pilot + next-step paper prototype**：核心机制已经有初步代码证据，后续重点是规模化、强 baseline、严格消融和与 backbone 的融合。

## 1. 核心问题定义：从“相似历史曲线”到“相似情境案例”

可以把每一个历史预测样本定义为一个 forecasting case：

[
\mathcal{I}_i = {X_i^{past}, C_i^{past}, C_i^{future}, M_i, Y_i^{future}}.
]

其中，(X_i^{past}) 是历史 lookback window，(C_i^{past}) 是历史情境，(C_i^{future}) 是预测时已知或可获得的未来情境，例如节假日、天气预报、日历、活动安排、政策状态等；(M_i) 是实体元信息，例如区域、用户、门店、传感器、行业类型；(Y_i^{future}) 是真实未来轨迹。当前预测样本可以写作：

[
q = {X_q^{past}, C_q^{past}, C_q^{future}, M_q}.
]

目标不是只根据 (X_q^{past}) 预测 (Y_q^{future})，而是从历史案例库 (\mathcal{M}) 中检索 Top-K 个最有预测参考价值的案例：

[
\mathcal{N}*q = \text{TopK}*{i \in \mathcal{M}} \ s(q,i),
]

再利用这些邻居的未来轨迹 (Y_i^{future})、上下文差异、实体关系和历史表现来辅助预测当前未来。

这个问题的关键在于相似度 (s(q,i)) 的定义。传统 kNN forecasting 或部分 retrieval-augmented forecasting 会更偏向：

[
s(q,i) = s_{\text{shape}}(X_q^{past}, X_i^{past}),
]

也就是历史曲线越像，邻居越重要。但你的核心观点应该是：

[
s(q,i) = f_{\theta}(X_q^{past}, X_i^{past}, C_q, C_i, M_q, M_i, H),
]

也就是相似性应该由**历史模式、情境条件、实体属性和预测目标 horizon**共同决定。

一句话概括这个问题：

> Existing retrieval-based forecasting asks: which historical windows look similar to the current window?
> Our method asks: under the current context, which historical cases are truly useful for predicting the future?

这句话很适合作为论文的主 claim。

------

## 2. 协同过滤思想如何映射到时间序列预测

你可以把推荐系统中的协同过滤做一个很漂亮的迁移类比。

在推荐系统中，协同过滤的基本假设是：**相似用户在相似物品上有相似偏好**。在你的问题中，可以改写为：**处于相似情境下、具有相似历史演化模式的时间序列案例，往往具有相似的未来演化趋势。**

具体映射可以这样理解：

| 推荐系统             | 时间序列预测                                           |
| -------------------- | ------------------------------------------------------ |
| User                 | 时间序列实体，例如区域、门店、用户、病人、传感器、道路 |
| Item                 | 预测目标、未来模式、未来事件、horizon-specific pattern |
| Rating               | 未来数值、趋势变化、峰值、风险、概率分布               |
| Context              | 天气、节假日、事件、区域状态、文本新闻、业务阶段       |
| Neighbor             | 相似用户/相似交互                                      |
| Collaborative signal | 相似用户的历史反馈                                     |

所以这个工作可以叫 **collaborative forecasting**，但不要停留在类比层面，而要进一步提出一个可学习的机制：**Forecast-aware context similarity learning**。也就是说，相似性不是人为定义的，而是通过“这个历史案例的未来是否真的对当前预测有帮助”来学习出来。

------

## 3. 整体框架：Memory bank + Context-aware retrieval + Collaborative aggregation

我建议整个方法设计为四个核心模块。

### 模块一：Contextual forecasting case memory

首先构建一个历史案例库，每个历史窗口都被存成一个 case：

[
\mathcal{M} = {(X_i^{past}, C_i^{past}, C_i^{future}, M_i, Y_i^{future})}_{i=1}^{N}.
]

这里要特别强调两点。第一，(Y_i^{future}) 只在历史案例中已经发生，因此可以被存入 memory bank；当前 query 的真实未来当然不可见，不存在 label leakage。第二，(C_q^{future}) 只能使用预测时可以获得的未来情境，例如日历、节假日、天气预报、已知活动，而不能使用事后才知道的信息。

这个 memory bank 不是普通训练集缓存，而是一个**可检索的 forecasting experience memory**。这和你之前的 MemCast、Instance Search、CastClaw 的记忆思想可以自然衔接。

------

### 模块二：Multi-view encoder

对 query 和历史 case 分别编码。至少需要四类表征：

[
z_i^{ts} = E_{ts}(X_i^{past}),
]

[
z_i^{ctx} = E_{ctx}(C_i^{past}, C_i^{future}),
]

[
z_i^{meta} = E_{meta}(M_i),
]

[
z_i^{task} = E_{task}(H, \text{target type}).
]

其中 (E_{ts}) 可以用 PatchTST、iTransformer、TimesNet、DLinear encoder 或你自己的轻量时序 encoder；(E_{ctx}) 可以编码天气、节假日、事件、文本、类别变量和连续变量；(E_{meta}) 用于编码实体属性；(E_{task}) 表示预测 horizon 和任务类型，例如短期预测、长期预测、峰值预测、风险预测等。

这里不要把 context 简单拼到 time series 后面，而要保留多视角结构，因为你的相似度本身就是多视角组合出来的。

------

### 模块三：Context-aware collaborative similarity

这是论文的核心创新模块。建议定义多种基础相似度：

[
s_{ts}(q,i) = \cos(z_q^{ts}, z_i^{ts}),
]

[
s_{ctx}(q,i) = \cos(z_q^{ctx}, z_i^{ctx}),
]

[
s_{meta}(q,i) = \cos(z_q^{meta}, z_i^{meta}).
]

然后不是固定加权，而是让当前 query 的情境决定不同相似度的重要性：

[
\alpha_q = \text{softmax}(g_{\theta}(z_q^{ctx}, z_q^{ts}, z_q^{meta})),
]

[
s(q,i) = \alpha_q^{ts}s_{ts}(q,i) + \alpha_q^{ctx}s_{ctx}(q,i) + \alpha_q^{meta}s_{meta}(q,i) - \lambda d_{\text{stale}}(q,i).
]

这里的 (d_{\text{stale}}(q,i)) 可以表示时间陈旧性或分布漂移惩罚，避免模型检索到很久以前但已经不可靠的案例。

这个设计很关键，因为它可以表达一种动态逻辑：普通工作日预测时，历史曲线形状可能最重要；极端高温时，天气情境更重要；节假日时，日历和事件情境更重要；跨区域预测时，实体属性更重要。也就是说，相似性本身是 context-conditioned 的。

这一点就是顶会创新点之一：**the similarity metric is not static, but conditioned on the forecasting context.**

------

### 模块四：Collaborative evidence aggregation

检索到 Top-K 邻居之后，不要简单做平均，否则容易显得像 kNN 改进。建议设计一个 collaborative evidence aggregation module。

对于每个邻居 (i)，我们有：

[
e_i = {Y_i^{future}, X_i^{past}, C_i, M_i, s(q,i)}.
]

可以先构建一个邻居未来轨迹的非参数预测：

[
\hat{Y}*{cf} = \sum*{i \in \mathcal{N}_q} w_i \cdot \text{Align}(Y_i^{future}),
]

其中：

[
w_i = \frac{\exp(s(q,i)/\tau)}{\sum_{j \in \mathcal{N}_q} \exp(s(q,j)/\tau)}.
]

但更强的做法是将它作为 residual correction：

[
\hat{Y} = \hat{Y}*{base} + \Delta \hat{Y}*{cf}.
]

也就是说，基础模型负责学习常规模式，协同检索模块负责补充 rare pattern、event-driven pattern 和 context-sensitive pattern。

更进一步，可以设计一个 gate：

[
\hat{Y} = \beta_q \hat{Y}*{base} + (1-\beta_q)\hat{Y}*{cf}.
]

其中 (\beta_q) 由 retrieval reliability 决定，例如 Top-K 相似度是否集中、邻居是否一致、邻居未来轨迹是否分歧很大、当前样本是否是异常情境。如果检索到的邻居很可靠，就更多相信 collaborative evidence；如果邻居分歧很大，就更多相信基础模型。

这个 gate 很重要，因为它能解决一个实际问题：不是每次检索都有好邻居。顶会审稿人一定会问“bad retrieval 怎么办”，这个模块就是回答。

------

## 4. 训练目标：让相似性真正对预测有用

这个工作最应该避免的是“我手工定义一个 context similarity，然后加权平均”。这样容易变成工程方法。要做成顶会，必须让相似度通过预测目标学习出来。

我建议引入 **forecast-aware metric learning**。核心思想是：如果历史案例 (i^+) 的未来轨迹比 (i^-) 更接近当前样本真实未来，那么模型应该学习到：

[
s(q,i^+) > s(q,i^-).
]

可以用 pairwise ranking loss：

[
\mathcal{L}_{rank} = - \log \sigma(s(q,i^+) - s(q,i^-)).
]

这里 (i^+) 和 (i^-) 的构造非常关键。正样本不是“历史曲线最像”的样本，而是“未来轨迹更有参考价值”的样本。可以定义：

[
d_{future}(q,i) = \text{MSE}(Y_q^{future}, Y_i^{future}) + \gamma d_{\text{trend}}(Y_q^{future}, Y_i^{future}).
]

如果 (d_{future}(q,i^+) < d_{future}(q,i^-))，则 (i^+) 是更好的预测邻居。

同时需要构造 hard negatives：

第一类是 **shape-similar but future-different negatives**，即历史曲线很像，但由于情境不同，未来走向完全不同。这类负样本最能证明 context-aware similarity 的必要性。

第二类是 **context-similar but pattern-different negatives**，即情境类似，但历史状态差异很大，未来也不应直接参考。

第三类是 **stale negatives**，即过去某些很像的老样本，但由于长期分布变化已经不可靠。

最终总损失可以写成：

[
\mathcal{L} = \mathcal{L}*{pred} + \lambda_1 \mathcal{L}*{rank} + \lambda_2 \mathcal{L}*{calib} + \lambda_3 \mathcal{L}*{reg}.
]

其中 (\mathcal{L}*{pred}) 是预测误差，例如 MSE、MAE 或 pinball loss；(\mathcal{L}*{rank}) 训练相似度；(\mathcal{L}*{calib}) 可以做概率预测校准；(\mathcal{L}*{reg}) 防止相似度权重塌缩。

这部分是论文的技术核心。它把“相似性”从静态距离变成了一个**由未来预测效用监督的可学习函数**。

------

## 5. 一个具象例子：电力负荷预测

以电力负荷预测为例，当前 query 是：

“某城市过去 48 小时负荷曲线出现持续上升，明天是工作日，但天气预报显示极端高温，且存在局部大型活动。”

如果只做 shape-based retrieval，模型可能检索到一些普通工作日的高负荷历史窗口。这些窗口过去曲线确实很像，但未来可能只是正常回落。

而 context-aware collaborative retrieval 会优先检索：

“历史上同样处于高温工作日、同样有活动、同样在夏季、同样负荷处于上升阶段的案例。”

这些邻居的未来轨迹可能显示：下午峰值进一步抬升，晚高峰持续时间更长，夜间回落更慢。于是模型利用这些邻居未来轨迹对基础预测进行修正。

这个例子可以写成论文中的 motivating example。它可以非常清楚地说明：**同样的历史曲线，在不同情境下对应不同未来；同样的未来趋势，往往需要从相似情境案例中协同推理。**

------

## 6. 论文可以主打的三个创新点

我建议把贡献收束成三个，不要太散。

**贡献一：提出 Context-aware collaborative forecasting 范式。**
你不是简单做 retrieval-augmented forecasting，而是明确提出“时间序列预测中的预测邻居应由情境调制”。这可以作为一个新的 problem formulation：forecasting by retrieving contextually predictive neighbors。

**贡献二：提出 Forecast-aware context similarity learning。**
你设计一个多视角、动态加权、由未来预测效用监督的相似度学习机制。相比 shape-only retrieval，你的方法能区分“历史像但未来不一样”和“历史不完全像但情境下未来相似”的案例。

**贡献三：提出 Collaborative evidence aggregation。**
你把检索到的历史案例未来轨迹作为 non-parametric evidence，与基础预测模型融合，用于残差修正、概率预测和可解释预测。这个模块能让方法不只是提升精度，还能输出“为什么这样预测”的历史证据。

如果需要写得更顶会一点，可以凝练成：

> We formulate time series forecasting as context-conditioned collaborative reasoning over historical forecasting cases. Instead of retrieving neighbors solely based on past temporal patterns, our method learns forecast-aware similarity functions that identify historical cases whose future trajectories are predictive under the current context.

------

## 7. 方法命名与标题建议

我最推荐的名字是：

**CoCast: Context-aware Collaborative Forecasting with Forecast-aware Similarity Learning**

这个名字比较稳，Co 表示 context-aware + collaborative，Cast 和 forecasting 也很贴合你已有 Cast 系列。

标题可以有几个版本：

**版本一，顶会风格：**
**When Similar Past Is Not Enough: Context-aware Collaborative Retrieval for Time Series Forecasting**

这个标题突出问题意识，很适合 ICML / NeurIPS / ICLR / KDD。

**版本二，方法名风格：**
**CoCast: Context-aware Collaborative Forecasting with Forecast-aware Similarity Learning**

这个更像完整方法论文。

**版本三，与你长期主线更一致：**
**Context-aware Collaborative Reasoning for Time Series Forecasting**

这个更强调 reasoning，但如果方法里没有 LLM 或显式推理模块，可能略大。

我建议用版本一作为论文标题，CoCast 作为方法名。

------

## 8. 实验设计：一定要证明 context-aware retrieval 的必要性

这个工作实验不能只做普通 MSE/MAE 提升。顶会审稿人会关心：你的 context-aware similarity 到底学到了什么？有没有真正解决 shape-only retrieval 的失败问题？

因此实验要分成四层。

第一层是常规预测性能。比较 DLinear、PatchTST、TimesNet、iTransformer、TFT、ContextFormer、RAFT、kNN retrieval、MQ-RetCNN 等模型。你的方法可以作为 plug-in 接到不同 base forecaster 上，例如 CoCast-DLinear、CoCast-PatchTST、CoCast-iTransformer，证明它是通用增强模块。

第二层是 context-sensitive subset。专门挑出节假日、极端天气、异常事件、需求突变、分布变化时段，证明你的方法在这些场景下提升更明显。这个实验非常关键，因为你的 claim 是 context-aware retrieval，而不是普通预测增强。

第三层是 retrieval quality。定义 retrieved neighbors 的 future distance：

[
\text{NFD@K} = \frac{1}{K}\sum_{i \in \mathcal{N}*q} d*{future}(q,i).
]

如果你的方法检索到的邻居未来更接近当前真实未来，就说明它真的找到了更有预测价值的案例，而不是只提升了预测误差。

第四层是 counterfactual context test。固定同一个历史曲线 (X_q^{past})，改变未来情境 (C_q^{future})，例如从普通工作日改成节假日、从常温改成高温，观察检索到的邻居是否发生合理变化，预测是否随情境变化。这是非常强的可解释实验，能证明模型真的使用了 context，而不是把 context 当噪声特征。

------

## 9. 消融实验要这样设计

消融实验可以直接服务你的 claim。

第一，去掉 context-aware similarity，只用 shape similarity。如果性能明显下降，说明情境相似性有用。

第二，去掉 dynamic gating，改成固定权重。如果性能下降，说明不同情境下相似性权重需要动态调整。

第三，去掉 ranking loss，只用预测损失。如果 retrieval quality 下降，说明 forecast-aware metric learning 有用。

第四，去掉 collaborative evidence aggregation，只把检索向量拼接进模型。如果效果下降，说明显式利用邻居未来轨迹是必要的。

第五，去掉 recency / shift penalty。如果在非平稳数据上效果下降，说明历史案例可靠性建模有用。

第六，比较不同 Top-K。这个能说明协同证据数量对模型的影响。如果 K 太小，证据不足；如果 K 太大，噪声邻居增加。

第七，比较 direct aggregation 和 residual correction。如果 residual correction 更稳，可以说明基础模型负责常规模式，协同模块负责上下文特异性修正。

------

## 10. 顶会工作需要一个更强的 benchmark 设计

如果只是提出方法，在常规数据集上提升一点，可能会被认为是 RAFT + context features。为了冲顶会，我建议你额外设计一个 **Contextual Neighbor Challenge**。

这个 benchmark 不一定要完全新建数据集，可以从已有电力、交通、零售、环境、金融数据中构造 probing tasks。核心是构造两类困难样本：

第一类叫 **same past, different future**。历史窗口形状很像，但由于情境不同，未来完全不同。例如同样是负荷上升，普通工作日未来回落，高温日未来继续上升。

第二类叫 **different past, similar future**。历史窗口不完全相同，但由于情境相似，未来演化趋势相似。例如不同区域的负荷基线不同，但都处于高温节假日前夕，未来都出现相似峰值抬升。

这两个集合可以直接证明 shape-only retrieval 的局限。你的方法如果在这两个 subset 上明显优于 RAFT / kNN / 普通 context model，论文说服力会非常强。

这个 benchmark 可以命名为：

**Contextual Neighbor Forecasting Benchmark, CNF-Bench**

或者：

**CoCast-Bench: A Benchmark for Context-aware Collaborative Forecasting**

如果时间有限，也可以不单独作为 benchmark paper，而是在方法论文中作为一个专门 evaluation protocol。

------

## 11. 这篇论文的故事线可以这样写

论文的 introduction 可以按照这个逻辑展开：

第一段：时间序列预测长期依赖历史模式学习，但真实世界预测往往受到情境因素影响，例如天气、节假日、事件、政策和区域状态。仅依赖历史曲线可能无法处理非平稳和事件驱动的未来变化。

第二段：retrieval-augmented forecasting 通过检索历史相似模式，为预测提供非参数记忆，缓解模型参数无法记住稀有模式的问题。然而，现有 retrieval 方法大多基于历史形状相似性，可能检索到“过去相似但未来无关”的案例。

第三段：context-aware forecasting 证明情境信息对预测有重要价值，但多数方法只是把 context 融入模型输入，而没有利用 context 来决定哪些历史案例值得参考。

第四段：受 context-aware collaborative filtering 启发，我们提出 context-aware collaborative forecasting。核心思想是：预测邻居应该由当前情境动态决定；历史案例的未来轨迹可以作为协同证据，用于修正和解释当前预测。

第五段：介绍 CoCast，包含 memory bank、context-aware similarity learning、forecast-aware ranking objective、collaborative evidence aggregation。最后列贡献。

这个故事线非常顺。

------

## 12. 可能的模型结构图

你后续画图时可以画成五层：

**Input query layer**：当前历史序列、未来可知情境、实体元信息。
**Case memory layer**：历史 forecasting cases，每个 case 包含 past window、context、future trajectory。
**Context-aware retrieval layer**：多视角编码、动态相似度、Top-K 检索与 reranking。
**Collaborative evidence layer**：邻居未来轨迹、邻居 residual、邻居可靠性、加权聚合。
**Forecasting layer**：base forecast + collaborative correction + uncertainty output。

图中最重要的是突出两条路径：一条是 parametric path，即 base forecaster；另一条是 non-parametric collaborative path，即 memory retrieval。最后通过 gate 融合。

------

## 13. 这项工作的风险与规避

最大的风险是被审稿人认为“只是 kNN / RAFT 加 context”。所以你必须强调三点区别。

第一，RAFT 主要检索历史模式相似样本，而你的方法检索的是**情境条件下对未来有用的样本**。

第二，你不是手工定义 context similarity，而是通过 future utility 监督学习相似度，也就是 forecast-aware metric learning。

第三，你不是简单平均邻居未来，而是设计了 collaborative evidence aggregation，并进一步提供 retrieval quality、counterfactual context 和 context-sensitive subset 的系统实验。

还有一个重要风险是 label leakage。你要在方法中明确：训练阶段可以用历史 case 的真实未来来构造 ranking supervision；推理阶段当前 query 的真实未来不可见，只能使用历史案例中已经发生过的未来和当前可提前获得的未来情境。

------

## 14. 当前最小可行版本与下一步路线

根据当前代码进度，最小可行版本不应再从零开始定义。现在应分成三个阶段推进。

### 14.1 已完成的 v0：CastCF learned-metric pilot

当前 v0 已经完成以下闭环：

1. Memory bank 存储 FreshRetailNet-50K 的历史 forecasting cases。
2. Query 只检索 memory cases，不在 query 集合内互相检索。
3. shape、context、meta 三路候选召回。
4. 基于 future utility 的 pairwise ranking supervision。
5. 线性 learned metric reranker。
6. Top-K 邻居未来轨迹加权聚合预测。
7. MAE、MSE、NFD@K 和 subset metrics。

这个版本已经足以作为 pilot evidence：它证明数据、case memory、context retrieval、多路召回和 forecast-aware metric learning 都能跑通，并且 `learned_metric` 已经优于固定加权的 `castcf_multiroute`。

但 v0 不能直接作为最终论文方法，因为它仍然是 numpy 级别轻量原型，没有 neural encoder，没有 strong forecasting backbone，也没有完整 benchmark。

### 14.2 下一步 v1：论文最小方法版本

建议 v1 收束为一个可投稿的最小方法，而不是一次性实现所有设想：

1. 选择一个强但可控的 base forecaster，例如 PatchTST、iTransformer 或 TSMixer。
2. 保留当前 case memory 和多路候选召回框架。
3. 将当前 linear learned metric 升级为 multi-view scorer，但先不做过重的生成式或大模型模块。
4. 把邻居未来轨迹作为 residual correction，而不是只做直接邻居加权：

[
\hat{Y} = \hat{Y}_{base} + \Delta \hat{Y}_{cf}.
]

5. 用 pairwise ranking loss 保持 retrieval quality，同时加入 prediction loss：

[
\mathcal{L} = \mathcal{L}_{pred} + \lambda \mathcal{L}_{rank}.
]

6. 先实现一个轻量 retrieval reliability gate，只回答一个问题：当前 query 是否应该相信 retrieved evidence。

这个 v1 的主张应是：

> CoCast is a retrieval memory module that improves a base forecaster by learning forecast-useful historical evidence under the current context.

也就是说，v1 不再只是 kNN forecasting，而是 **base forecaster + forecast-aware collaborative retrieval memory**。

### 14.3 v1 的实验优先级

为了让 v1 具备论文说服力，实验应按以下优先级补齐：

1. **扩大 FreshRetailNet 规模。** 从 `max_series=500` 扩到更大规模，至少报告多个规模下的趋势。
2. **加入 validation split。** 从官方 train 内部切 validation query，用于调 `route_k`、`k`、学习率和 loss 权重，避免反复使用官方 eval。
3. **多 seed。** 当前 smoke 配置只能说明方向有信号，不能说明稳定性。
4. **强 baseline。** 至少比较 DLinear、PatchTST、iTransformer 或 TSMixer，再比较 RAFT / shape retrieval / context model。
5. **核心消融。** 去掉 context、去掉 meta、去掉 learned metric、去掉 ranking loss、不同 `k` 和 `route_k`。
6. **context-sensitive subset。** 单独报告 discount、activity、holiday、stockout、context shift 等样本。
7. **same-past-different-future probe。** 直接证明 shape-only retrieval 会失败，而 context-aware learned metric 能换邻居。

### 14.4 v2：完整顶会版本

v2 再考虑更完整的模块：

1. 动态 query-conditioned similarity weights。
2. hard negatives：shape-similar future-different、context-similar pattern-different、stale negatives。
3. calibration loss 和 uncertainty output。
4. 更严格的 retrieval reliability gate。
5. 多数据集：零售、电力、交通、环境等 context-rich forecasting 场景。
6. counterfactual context test。
7. CoCast-Bench 或 Contextual Neighbor Forecasting Benchmark。

换句话说，当前最务实的路线是：**先把已跑通的 learned metric pilot 升级成 backbone-compatible CoCast-v1，再用严格评测证明它不是简单的 kNN + context features。**

------

## 15. 我建议最终 paper 的一句话定位

这句话可以作为你整个工作的核心：

**We propose to forecast the future by collaboratively reasoning over historical cases that are not only temporally similar, but also contextually predictive.**

中文就是：

**我们不是寻找“过去最像”的历史窗口，而是寻找“在当前情境下最能预示未来”的历史案例。**

这句话非常有辨识度，也能和你长期做的情境认知时间序列分析、Instance Search、Memory-enhanced forecasting、CastClaw / LoadCast 形成统一主线。总体来看，这个方向完全可以发展成顶会工作，关键是把它从“协同过滤思想迁移”提升为一个新的预测范式：**Context-conditioned collaborative evidence for forecasting**。
