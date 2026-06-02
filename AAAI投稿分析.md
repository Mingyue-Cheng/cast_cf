AAAI投稿分析 我认为：**这个 idea 是足够支撑投稿 AAAI 的，但不是以当前 v0 形态直接投稿，而是需要把它升级成一个“问题定义清晰 + 方法机制完整 + 实验验证有辨识度”的算法论文。** 如果只是做“时间序列相似度 + context 相似度 + Top-K 邻居加权平均”，它更像一个合理 baseline 或 workshop 级别想法；但如果你把它上升为 **Context-aware collaborative forecasting / Forecast-aware similarity learning**，并证明它能解决 “same past, different future” 这一类关键预测难题，我觉得完全可以冲 AAAI。

AAAI 的主技术方向覆盖很广，官方 call 明确包括 machine learning、data mining、reasoning、search、planning、human-in-the-loop AI 等方向，且也鼓励 AI 与应用领域之间的交叉工作，例如 healthcare、sustainability、transportation、commerce 等场景。你的工作如果落在“时间序列预测 + 检索增强 + 情境感知 + 协同推理/相似性学习”，是符合 AAAI 范围的。([AAAI](https://aaai.org/conference/aaai/aaai-26/main-technical-track-call/?utm_source=chatgpt.com)) 但 AAAI 对完整性和可复现性要求也比较明确，官方 reproducibility checklist 会关注实验设置、超参数范围、数据预处理代码等细节，因此这篇工作不能只靠 idea，必须有稳定、系统、可复现的实验闭环。([AAAI](https://aaai.org/conference/aaai/aaai-26/reproducibility-checklist/?utm_source=chatgpt.com))

我的判断可以更明确一点：**如果做成 CoCast-v0，只够支撑初步实验；如果做成 CoCast-v1/v2，就有 AAAI 投稿价值。** CoCast-v0 是固定权重的 hybrid similarity retrieval，也就是：

[
s(q,i)=\alpha s_{\text{ts}}(q,i)+(1-\alpha)s_{\text{ctx}}(q,i).
]

这个版本可以验证想法是否有效，但技术贡献偏弱，容易被审稿人认为是 “RAFT + context features”。因为 RAFT 已经提出通过检索训练集中与当前输入模式相似的历史候选，并利用这些候选的未来值增强预测；也就是说，“retrieval-augmented forecasting” 这个基础方向已经存在。([Proceedings of Machine Learning Research](https://proceedings.mlr.press/v267/han25d.html?utm_source=chatgpt.com)) 同时，ContextFormer 这类工作也已经强调并验证了将 categorical、continuous、time-varying、textual 等多模态情境信息融入预测模型的价值。([arXiv](https://arxiv.org/html/2410.12672v2?utm_source=chatgpt.com)) 因此，你的论文不能只是“检索 + 情境”，而要回答一个更强的问题：**什么样的历史案例在当前情境下才真正对未来预测有用？**

我建议把 AAAI 版的核心 claim 定成：

> Existing retrieval-augmented forecasting methods retrieve historical cases mainly based on temporal pattern similarity. However, in real-world forecasting, the predictive usefulness of a historical case is context-dependent. We propose a context-aware collaborative forecasting framework that learns forecast-aware similarities to retrieve historical cases whose future trajectories are predictive under the current context.

中文就是：

**现有检索增强预测主要寻找“过去形状相似”的历史窗口，但真实预测中，一个历史案例是否有参考价值取决于当前情境。我们提出情境感知的协同式时间序列预测框架，学习面向预测效用的相似性，从而检索在当前情境下真正有助于未来判断的历史案例。**

这个 claim 是足够支撑 AAAI 的，因为它不是单纯做性能增强，而是在定义一个更合理的预测邻居选择原则：**不是 past-similar，而是 contextually predictive。**

------

我建议 AAAI 版至少包含三个技术点。

第一个技术点是 **Context-conditioned similarity learning**。不要使用固定 (\alpha)，而是让当前 query 的情境动态决定历史形状相似、情境相似、实体相似的权重：

[
\boldsymbol{\alpha}*q = \text{softmax}(g*{\theta}(z_q^{ts}, z_q^{ctx}, z_q^{meta})),
]

[
s(q,i)=\alpha_q^{ts}s_{ts}(q,i)+\alpha_q^{ctx}s_{ctx}(q,i)+\alpha_q^{meta}s_{meta}(q,i).
]

这个设计比普通相似度更有论文贡献，因为它表达了一个关键机制：普通工作日可能更依赖历史曲线，高温天气更依赖天气情境，节假日更依赖日历和事件，跨区域预测更依赖实体属性。相似性不是固定规则，而是由预测情境调制。

第二个技术点是 **Forecast-aware ranking loss**。这一步很重要，它决定这个工作能不能从“工程拼接”变成“学习式方法”。你可以让模型学习：如果历史案例 (i^+) 的未来轨迹比 (i^-) 更接近 query 的真实未来，那么应满足：

[
s(q,i^+) > s(q,i^-).
]

对应 loss 是：

[
\mathcal{L}_{rank}=-\log\sigma(s(q,i^+)-s(q,i^-)).
]

这样，相似度的监督信号不是“过去曲线是否像”，而是“这个邻居的未来是否真的有预测参考价值”。这就是 forecast-aware similarity learning。这个机制很关键，因为它能直接和 RAFT 区分开：RAFT 更偏向基于历史模式检索，你的方法学习的是**预测效用驱动的邻居排序**。

第三个技术点是 **Collaborative evidence aggregation**。检索到 Top-K 历史案例后，不应只是简单平均未来轨迹，而应作为协同证据对 base forecaster 进行 residual correction：

[
\hat{Y} = \hat{Y}*{base} + \Delta \hat{Y}*{cf}.
]

其中：

[
\Delta \hat{Y}*{cf}=h*{\theta}({Y_i^{future}, s(q,i), C_q-C_i, X_q-X_i}_{i \in \mathcal{N}_q}).
]

这会让方法从 kNN-style forecasting 变成一个“参数模型 + 非参数历史案例证据”的混合预测框架。AAAI 审稿人会更容易接受这种有明确机制、有学习目标、有解释价值的方法。

------

实验上，我觉得 AAAI 版必须有一个强验证点，否则很容易被认为只是 incremental。这个强验证点就是：

**same past, different future / different past, similar future。**

你可以构造两个 context-sensitive evaluation subsets。第一类是 **same past, different future**：历史窗口形状相似，但由于情境不同，未来轨迹不同。例如电力负荷中，同样是负荷持续上升，普通工作日未来可能回落，而高温工作日未来可能继续冲高。第二类是 **different past, similar future**：历史窗口不完全相似，但由于情境相似，未来演化相似。例如不同区域负荷基线不同，但都处于夏季高温节假日前夕，未来都出现峰值抬升。

这两个 subset 是你论文的“杀手实验”。它们直接证明：**仅仅依靠历史曲线形状是不够的，必须做 context-aware collaborative retrieval。** 这也能呼应近期 context-aware forecasting benchmark 的趋势，例如 Context is Key 明确构造了需要理解文本情境才能解决的预测任务。([OpenReview](https://openreview.net/forum?id=ih2WuBT1Fn&utm_source=chatgpt.com))

如果我是审稿人，我会希望看到四类结果。第一，常规 forecasting performance：MSE、MAE、CRPS 或 pinball loss。第二，context-sensitive subset performance：在节假日、极端天气、事件驱动、异常变化窗口上是否提升更明显。第三，retrieval quality：检索到的邻居未来轨迹是否真的更接近当前真实未来，例如报告 NFD@K。第四，case study：展示 shape-only retrieval 找错邻居，而 context-aware retrieval 找到更合理历史案例。

------

所以我对这个 idea 的 AAAI 可投性判断如下。

**只做 CoCast-v0：AAAI 风险较高。** 固定相似度加权、Top-K 未来轨迹平均、接一个 base forecaster，这个版本可以作为实验起点，但作为 AAAI 主方法会显得技术深度不够。审稿人很可能会质疑：这和 RAFT、kNN forecasting、context feature fusion 的区别在哪里？

**做成 CoCast-v1：可以投稿 AAAI。** 也就是加入动态 context-conditioned similarity、forecast-aware ranking loss、collaborative residual correction，并在多个数据集上系统验证。这时论文已经有明确技术贡献，不只是简单组合。

**做成 CoCast-v2：比较有竞争力。** 在 v1 基础上，再加入 context-sensitive benchmark/probing、same-past-different-future 分析、counterfactual context test、retrieval quality evaluation。这个版本不仅有方法，还有新的 evaluation insight，AAAI 竞争力会明显增强。

我建议你把这篇 AAAI 论文的定位写成：

**CoCast: Context-aware collaborative forecasting with forecast-aware similarity learning**

或者更有问题意识的标题：

**When Similar Past Is Not Enough: Context-aware Collaborative Retrieval for Time Series Forecasting**

第二个标题更适合 AAAI，因为它先提出一个清晰的科学问题：**过去相似不等于未来相似。** 这比单纯方法名更容易让审稿人记住。

------

我给一个比较直接的结论：

**这个 idea 足够支撑 AAAI，但前提是你不能把它写成“协同过滤启发的相似样本加权预测”，而要写成“面向情境预测效用的历史案例检索与协同证据建模”。**

它的 AAAI 论文核心应当是：

**在时间序列预测中，历史案例的参考价值不是由过去形状单独决定的，而是由当前情境、实体属性和预测目标共同决定的；因此，需要学习一种 context-conditioned、forecast-aware 的相似性函数，从历史案例库中检索真正有助于未来预测的协同证据。**

如果第一版实验显示：CoCast-v0 已经能稳定超过 shape-only retrieval，那么我认为就值得正式立项冲 AAAI。接下来最关键的是把 v0 快速跑出来，然后升级 v1 的 ranking loss 和 dynamic similarity。