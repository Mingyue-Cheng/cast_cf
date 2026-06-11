# 经典协同过滤工作调研：ItemKNN 与 BPR-MF

调研日期：2026-06-02

这份笔记聚焦两个经典协同过滤支柱：

1. **ItemKNN / Item-based CF**：从用户-物品交互矩阵中学习物品之间的相似性，再用目标用户已交互物品的邻居来推荐或预测。
2. **BPR-MF**：把隐式反馈推荐从评分回归改成 pairwise personalized ranking，用矩阵分解打分器学习“用户更偏好已观察物品而不是未观察物品”。

对当前 CastCF/CoCast 来说，ItemKNN 给的是“case-to-case 相似检索 + 邻居聚合”的原型，BPR-MF 给的是“用 pairwise 排序目标学习相似度/效用函数”的原型。

## 1. 必读论文与定位

| 论文 | 年份 | 位置 | 为什么经典 | 和 ItemKNN / BPR-MF 的关系 |
|---|---:|---|---|---|
| [Item-based Collaborative Filtering Recommendation Algorithms](https://ra.ethz.ch/cdstore/www10/papers/519/sdm2.html) | 2001 | WWW | 系统提出并评估 item-based CF，目标是解决 user-based kNN 在大规模网站上的计算和稀疏性问题 | ItemKNN 的核心原型：item-item similarity + weighted-sum / regression prediction |
| [Amazon.com Recommendations: Item-to-Item Collaborative Filtering](https://doi.org/10.1109/MIC.2003.1167344) | 2003 | IEEE Internet Computing | 工业界大规模 item-to-item CF 的代表，强调离线构建 item similarity，在线快速推荐 | 说明 ItemKNN 为什么适合生产：物品相似图比用户邻域更稳定、更可缓存 |
| [Item-based Top-N Recommendation Algorithms](https://dblp.org/rec/journals/tois/DeshpandeK04) | 2004 | ACM TOIS | 把 item-based 方法推到 Top-N recommendation，强调 item similarity 与 basket-to-candidate aggregation | 更贴近隐式反馈/Top-N 场景的 ItemKNN |
| [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/abs/1205.2618) | 2009 | UAI | 提出 BPR-Opt 和 LearnBPR，用 pairwise ranking 直接优化隐式反馈推荐 | BPR-MF 的原始出处；同一框架也能训练 BPR-kNN |
| [Collaborative Filtering for Implicit Feedback Datasets](https://doi.org/10.1109/ICDM.2008.22) | 2008 | ICDM | Weighted Regularized MF/ALS 的经典隐式反馈矩阵分解基线 | BPR-MF 在实验中重点对比的 WR-MF 背景 |
| [Matrix Factorization Techniques for Recommender Systems](https://doi.org/10.1109/MC.2009.263) | 2009 | IEEE Computer | 矩阵分解推荐的高层综述和 Netflix Prize 时代代表性总结 | 帮助理解 MF 为什么成为 BPR 的主打模型类 |

如果只读两篇：先读 Sarwar et al. 2001，再读 Rendle et al. 2009。前者理解“邻居式协同过滤”，后者理解“隐式反馈推荐为什么要用排序目标”。

## 2. ItemKNN / Item-based CF

### 2.1 背景问题

早期协同过滤常用 user-based kNN：找和目标用户相似的用户，再看这些邻居用户喜欢什么。Sarwar et al. 2001 指出，大型网站上用户数快速增长，user-based CF 会遇到三个问题：

- 每次推荐都要在大量用户中找邻居，在线计算压力大。
- 用户-物品矩阵非常稀疏，用户相似度不稳定。
- 覆盖率和质量容易受冷门物品、低交互用户影响。

Item-based CF 的转向是：**先离线分析用户-物品矩阵，得到 item-item similarity；在线阶段只根据用户已评分/已交互物品聚合候选 item 的相似度。**

这个想法的关键工程收益是：物品关系通常比用户兴趣变化慢，item similarity matrix 可以离线预计算、缓存和增量更新。

### 2.2 核心算法

给定用户-物品评分矩阵 \(R\)，ItemKNN 先计算物品 \(i,j\) 的相似度 \(s_{ij}\)。Sarwar et al. 2001 讨论了三类相似度：

1. **Cosine similarity**：把物品看成由用户评分组成的向量。
   \[
   s_{ij}=\frac{\vec r_i \cdot \vec r_j}{\|\vec r_i\|\|\vec r_j\|}
   \]

2. **Correlation / Pearson similarity**：在共同评分用户上计算相关性，减少物品评分均值差异的影响。

3. **Adjusted cosine similarity**：对每个用户减去该用户的平均评分，修正“有些用户整体打分偏高/偏低”的尺度偏差。
   \[
   s_{ij}=
   \frac{\sum_{u \in U_{ij}}(R_{u,i}-\bar R_u)(R_{u,j}-\bar R_u)}
   {\sqrt{\sum_{u \in U_{ij}}(R_{u,i}-\bar R_u)^2}
   \sqrt{\sum_{u \in U_{ij}}(R_{u,j}-\bar R_u)^2}}
   \]

预测用户 \(u\) 对目标物品 \(i\) 的评分时，取用户已评分物品中与 \(i\) 最相似的邻居集合 \(N_i(u)\)，做加权平均：

\[
\hat R_{u,i}=
\frac{\sum_{j \in N_i(u)} s_{ij} R_{u,j}}
{\sum_{j \in N_i(u)} |s_{ij}|}
\]

Sarwar et al. 也讨论了 regression-style prediction：不直接使用相似物品的原始评分，而用线性回归把相似物品评分映射到目标物品评分尺度上。

### 2.3 Top-N ItemKNN

Deshpande & Karypis 2004 更直接面向 Top-N recommendation。核心不是预测一个显式评分，而是对候选物品排序。形式上可以理解为：

\[
score(u,i)=\sum_{j \in I_u^+} g(s_{ij}, w_j)
\]

其中 \(I_u^+\) 是用户已经交互过的物品集合，\(g\) 是把候选物品 \(i\) 与用户“basket”中历史物品相似度合并的函数。常见做法包括 sum、weighted sum、top-k sum、normalized sum 等。

这版 ItemKNN 对后来的隐式反馈推荐更重要，因为真实业务里经常没有明确评分，只有点击、购买、观看、收藏等正反馈。

### 2.4 主要 insight

ItemKNN 的核心 insight 不是“最近邻”本身，而是：

> 在大规模稀疏推荐中，item-item 关系比 user-user 关系更稳定、更适合离线建模；在线推荐可以退化成对用户历史交互物品邻域的快速聚合。

这也是它经久不衰的原因。即使在深度推荐时代，item-to-item 相似图仍常作为召回、解释、冷启动补充或候选生成模块。

### 2.5 局限

- **静态相似度**：经典 ItemKNN 的 \(s_{ij}\) 通常是全局的，不随用户、时间、上下文动态变化。
- **目标不完全一致**：显式评分预测版优化 RMSE/MAE，不一定等价于 Top-N ranking。
- **流行度偏置**：热门物品拥有更多共现，容易主导相似图。
- **冷启动问题**：新物品或新用户缺少交互，无法可靠估计相似度。
- **相关不等于有用**：相似/共现的物品不一定能提升当前用户的真实决策效用。

## 3. BPR-MF

### 3.1 背景问题

隐式反馈推荐里，训练数据通常只有正样本：用户点击、购买、观看过某些物品。未观察到的用户-物品对不等于负反馈，可能是用户不知道、没曝光、没机会点击。

Rendle et al. 2009 的关键判断是：隐式反馈推荐的目标不是预测评分，而是为每个用户生成个性化排序。因此应该直接优化 pairwise ranking。

### 3.2 BPR 训练数据

设隐式反馈集合为：

\[
S \subseteq U \times I
\]

对用户 \(u\)，已观察正反馈物品集合为：

\[
I_u^+ = \{i \in I : (u,i)\in S\}
\]

BPR 构造训练三元组：

\[
D_S = \{(u,i,j) \mid i \in I_u^+, j \in I \setminus I_u^+\}
\]

语义是：对用户 \(u\)，已观察物品 \(i\) 应该排在未观察物品 \(j\) 前面。

### 3.3 BPR-Opt

定义模型对用户 \(u\) 和物品 \(i\) 的打分为 \(\hat x_{ui}\)，pairwise 差值为：

\[
\hat x_{uij} = \hat x_{ui} - \hat x_{uj}
\]

BPR 最大化：

\[
\text{BPR-Opt} =
\sum_{(u,i,j)\in D_S}\ln \sigma(\hat x_{uij})
-\lambda_\Theta \|\Theta\|^2
\]

等价地，最小化常见 pairwise logistic loss：

\[
\mathcal{L}_{BPR} =
-\sum_{(u,i,j)\in D_S}\ln \sigma(\hat x_{ui}-\hat x_{uj})
+\lambda_\Theta \|\Theta\|^2
\]

BPR 论文强调它和 AUC 优化有直接关系：AUC 关心正样本是否排在负样本前面，BPR 用可微的 \(\ln\sigma(\cdot)\) 替代不可微的 0/1 排序指标。

### 3.4 BPR-MF 是什么

BPR 本身是目标函数和学习算法，不限定模型结构。BPR-MF 是把矩阵分解作为打分器：

\[
\hat x_{ui}=w_u^\top h_i
\]

其中 \(w_u\) 是用户向量，\(h_i\) 是物品向量。训练时随机采样三元组 \((u,i,j)\)，用 SGD 更新 \(w_u,h_i,h_j\)，使 \(\hat x_{ui}>\hat x_{uj}\)。

所以要区分：

- **MF**：模型类，定义怎么给 user-item pair 打分。
- **BPR**：排序目标，定义什么样的打分是好的。
- **BPR-MF**：用 MF 打分器优化 BPR pairwise loss。

Rendle et al. 2009 同时还展示了 BPR-kNN：把 item-item similarity matrix 当作可学习参数，用同样的 BPR 目标训练 kNN 模型。这一点很重要，因为 BPR 并不只属于矩阵分解。

### 3.5 主要 insight

BPR-MF 的核心 insight 是：

> 隐式反馈推荐的监督信号天然是相对偏好，而不是绝对评分；模型应该学习“已观察物品比未观察物品更应该排在前面”，而不是把未观察项简单当 0 做回归。

这使 BPR-MF 成为长期强基线：简单、可扩展、容易负采样，且目标和 Top-N ranking 更一致。

### 3.6 局限

- **未观察不等于负样本**：BPR 默认 \(i \in I_u^+\) 优于 \(j \notin I_u^+\)，但 \(j\) 可能只是没曝光。
- **曝光偏差**：用户看到什么由平台策略决定，点击日志不是随机采样。
- **采样策略影响很大**：uniform negative sampling、popularity-aware sampling、hard negative sampling 会改变学习结果。
- **优化 AUC-like 排序，不等于业务指标**：AUC 高不一定代表 NDCG、Recall@K、收益、覆盖率或多样性高。
- **默认无上下文/时间**：经典 BPR-MF 不知道会话、时间、价格、节假日、位置等 context。
- **可解释性弱于 ItemKNN**：latent factors 很难像 item-item 邻居那样直接解释。

## 4. ItemKNN 与 BPR-MF 的关系

两者不是简单的“老方法 vs 新方法”，而是解决了协同过滤的两个不同层面。

| 维度 | ItemKNN | BPR-MF |
|---|---|---|
| 表征方式 | 显式 item-item similarity matrix | user/item latent factors |
| 推荐机制 | 根据用户历史物品邻居聚合候选分数 | 对每个 user-item pair 直接打分排序 |
| 优化目标 | 经典版本多是启发式相似度 + weighted sum；Top-N 版更贴近排序但未必端到端优化 | pairwise ranking objective |
| 优势 | 简单、快、稳定、可解释、适合召回 | 目标贴合隐式反馈 ranking，泛化能力更强 |
| 弱点 | 相似度静态，难表达高阶偏好 | 可解释性弱，受负采样和曝光偏差影响 |
| 桥接点 | BPR-kNN 可以用 BPR 训练 item similarity | BPR 也可视作学习一个更适合 ranking 的相似/打分函数 |

最值得保留的观点是：**ItemKNN 给了可解释的邻居结构，BPR 给了让邻居/打分函数对任务目标负责的训练原则。**

## 5. 对 CastCF / CoCast 的启发

当前代码里已经有明显的 CF 影子：

- `shape_knn` 对应最朴素的 ItemKNN：只用 \(X_{past}\) 的相似度找邻居。
- `context_knn` 对应 context-only item/case similarity。
- `castcf_lite` 对应 hybrid ItemKNN：用 shape/context/meta 的静态加权相似度 rerank。
- `learned_metric` 和 `training_pairs` 已经接近 BPR：用“未来轨迹是否更接近”构造 positive/negative case pair，再训练 pairwise scorer。

更准确的论文 framing 可以写成：

> CastCF generalizes ItemKNN from item recommendation to forecasting-case retrieval, and replaces static similarity with forecast-utility-supervised pairwise metric learning inspired by BPR.

### 5.1 推荐系统到时间序列 case retrieval 的映射

| 推荐系统 | CastCF / forecasting retrieval |
|---|---|
| user | 当前 query case / 待预测实体窗口 |
| item | 历史 forecasting case |
| user history / basket | query 的 past shape、context、meta |
| item similarity | case-case similarity |
| rating / click / purchase | 历史 case 的 future trajectory 对当前 query 的预测效用 |
| Top-N recommendation | Top-K forecast-useful historical cases |
| BPR positive item | future trajectory 更接近 query future 的历史 case |
| BPR negative item | shape/context 看似相似但 future utility 更差的历史 case |

### 5.2 CastCF 里最该借鉴的不是 MF，而是 BPR 的监督形式

直接把 BPR-MF 套到时间序列上未必自然，因为 CastCF 没有传统 user-item interaction matrix。更自然的是借鉴 BPR 的 pairwise ranking principle：

\[
D_{CF}=
\{(q,i^+,i^-)\mid d_{future}(q,i^+) < d_{future}(q,i^-)-m\}
\]

\[
\mathcal{L}_{CastCF-rank}
=-\sum_{(q,i^+,i^-)}
\ln\sigma(s_\theta(q,i^+)-s_\theta(q,i^-))
+\lambda\|\theta\|^2
\]

其中 \(d_{future}\) 可以是 MAE、MSE、trend distance、peak distance 或 horizon-weighted distance。这样训练出来的 \(s_\theta\) 不是“看起来相似”，而是“对未来预测更有用”。

### 5.3 建议在论文实验中加入的经典 CF 对照

| Baseline | 对应经典思想 | 目的 |
|---|---|---|
| `shape_itemknn` | Sarwar-style ItemKNN | 证明只看 past shape 的邻居不足 |
| `context_itemknn` | context-aware similarity | 证明 context 单独有信号 |
| `static_hybrid_itemknn` | weighted item/case similarity | 对照手工权重融合 |
| `bpr_metric_knn` | BPR-kNN / pairwise learned similarity | 证明 forecast-utility ranking loss 比静态相似度更好 |
| `wrmf_like_metric` | Hu-Koren-Volinsky implicit MF 思路 | 如果要对比 pointwise/weighted objective，可以作为 BPR 的反面基线 |

评价上不要只看 MAE/MSE，应该报告：

- `NFD@K`：邻居未来轨迹与 query future 的距离。
- `RecallUseful@K`：Top-K 是否覆盖 future-distance 最小的一批 oracle useful cases。
- `same-past-different-future` subset：历史形状相似但 context 不同的样本。
- `same-context-different-past` subset：context 相似但历史状态不同的样本。
- event / promotion / holiday / stockout / weather-shift subsets。

### 5.4 一个可写进 Related Work 的差异点

可以这样组织：

1. ItemKNN 证明了 item/case similarity graph 对大规模推荐有效，但经典相似度是静态的、任务无关的。
2. BPR 证明了隐式反馈推荐应该优化 pairwise ranking，而不是评分回归。
3. CastCF 借鉴两者，但问题对象不同：它不是推荐 item，而是检索 forecast-useful historical cases；它的正负样本不是点击/未点击，而是由未来轨迹效用定义。

这能形成一个比较清楚的创新边界：

> From item similarity to forecast-case utility similarity; from interaction ranking to future-trajectory utility ranking.

## 6. 读论文时应抓的关键问题

读 ItemKNN 时重点看：

- 相似度到底在共同评分用户上怎么算？
- 为什么 item-based 比 user-based 更适合大规模在线推荐？
- weighted sum prediction 和 Top-N scoring 有什么差异？
- 物品相似图什么时候会失效？

读 BPR-MF 时重点看：

- 为什么 implicit feedback 不能直接当 0/1 rating regression？
- \(D_S\) 三元组怎么构造？
- BPR-Opt 和 AUC 的关系是什么？
- BPR-MF 与 WR-MF 的差异到底是模型差异，还是优化目标差异？
- negative sampling 会引入什么偏差？

读到 CastCF 上，应追问：

- 我们的 positive/negative case 是否真的代表 forecast utility，而不是只代表 future shape 相似？
- 是否有 hard negative：past shape 很像但 future 完全不同？
- 静态加权相似度和 pairwise learned metric 的差异是否足够明显？
- 如果 recent baseline 很强，CastCF 的收益是否集中在 context-sensitive subset？

## 7. 可直接引用的 BibTeX

```bibtex
@inproceedings{sarwar2001item,
  title = {Item-based Collaborative Filtering Recommendation Algorithms},
  author = {Sarwar, Badrul and Karypis, George and Konstan, Joseph and Riedl, John},
  booktitle = {Proceedings of the 10th International Conference on World Wide Web},
  pages = {285--295},
  year = {2001},
  doi = {10.1145/371920.372071}
}

@article{linden2003amazon,
  title = {Amazon.com Recommendations: Item-to-Item Collaborative Filtering},
  author = {Linden, Greg and Smith, Brent and York, Jeremy},
  journal = {IEEE Internet Computing},
  volume = {7},
  number = {1},
  pages = {76--80},
  year = {2003},
  doi = {10.1109/MIC.2003.1167344}
}

@article{deshpande2004item,
  title = {Item-Based Top-N Recommendation Algorithms},
  author = {Deshpande, Mukund and Karypis, George},
  journal = {ACM Transactions on Information Systems},
  volume = {22},
  number = {1},
  pages = {143--177},
  year = {2004},
  doi = {10.1145/963770.963776}
}

@inproceedings{rendle2009bpr,
  title = {BPR: Bayesian Personalized Ranking from Implicit Feedback},
  author = {Rendle, Steffen and Freudenthaler, Christoph and Gantner, Zeno and Schmidt-Thieme, Lars},
  booktitle = {Proceedings of the Twenty-Fifth Conference on Uncertainty in Artificial Intelligence},
  pages = {452--461},
  year = {2009},
  url = {https://arxiv.org/abs/1205.2618}
}

@inproceedings{hu2008implicit,
  title = {Collaborative Filtering for Implicit Feedback Datasets},
  author = {Hu, Yifan and Koren, Yehuda and Volinsky, Chris},
  booktitle = {2008 Eighth IEEE International Conference on Data Mining},
  pages = {263--272},
  year = {2008},
  doi = {10.1109/ICDM.2008.22}
}

@article{koren2009matrix,
  title = {Matrix Factorization Techniques for Recommender Systems},
  author = {Koren, Yehuda and Bell, Robert and Volinsky, Chris},
  journal = {Computer},
  volume = {42},
  number = {8},
  pages = {30--37},
  year = {2009},
  doi = {10.1109/MC.2009.263}
}
```

## 8. 来源核对

- Sarwar et al. 2001：WWW HTML/PDF 页面核对标题、作者、会议、算法结构、item similarity 与 weighted-sum prediction。
- Rendle et al. 2009：arXiv 页面、DBLP 页面与 PDF 文本核对 UAI 2009、BPR-Opt、LearnBPR、BPR-MF/BPR-kNN、AUC 实验。
- Deshpande & Karypis 2004：DBLP/CiNii 页面核对 TOIS、DOI、Top-N item-based recommendation 摘要。
- Linden et al. 2003：DOI/CiNii 页面核对 IEEE Internet Computing、页码与 item-to-item CF 定位。
- Semantic Scholar API 本次返回 429，因此没有把其 citation count 写入表格。
