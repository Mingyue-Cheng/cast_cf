# Context-aware Time Series Forecasting 数据集调研

更新时间：2026-06-02

## 0. 结论先行

如果目标是服务当前 `CoCast / Context-aware Collaborative Forecasting` 方向，最有价值的数据集不是普通长期预测 benchmark，而是能回答这个问题的数据：

> 同样的历史曲线，在不同情境下是否走向不同未来；不同历史曲线，在相似情境下是否走向相似未来？

按这个标准，我建议分三层使用：

| 优先级 | 数据集/Benchmark | 用途 |
|---|---|---|
| P0 | FreshRetailNet-50K, Favorita, M5, GEFCom2014/PSML, Beijing Air Quality | 最适合做 `context-aware retrieval` 方法主实验，结构化 context 明确，实体多，可构造 retrieval memory |
| P1 | Context is Key, TemporalBench, CGTSF, Time-MMD, MoTime, MTBench | 适合做 context reasoning / text-context / event-conditioned diagnostic，不一定都适合大规模训练 |
| P2 | ECL, ETT, PEMS-SF, Bitcoin/Bitcoin-News, Monash 部分数据 | 可做 sanity check 或补充消融，但单独支撑顶会 claim 会偏弱 |

关键判断：**不要把所有带 covariate 的 forecasting dataset 都叫 context-aware dataset**。真正适合当前工作的数据集至少要满足以下条件之一：

1. 有明确的 `C_future`：预测时可提前获得的未来情境，例如 calendar、holiday、planned promotion、weather forecast、scheduled event。
2. 有实体元信息 `M`：store、product、sensor、city、user、region、category 等，能支持 collaborative neighbor retrieval。
3. 有事件或文本 context：news、incident、stockout、maintenance、storm、clinical event、policy/event narrative。
4. 能构造 hard subsets：`same past, different future` 和 `different past, similar future`。
5. 能严格避免 label leakage：不能把实际未来 weather、未来销量统计、未来库存结果当成预测时可知 context。

---

## 1. 直接面向 context-aware / multimodal TSF 的 benchmark

### 1.1 Context is Key (CiK)

链接：

- Paper: [Context is Key: A Benchmark for Forecasting with Essential Textual Information](https://arxiv.org/abs/2410.18959)
- OpenReview: [ICML 2025 page](https://openreview.net/forum?id=ih2WuBT1Fn)
- Code: [ServiceNow/context-is-key-forecasting](https://github.com/ServiceNow/context-is-key-forecasting)
- Dataset: [ServiceNow/context-is-key on Hugging Face](https://huggingface.co/datasets/ServiceNow/context-is-key)

定位：专门评估“文本 context 是否对 forecast 必要”的 benchmark。

核心信息：

- ICML 2025 benchmark。
- 71 个 distinct forecasting tasks，覆盖 7 个真实应用域。
- 使用约 2,644 条来自公开数据源的 time series。
- Hugging Face 数据卡显示有 355 行样本，只有 test split；`test` 是最新版本，`ICML2025` 是论文实验版本。
- context 类型被显式标注为：
  - `Intemporal information`
  - `Historical information`
  - `Future information`
  - `Covariate information`
  - `Causal information`
- 来源包括 PeMS traffic、solar irradiance/cloud cover、solar PV、Montreal fire logs、Causal Chambers、USA unemployment、NN5 ATM、electricity consumption 等。

适合做什么：

- 很适合做 context-use diagnostic。
- 适合验证模型是否真的理解 `future information / causal information / covariate information`。
- 适合给 CoCast 加一个小型外部诊断实验：比较 retrieval neighbor 是否随文本 context 改变。

限制：

- 它主要是 test-only benchmark，不适合作为大规模训练集。
- 任务是人工设计的 context-aided forecasting task，不是自然生产系统中的 continuous forecasting log。
- 对 CoCast 来说，更像“诊断集”，不是主训练数据。

CoCast 价值：

- 可以借鉴它的 context taxonomy。
- 可以借鉴 `Region of Interest` 和 context relevance validation。
- 可以把 `future/covariate/causal context` 作为当前项目的 diagnostic metrics 维度。

---

### 1.2 TemporalBench

链接：

- Paper: [TemporalBench: A Benchmark for Evaluating LLM-Based Agents on Contextual and Event-Informed Time Series Tasks](https://arxiv.org/abs/2602.13272)
- Dataset: [Melady/TemporalBench](https://huggingface.co/datasets/Melady/TemporalBench)
- Leaderboard: [TemporalBench Leaderboard](https://huggingface.co/spaces/Melady/TemporalBench_Leaderboard)

定位：评估 LLM/Agent 是否能做 contextual temporal reasoning 和 event-informed prediction。

核心信息：

- 2026 年 arXiv benchmark。
- 四类任务：
  - T1: Historical Time-Series Understanding
  - T2: Context-Free Future Prediction
  - T3: Contextual Temporal Reasoning
  - T4: Event-Informed Prediction
- 覆盖 4 个真实域：
  - Retail: FreshRetailNet-50K
  - Healthcare: MIMIC-IV
  - Energy: PSML
  - Physical systems: Causal Chambers
- 支持 multiple-choice reasoning tasks 和 numerical forecasting objectives。

适合做什么：

- 非常适合做 event-conditioned evaluation。
- T2 vs T4 的对比可以直接检验“加入事件 context 后预测是否合理变化”。
- 适合作为 CoCast 的额外 evaluator，而不是第一版主训练数据。

限制：

- 主要面向 LLM-based agents，不是传统 TSF 模型 benchmark。
- 部分原始数据源有访问门槛，例如 MIMIC-IV。
- T4 事件有一部分是注入/构造的，不一定等价于真实业务事件。

CoCast 价值：

- 它的 T2/T4 设计非常适合转化成 CoCast 的 `counterfactual context test`。
- 可以把 “同一历史窗口 + 不同未来事件” 作为 hard retrieval 场景。

---

### 1.3 CGTSF / ChatTime 数据集

链接：

- Dataset: [ChengsenWang/CGTSF](https://huggingface.co/datasets/ChengsenWang/CGTSF)
- Paper: [ChatTime: A Unified Multimodal Time Series Foundation Model Bridging Numerical and Textual Data](https://arxiv.org/abs/2412.11376)

定位：Context-Guided Time Series Forecasting，偏“文本辅助信息 + 数值序列”的 multimodal TSF。

核心信息：

- 包含 3 个 multimodal datasets：
  - MSPG：Melbourne 27 个光伏站点，13 个月，15 分钟频率。
  - LEU：London 16 个家庭，24 个月，30 分钟频率。
  - PTF：Paris 32 个交通探测器，12 个月，小时频率。
- Hugging Face 数据卡显示共 33,693 rows，约 94.5 MB。
- 辅助文本包括：
  - dataset background
  - weather forecast from Open-Meteo
  - weather code, temperature, sunrise/sunset
  - date, day of week, holiday information
- 作者强调只使用 background、weather、date，避免 future data leakage。

适合做什么：

- 快速验证 text-context TSF。
- 能作为 CoCast 的轻量 multimodal prototype。
- energy / traffic 两个域都贴合当前项目。

限制：

- 规模中等，实体数不多。
- context 已被拼成文本，若要做多视角 retrieval，需要重新拆出 weather/date/entity fields。

CoCast 价值：

- 第一版可以直接用它做 `context-text encoder + retrieval rerank`。
- 特别适合做 “weather/context 相似但历史 shape 不完全相似” 的 neighbor retrieval 测试。

---

### 1.4 Time-MMD

链接：

- Paper: [Time-MMD: Multi-Domain Multimodal Dataset for Time Series Analysis](https://arxiv.org/abs/2406.08627)
- NeurIPS PDF: [Time-MMD paper](https://papers.nips.cc/paper_files/paper/2024/file/8e7768122f3eeec6d77cd2b424b72413-Paper-Datasets_and_Benchmarks_Track.pdf)
- Code/Data: [AdityaLab/Time-MMD](https://github.com/AdityaLab/Time-MMD)

定位：多领域、多模态 time series dataset，数值序列与文本序列对齐。

核心信息：

- NeurIPS 2024 Datasets and Benchmarks。
- 覆盖 9 个 primary data domains。
- 数据由 numerical sequences 和 textual sequences 构成。
- 文本格式包含 `start_date, end_date, fact, pred`，支持与时间区间对齐。
- 官方 repo 提供 Short-Term Forecasting、Long-Term Forecasting、Imputation、Anomaly Detection 的下游任务说明。

适合做什么：

- 评估 dynamic textual context 对 TSF 的影响。
- 做跨领域 context-aware forecasting。
- 对 CoCast 来说，可以用文本时间段作为 `C_past/C_future`。

限制：

- MoTime 论文指出 Time-MMD 每个域的 channel 数较少，没有一个域超过 11 channels。
- 如果主打 collaborative retrieval，实体规模和实体 metadata 可能不如 retail/energy 数据集充分。

CoCast 价值：

- 适合作为 P1 外部验证，尤其是 “text-numerical alignment”。
- 不建议作为第一版唯一主数据集。

---

### 1.5 MoTime

链接：

- Paper: [MoTime: A Dataset Suite for Multimodal Time Series Forecasting](https://arxiv.org/abs/2505.15072)
- HTML: [ar5iv rendering](https://ar5iv.labs.arxiv.org/html/2505.15072)
- Dataset: [Kaggle MoTime](https://www.kaggle.com/datasets/krissssss/multimodal-time-series-forecasting)

定位：面向 entity-centric forecasting 的 multimodal dataset suite，强调 static context、metadata、text、image 和 cold-start。

核心信息：

- 覆盖 8 个数据集，来自 e-commerce、web traffic、media、user behavior 等域。
- 主要数据集包括：
  - PixelRec
  - TaobaoFashion
  - MovieLens
  - AmazonReview
  - Tianchi
  - News
  - WikiPeople
  - VISUELLE
- 每个 series 绑定文本、图像、metadata 中的一种或多种。
- 支持两种场景：
  - varying-history forecasting
  - cold-start forecasting
- 论文特别强调：即使历史曲线相似，如果缺少 item identity / category / description，也可能误判未来。

适合做什么：

- 非常适合做 entity metadata-aware forecasting。
- 适合做 cold-start 和 semantic retrieval。
- 适合检验 “不同 item 历史相似，但语义属性导致未来分歧”。

限制：

- 多数 context 是 static entity modality，不是未来事件或未来可知 context。
- 如果 CoCast 主 claim 是 `C_future` 调制 retrieval，需要额外构造 calendar/event/context split。

CoCast 价值：

- 很适合做 `M_i` 和 semantic neighbor retrieval。
- 可以支持一个强实验：仅凭 item text/image metadata 检索历史案例，判断 cold-start 或短历史 forecasting。

---

### 1.6 MTBench

链接：

- Paper: [MTBench: A Multimodal Time Series Benchmark for Temporal Reasoning and Question Answering](https://arxiv.org/abs/2503.16858)
- Project: [Graph and Geometric Learning Lab MTBench](https://graph-and-geometric-learning.github.io/projects/mtbench)
- OpenReview: [MTBench OpenReview](https://openreview.net/forum?id=4lUoPAXrCe)

定位：金融和天气域的 time series + text benchmark，偏 LLM temporal reasoning。

核心信息：

- 覆盖 finance 和 weather 两个域。
- 支持 forecasting、semantic trend analysis、technical indicator prediction、news-driven QA。
- 项目页展示了 `TS only` 与 `TS + Text` 的对比评估。

适合做什么：

- text-conditioned forecast / trend reasoning。
- 金融新闻驱动的 context-aware 预测。
- 多模态 reasoning benchmark。

限制：

- 更偏 LLM evaluation，不一定适合传统 deep TSF 训练。
- 金融数据中 leakage 和时间对齐需要特别谨慎。

CoCast 价值：

- 可以作为文本事件 context 的外部验证。
- 不建议作为第一版主实验，因为金融预测噪声大，reviewer 容易质疑可预测性。

---

## 2. 适合二次构造成 context-aware forecasting benchmark 的真实数据集

### 2.1 FreshRetailNet-50K

链接：

- Paper: [FreshRetailNet-50K](https://arxiv.org/abs/2505.16319)
- Dataset: [Dingdong-Inc/FreshRetailNet-50K](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K)

定位：生鲜零售需求预测，带 stockout、promotion、weather、holiday 等 context。

核心信息：

- 50,000 条 store-product time series。
- 898 家店，18 个城市，863 个生鲜 SKU。
- Hourly sales data。
- Hugging Face 数据卡显示约 4.85M rows，train 4.5M rows，eval 350k rows。
- 字段包括：
  - `city_id`, `store_id`, `product_id`, category ids
  - `sale_amount`, `hours_sale`
  - `stock_hour6_22_cnt`, `hours_stock_status`
  - `discount`
  - `holiday_flag`, `activity_flag`
  - `precpt`, `avg_temperature`, `avg_humidity`, `avg_wind_level`

为什么强：

- 有实体：store/product/category/city。
- 有未来可知或近似可知 context：discount、holiday、activity、weather forecast 变体。
- 有事件：stockout。
- 有密集 hourly pattern。
- 非常适合构造 `same past, different future`：
  - 历史销量形状相似，但一个未来有 discount/stockout/holiday/weather change，另一个没有。

风险：

- `sale_amount` 在 stockout 时是 censored demand，不等于真实 demand。
- 如果用 stock status，要区分它是 prediction target 的干扰因素、事件 context，还是 label reconstruction signal。

CoCast 推荐用法：

- 作为 P0 主实验之一。
- Case 定义：
  - `X_past`: recent hourly sales
  - `C_future`: discount, holiday/activity, weather forecast
  - `M`: city/store/product/category
  - `Y_future`: future recovered/observed sales
- Hard negatives：
  - same sales history + different discount
  - same product/category + different city weather
  - same store/product + stockout vs non-stockout future

---

### 2.2 M5 Forecasting Accuracy

链接：

- Paper: [The M5 competition: Background, organization, and implementation](https://www.sciencedirect.com/science/article/pii/S0169207021001187)
- Dataset mirror: [Zenodo M5 Forecasting Accuracy dataset](https://zenodo.org/records/12636070)

定位：Walmart hierarchical retail sales forecasting。

核心信息：

- 42,840 条 hierarchical unit sales time series。
- 论文明确指出 M5 相比以往 M competitions 加入了 exogenous/explanatory variables、grouped correlated time series 和 intermittent series。
- 数据集文件包括 sales、calendar、sell prices 等。
- 常用 context：
  - calendar/event
  - SNAP
  - sell price
  - item/store/state/category hierarchy

为什么强：

- 标准、可复现、社区熟悉。
- 有大规模 item-store entity。
- 有 calendar/price/event context。
- 很适合 collaborative forecasting。

风险：

- 很多强 baseline 已经在 M5 上做过，单纯提升 M5 分数不新。
- 日粒度，事件 impact 可能不如 hourly retail 数据明显。

CoCast 推荐用法：

- 用来证明方法在经典 hierarchical retail benchmark 上有效。
- 重点不是总榜分数，而是 context-sensitive subset：
  - event days
  - SNAP days
  - price-change periods
  - intermittent demand product families

---

### 2.3 Corporación Favorita

链接：

- Nixtla processed dataset: [Favorita - datasetsforecast](https://nixtlaverse.nixtla.io/datasetsforecast/favorita.html)
- Kaggle source: [Corporación Favorita Grocery Sales Forecasting](https://www.kaggle.com/c/favorita-grocery-sales-forecasting/data)

定位：Ecuador grocery retail sales forecasting。

核心信息：

- Nixtla 文档显示 processed Favorita 包含 371,312 条 series。
- 时间范围：2013-01 到 2017-08。
- daily sales history。
- additional information：
  - promotions
  - items
  - stores
  - holidays
  - geographic hierarchy: states, cities, stores
- ContextFormer 论文使用 Store Sales Competition 版本，说明其包含 34 个 product families、55 家 Favorita stores，并提供 store metadata 和 oil prices 等 time-varying metadata。

为什么强：

- 比 M5 更有地理和国家事件特色。
- promotions / holidays / oil price 对需求有明显 context effect。
- `store × product family` 的 neighbor retrieval 很自然。

风险：

- 原始 Kaggle 数据较大，预处理工作量高。
- oil price 是否应作为未来可知变量需要明确设定；如果是预测时实际不可知，应只使用 forecasted oil 或历史 oil。

CoCast 推荐用法：

- P0 主实验。
- 构造：
  - `M`: store/city/state/type/cluster/product family
  - `C_future`: holiday, promotion plan, known calendar
  - `C_past`: previous promotion/oil/holiday
  - hard negative: similar sales past but different promotion/holiday/oil regime

---

### 2.4 GEFCom2014 Load / Solar / Wind / Price

链接：

- Paper PDF: [Probabilistic energy forecasting: Global Energy Forecasting Competition 2014 and beyond](https://robjhyndman.com/papers/gefcom2014.pdf)
- Data note: [GEFCom2014 Load Forecasting Data](https://blog.drhongtao.com/2017/03/gefcom2014-load-forecasting-data.html)

定位：energy forecasting competition，包含 load、price、wind、solar tracks。

核心信息：

- GEFCom2014 是 probabilistic energy forecasting competition。
- 论文介绍了 load、price、wind、solar 四个 tracks。
- Tao Hong 的数据说明中提到：
  - load forecasting track 第一轮给 69 个月 hourly load data 和 117 个月 hourly temperature data。
  - 完整数据有约 7 年 matching load and temperature data。
  - GEFCom2014-E 扩展集包含 11 年 hourly temperature 和 9 年 hourly load。

为什么强：

- Energy load 对 weather/context 的依赖非常明确。
- 滚动预测设置天然适合防止 leakage。
- 可以构造极端温度、节假日、季节切换等 context-sensitive subset。

风险：

- 老数据集，任务相对经典。
- entity metadata 可能不如 retail 丰富。

CoCast 推荐用法：

- P0/P1 energy experiment。
- 用 temperature/weather forecast 作为 `C_future`。
- 做 `same load shape, different temperature future` hard subset。

---

### 2.5 PSML

链接：

- Paper: [A Multi-scale Time-series Dataset with Benchmark for Machine Learning in Decarbonized Energy Grids](https://arxiv.org/abs/2110.06324)
- Code/Data: [Zenodo PSML](https://zenodo.org/records/5663995)

定位：decarbonized energy grids 的 multi-scale time-series dataset。

核心信息：

- Zenodo 描述显示 PSML 包含：
  - electric load
  - renewable generation
  - weather
  - voltage and current measurements
  - multiple spatio-temporal scales
- 支持 use cases：
  - disturbance event detection/classification/localization
  - hierarchical forecasting under uncertainties and extreme events
  - physically constrained synthetic generation

为什么强：

- 有物理系统结构和多尺度 context。
- 有 extreme events / disturbance 语义。
- 适合做 event-aware forecasting。

风险：

- 来源是 co-simulation，不是纯真实观测。
- 对普通 forecasting reviewer 来说，可能需要解释为什么仿真系统仍有外部有效性。

CoCast 推荐用法：

- 与 TemporalBench 的 PSML 任务呼应。
- 作为 energy event-conditioned 评估集，而不是唯一主数据。

---

### 2.6 Beijing Air Quality

链接：

- ContextFormer paper: [Context Matters: Leveraging Contextual Features for Time Series Forecasting](https://arxiv.org/abs/2410.12672)
- 常见来源：Beijing Multi-Site Air-Quality Data / UCI-style releases

定位：空气质量预测，污染物序列 + 气象 context。

ContextFormer 论文中的设置：

- 12 个 Beijing locations。
- hourly pollutant concentration。
- target channels：
  - CO, NO2, SO2, O3, PM2.5, PM10
- continuous metadata：
  - temperature
  - humidity
  - wind speed
  - pressure
  - dew point
- categorical metadata：
  - location
  - wind direction
- 时间范围：2013 到 2017。

为什么强：

- 气象变量对污染物未来变化有明确作用。
- location/wind direction 能构造 spatial-context retrieval。
- 多变量 target，适合测试 context-aware multi-view similarity。

风险：

- 如果使用“未来实际气象值”，必须说明这是 oracle future covariate；更合理的是使用 weather forecast 或只做分析性实验。

CoCast 推荐用法：

- P0/P1 环境域实验。
- hard cases：
  - same pollution past + different wind/weather future
  - different location + similar wind/weather regime

---

### 2.7 PEMS-SF / PeMS traffic

链接：

- Monash repository: [Monash Forecasting Repository](https://forecastingdata.org/)
- PeMS: [California Performance Measurement System](https://pems.dot.ca.gov/)
- ContextFormer 使用 PEMS-SF；CiK 也用 PeMS 2024 traffic occupancy data。

定位：交通 occupancy / volume forecasting。

核心信息：

- Monash repository 中 San Francisco Traffic 包含 862 条 series，hourly frequency。
- ContextFormer 使用 PEMS-SF，传感器数约 861，主要 metadata 是 sensor ID。
- CiK 使用 PeMS 2024 live data 构造 traffic context tasks。

为什么有用：

- 交通对 holiday、incident、weather、location/event 有强 context dependency。
- PeMS 原始系统是持续更新的，适合构造新鲜测试集降低记忆风险。

限制：

- PEMS-SF benchmark 本身 context 很弱，通常只有 sensor ID。
- 需要外接 calendar、holiday、weather、incident 才能成为强 context-aware dataset。

CoCast 推荐用法：

- 不建议直接拿 PEMS-SF 做主 claim。
- 建议二次构造：
  - add calendar/holiday
  - add weather
  - add traffic incident/event if available
  - 构造 holiday/freeway closure/extreme weather subset

---

### 2.8 ETT / ECL / Bitcoin / Bitcoin-News

链接：

- ContextFormer paper: [Context Matters](https://arxiv.org/abs/2410.12672)
- Monash repository: [Monash Forecasting Repository](https://forecastingdata.org/)
- ECL: [UCI ElectricityLoadDiagrams20112014](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014)

定位：常见 TSF benchmark，可作为弱 context 或补充实验。

ContextFormer 论文中的 context 设置：

| Dataset | Target | Context / metadata |
|---|---|---|
| ECL | electricity consumption | user ID |
| ETTm2 | oil temperature | 6 power load features |
| Bitcoin | bitcoin price | 17 continuous factors，例如 hash rate、block size、mining difficulty、search trends 等 |
| Bitcoin-News | bitcoin price | previous-day BTCUSD financial news embeddings |

适合做什么：

- 证明 context module 可以在常见 benchmark 上运行。
- Bitcoin-News 适合文本 context 小实验。
- ETT 可做 covariate-aware sanity check。

限制：

- ECL 只有 user ID 时，context 太弱。
- ETT 的 covariates 更像 multivariate forecasting features，不一定是“情境”。
- Bitcoin 金融预测噪声大，容易带来 reviewer 争议。

CoCast 推荐用法：

- 作为 P2 补充。
- 不建议作为主实验支柱。

---

## 3. 推荐实验路线

### 路线 A：结构化 context 主线，最稳

主数据：

1. FreshRetailNet-50K
2. Favorita
3. M5
4. GEFCom2014 或 Beijing Air Quality

优点：

- context 明确。
- 实体多。
- 可以构造 retrieval memory。
- 可以做 strong hard-negative mining。

适合论文主 claim：

> Forecasting neighbors should be retrieved by future-predictive context, not only by past-shape similarity.

### 路线 B：文本 / 事件 context 主线，更新颖

主数据：

1. CGTSF
2. Time-MMD
3. MoTime
4. Context is Key / TemporalBench / MTBench 作为 diagnostic

优点：

- 更贴近 LLM + TS / multimodal forecasting。
- 能突出 text/event context。
- 和 ContextFormer、CiK、TemporalBench 形成清晰 related work 对话。

风险：

- 训练和 evaluation protocol 需要自己整理。
- 部分 benchmark 偏 LLM reasoning，不一定适合传统 TSF 方法公平对比。

### 路线 C：CoCast-Bench 自建 protocol

从 P0 数据中构造两个 probe subsets：

1. `Same Past, Different Future`
   - 历史窗口相似。
   - 未来 context 不同。
   - 未来轨迹显著不同。
   - 目的：证明 shape-only retrieval 会失败。

2. `Different Past, Similar Future`
   - 历史窗口不完全相似。
   - context/entity/event 相似。
   - 未来轨迹相似。
   - 目的：证明 context-aware retrieval 能找到 shape 不近但 forecast-useful 的 neighbor。

建议优先在 FreshRetailNet-50K / Favorita / GEFCom2014 上做，因为这三类最容易找到真实 context-driven future divergence。

---

## 4. 数据集选择矩阵

| 数据集 | Context 类型 | Entity 多样性 | Future-known context | Text/Event | Retrieval 适配 | 推荐 |
|---|---:|---:|---:|---:|---:|---|
| FreshRetailNet-50K | 强 | 强 | 强 | 中 | 强 | P0 |
| Favorita | 强 | 强 | 中 | 中 | 强 | P0 |
| M5 | 强 | 强 | 强 | 中 | 强 | P0 |
| GEFCom2014 | 中 | 中 | 强 | 弱 | 中 | P0/P1 |
| Beijing AQ | 强 | 中 | 中 | 弱 | 中 | P0/P1 |
| PSML | 强 | 中 | 中 | 强 | 中 | P1 |
| CGTSF | 强 | 中 | 强 | 强 | 中 | P1 |
| Time-MMD | 强 | 中 | 中 | 强 | 中 | P1 |
| MoTime | 强 | 强 | 弱 | 强 | 强 | P1 |
| Context is Key | 强 | 弱 | 强 | 强 | 弱 | Diagnostic |
| TemporalBench | 强 | 中 | 强 | 强 | 中 | Diagnostic |
| MTBench | 强 | 中 | 中 | 强 | 中 | Diagnostic |
| PEMS-SF | 弱，需增强 | 强 | 弱 | 弱 | 中 | P2 |
| ECL | 弱 | 中 | 弱 | 弱 | 中 | P2 |
| ETT | 中 | 弱 | 中 | 弱 | 弱 | P2 |
| Bitcoin-News | 强 | 弱 | 中 | 强 | 弱 | P2 |

---

## 5. 对当前 CoCast 方案的直接建议

### 5.1 第一版不要贪多

建议第一版只选 3 个主数据域：

1. Retail：FreshRetailNet-50K 或 Favorita
2. Energy/Weather：GEFCom2014 或 PSML
3. Environment/Traffic：Beijing AQ 或 CGTSF-PTF

这样能覆盖：

- structured categorical context
- continuous weather context
- event/promotion context
- entity metadata
- future-known covariates

### 5.2 每个数据集统一成 forecasting case

统一 schema：

```text
case_i = {
  X_past: historical target window,
  C_past: historical context/covariates,
  C_future: known future context available at prediction time,
  M: entity metadata,
  Y_future: realized future target
}
```

不同数据集的映射：

| 数据集 | X_past | C_future | M | Y_future |
|---|---|---|---|---|
| FreshRetailNet | hourly sales | discount, holiday, activity, weather forecast | city/store/product/category | future sales |
| Favorita | daily sales | promotion, holiday, calendar | store/city/state/product family | future sales |
| M5 | daily unit sales | calendar, event, SNAP, price | item/store/state/category | future sales |
| GEFCom2014 | hourly load | temperature forecast, calendar | zone/station | future load |
| Beijing AQ | pollutant history | weather forecast, wind direction | location | future pollutant vector |
| CGTSF | energy/traffic history | weather/date text | site/household/detector | future values |

### 5.3 必须显式写 leakage policy

推荐在论文方法/实验中写清楚：

1. 历史 case 的 `Y_future` 可以存入 memory，因为它发生在训练历史中。
2. 当前 query 的 `Y_future` 在推理时不可见。
3. `C_future` 只允许使用预测时可提前知道的信息：
   - calendar / holiday
   - scheduled promotion / activity
   - planned event
   - weather forecast
   - official future covariate forecast
4. 如果使用真实未来天气，只能标注为 `oracle future covariate`，不能和真实部署设置混淆。
5. 文本/news 必须按发布时间对齐，不能用 horizon 之后发布的内容。

### 5.4 推荐 evaluation metrics

普通预测指标：

- MSE / MAE
- sMAPE / MASE
- pinball loss / CRPS if probabilistic

更重要的 CoCast 指标：

1. `NFD@K`: retrieved neighbor future distance
2. `Context-Sensitive Subset Error`: event / holiday / extreme weather / promotion subset
3. `Same-Past-Different-Future Accuracy`: 是否避开 shape-similar 但 future-wrong neighbors
4. `Different-Past-Similar-Future Recall`: 是否找回 context-similar 且 future-useful neighbors
5. `Counterfactual Retrieval Shift`: 改变 `C_future` 后 Top-K neighbor 是否合理变化
6. `Retrieval Reliability Calibration`: gate 是否在 bad retrieval 时降低 collaborative evidence 权重

---

## 6. 最推荐的落地顺序

### Step 1：先做 FreshRetailNet-50K

原因：

- 当前公开数据里，它最接近 CoCast 需要的完整 schema。
- 字段天然包括 entity、weather、holiday、discount、stockout。
- Hourly resolution 能构造更多 case。

### Step 2：再做 Favorita 或 M5

原因：

- 经典 retail benchmark，reviewer 熟悉。
- 可证明不是只在一个新数据集上有效。
- hierarchical/entity retrieval 非常自然。

### Step 3：补一个非零售域

候选：

- GEFCom2014：load + temperature。
- Beijing AQ：pollutants + meteorology + location。
- CGTSF：text-form weather/date context。

原因：

- 避免 reviewer 认为 CoCast 只是 retail recommender forecasting。

### Step 4：用 CiK / TemporalBench 做 diagnostic

原因：

- 它们能证明 context reasoning，而不仅是 error reduction。
- 但不适合作为主训练数据。

---

## 7. 一句话总结

如果这项工作要打成顶会方法论文，数据集选择应围绕一个核心证据链：

> shape-only retrieval 会在 context-sensitive futures 上选错邻居；context-aware collaborative retrieval 能找到在当前情境下真正 forecast-useful 的历史案例。

因此，主实验优先选择 **FreshRetailNet-50K + Favorita/M5 + GEFCom2014/Beijing AQ**；再用 **CiK/TemporalBench/CGTSF/Time-MMD/MoTime** 做文本、事件、context reasoning 的补充诊断。

