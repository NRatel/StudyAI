# 论文与FAQ

> 本文件用于论文脉络、误区澄清与延伸阅读，不替代主章节正文。

> 本文件汇总"RAG 与应用架构"模块的关键论文、常见误区与延伸资源。

---

## 一、关键论文

### 时间线总览

| 年份 | 作者 | 论文/方法 | 核心关键词 |
|------|------|-----------|------------|
| 2020 | Lewis et al. | RAG | 检索增强生成开山之作 |
| 2020 | Karpukhin et al. | DPR | 稠密段落检索，取代 BM25 |
| 2022 | Izacard et al. | Atlas | 少样本学习 + 检索增强 |
| 2023 | Gao et al. | RAG Survey | 系统性综述 RAG 范式 |
| 2023 | Shi et al. | REPLUG | 将检索器作为 LLM 的即插即用模块 |
| 2023 | Gao et al. | HyDE | 假设文档 Embedding |
| 2023 | Asai et al. | Self-RAG | 模型自主判断是否检索及检索质量 |
| 2024 | Yan et al. | CRAG | 检索结果质量纠正 |

### 逐篇简评

> **建议先读 Lewis 2020 (RAG)**。这是整个 RAG 范式的奠基论文，首次提出将检索器与生成器端到端结合的统一框架。理解这篇论文后，后续的 DPR（检索器怎么训练）、HyDE（查询怎么优化）、Self-RAG（检索怎么自适应）等工作的动机和定位就清晰了。

---

#### 1. Lewis et al., 2020 — RAG

**标题**：*Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*

**简评**：RAG 概念的开山论文。提出将预训练的检索器（DPR）与预训练的生成器（BART）端到端结合，在知识密集型任务（开放域问答、事实验证等）上显著超越纯生成模型。定义了 RAG-Sequence 和 RAG-Token 两种变体。

**核心贡献**：
- 提出"检索 + 生成"的统一端到端框架，检索器和生成器可联合微调
- 在 Natural Questions、TriviaQA、WebQuestions 等基准上取得了当时最优结果
- 证明了外部知识检索可以显著减少生成模型的幻觉
- 定义了两种检索粒度：RAG-Sequence（每个生成序列检索一次）和 RAG-Token（每生成一个 token 检索一次）

**关键公式**：

RAG-Sequence 的边际化概率：

$$P_{\text{RAG-Seq}}(y \mid x) \approx \sum_{z \in \text{Top-K}} P_\eta(z \mid x) \prod_i P_\theta(y_i \mid x, z, y_{1:i-1})$$

其中 $P_\eta(z \mid x)$ 是检索器给文档 $z$ 的概率，$P_\theta$ 是生成器。

**对后续工作的影响**：奠定了整个 RAG 研究方向，后续几乎所有 RAG 改进工作都以此为基线。

---

#### 2. Karpukhin et al., 2020 — DPR

**标题**：*Dense Passage Retrieval for Open-Domain Question Answering*

**简评**：提出用双塔 BERT 模型替代 BM25 做段落检索。通过对比学习训练 Question Encoder 和 Passage Encoder，使得查询和相关段落在向量空间中靠近。在开放域问答任务上，DPR 首次证明了稠密检索可以全面超越 BM25。

**核心贡献**：
- 证明经过良好训练的稠密检索器可以显著超越 BM25 等稀疏检索方法
- 提出 In-Batch Negatives 训练策略：批次内其他样本的正例作为负例，极大提高训练效率
- 开源了预训练好的 DPR 模型和 Natural Questions 的检索数据集
- 为后续 RAG 系统中的检索组件奠定了标准做法

**训练目标**：

$$\mathcal{L} = -\log \frac{e^{\text{sim}(q, d^+)}}{e^{\text{sim}(q, d^+)} + \sum_{j=1}^{n} e^{\text{sim}(q, d^-_j)}}$$

**工程启示**：DPR 的双塔架构至今仍是 RAG 系统中检索器的主流设计。但需注意——DPR 的训练需要高质量的 (query, positive_passage) 标注数据，这在企业场景中往往是瓶颈。

---

#### 3. Izacard et al., 2022 — Atlas

**标题**：*Atlas: Few-shot Learning with Retrieval Augmented Language Models*

**简评**：将检索增强与少样本学习结合。Atlas 使用 T5 作为生成器、Contriever 作为检索器，在仅有极少标注样本的情况下（如 64 个），通过检索外部知识达到了接近全量微调的效果。证明了 RAG 在低资源场景中的巨大潜力。

**核心贡献**：
- 提出 Attention Distillation：用生成器对检索文档的注意力权重作为信号来训练检索器
- 在 MMLU、Natural Questions、TriviaQA 等基准上，64-shot Atlas 达到了全监督模型的 80%~95% 性能
- 证明了检索增强可以大幅降低模型对参数规模的依赖——小模型 + 好检索器 ≈ 大模型
- 系统分析了检索器和生成器联合训练的策略

**对后续工作的影响**：Atlas 启发了一系列"检索增强 + 少样本"的研究方向，表明 RAG 不仅能减少幻觉，还能提高数据效率。

---

#### 4. Gao et al., 2023 — RAG Survey

**标题**：*Retrieval-Augmented Generation for Large Language Models: A Survey*

**简评**：RAG 领域最全面的综述论文。系统性地整理了 RAG 的发展脉络（Naive RAG → Advanced RAG → Modular RAG），详细分析了检索、增强、生成三个阶段的技术选项和优化策略，并总结了评估方法和未来方向。

**核心贡献**：
- 首次提出 Naive RAG / Advanced RAG / Modular RAG 的三阶段分类框架
- 系统梳理了 RAG 的技术栈：检索源（文本/知识图谱/表格）、检索粒度（token/chunk/doc）、检索时机（一次/每步/自适应）
- 总结了 RAG 评估的四个维度：Context Relevance、Faithfulness、Answer Relevance、Context Recall
- 分析了 RAG 与微调、长上下文窗口等替代方案的互补关系

**为什么必读**：这篇综述是理解 RAG 全貌最高效的入口。如果时间有限只读一篇，读这篇。

---

#### 5. Shi et al., 2023 — REPLUG

**标题**：*REPLUG: Retrieval-Augmented Black-Box Language Models*

**简评**：提出了一种将检索器作为 LLM "即插即用"模块的方法——不需要修改 LLM 内部结构或做微调，只在输入端增加检索到的文档作为上下文。这使得 RAG 可以应用于任何黑盒 LLM（如 GPT-4），极大降低了部署门槛。

**核心贡献**：
- 提出 REPLUG 框架：检索器为任意黑盒 LLM 提供上下文增强
- 提出 REPLUG LSR（LM-Supervised Retrieval）：用 LLM 的困惑度作为信号来训练检索器，无需人工标注
- 在 GPT-3 上验证了不修改模型参数即可通过检索提升性能
- 证明了"更好的检索器"和"更大的 LLM"带来的增益是正交的、可叠加的

**工程启示**：REPLUG 的思路是当前大多数 RAG 应用的实际架构——你不需要（也通常不能）修改 LLM 的参数，只需在 Prompt 中注入检索结果即可。

---

### 补充论文

| 年份 | 论文 | 说明 |
|------|------|------|
| 2023 | Asai et al., *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection* | 模型自主判断是否检索、检索质量、答案质量 |
| 2024 | Yan et al., *Corrective Retrieval Augmented Generation (CRAG)* | 在检索后增加纠正步骤，处理低质量检索结果 |
| 2023 | Es et al., *RAGAS: Automated Evaluation of Retrieval Augmented Generation* | 提出 Faithfulness / Relevancy / Precision / Recall 四维评估框架 |
| 2022 | Reimers & Gurevych, *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks* | 句子级 Embedding 的基础模型 |
| 2024 | Chen et al., *BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity* | 多语言多功能多粒度 Embedding 模型 |

---

## 二、常见误区与FAQ

### Q1：RAG 和 Fine-tuning 是互斥的吗？

**不是，它们是互补的。**

这是最常见的误解之一。RAG 和 Fine-tuning 解决不同层面的问题：

| 维度 | RAG | Fine-tuning |
|------|-----|-------------|
| 解决什么 | **知识注入**——让模型访问外部信息 | **行为适配**——让模型按特定方式回答 |
| 知识更新 | 改知识库即可，秒级生效 | 需要重新训练，小时~天级 |
| 可溯源性 | 天然支持引用标注 | 无法追溯知识来源 |
| 成本 | 向量数据库 + 检索延迟 | 训练计算成本 |

**最佳实践**：RAG + Fine-tuning 结合。先用 RAG 注入知识，再用 Fine-tuning（如 LoRA）让模型更好地理解和组织检索到的内容。例如，一个医疗问答系统可以：
1. RAG：检索最新的医学文献
2. Fine-tuning：让模型学会以专业、审慎的语气回答医学问题

---

### Q2：为什么我的 RAG 系统检索到了相关文档，但生成的答案还是错的？

**这通常是"最后一公里"问题——LLM 没有正确使用检索到的上下文。** 常见原因：

1. **Prompt 设计不佳**：没有明确指示模型"基于以下内容回答"，模型倾向使用自身知识。
   - 解法：在 Prompt 中加入强约束，如"仅根据以下参考文档回答，不要使用你自己的知识"
2. **上下文太长 / 太多噪声**：Top-K 中混入无关文档，稀释了有用信息。
   - 解法：Reranker 精排 + 上下文压缩，只保留最相关的段落
3. **"Lost in the Middle"现象**：LLM 倾向于关注上下文的开头和结尾，忽略中间部分。
   - 解法：将最相关的文档放在 Prompt 的开头或结尾位置
4. **LLM 能力不足**：模型太小，无法在复杂上下文中提取正确信息。
   - 解法：换更强的模型（如从 gpt-4o-mini 升级到 gpt-4o）

---

### Q3：chunk_size 到底设多大？有没有"银弹"？

**没有通用最优值，但有系统化的调优方法。**

chunk_size 的核心权衡：

| chunk_size | 检索精度 | 上下文完整性 | Token 成本 |
|-----------|---------|-------------|-----------|
| 小（128~256 tokens） | 高（检索更精准） | 低（容易丢失上下文） | 低 |
| 中（256~512 tokens） | 中等 | 中等 | 中等 |
| 大（512~1024 tokens） | 低（噪声多） | 高（上下文完整） | 高 |

**学习建议**：

1. **默认起点**：chunk_size=512, overlap=50（对大多数场景是安全的起点）
2. **Parent-Child 模式**：小块检索（128~256）+ 大块送入 LLM（512~1024），兼顾两者
3. **数据驱动调优**：构建一个小规模评估数据集（50~100 个 Q&A 对），用 Recall@K 对比不同 chunk_size 的效果
4. **文档类型适配**：
   - 技术文档 → 按标题/章节切分
   - 对话记录 → 按轮次切分
   - 法律合同 → 按条款切分

---

### Q4：混合检索（BM25 + 向量）真的比纯向量检索好吗？

**在大多数实际场景中，是的。** 但"好多少"取决于查询分布：

- 如果你的查询大多是**语义性的**（"如何提升检索质量"），纯向量检索已经很好，混合检索的提升有限（~2~5%）
- 如果查询中包含**精确匹配需求**（错误码、产品型号、专有名词），混合检索的提升会很显著（~10~20%）
- 如果查询**类型多样**（既有语义查询又有精确查询），混合检索几乎总是更优

**何时不需要混合检索**：
- 只有纯语义查询的场景（如闲聊对话）
- 文档中几乎没有专有名词或技术术语
- 延迟要求极高，无法承受两路检索的开销

**RRF 的 k 值如何设**：默认 k=60 在绝大多数场景下表现良好。k 越大，两个检索器的贡献越平均；k 越小，排名靠前的结果权重越大。

---

### Q5：Reranker 的投入产出比如何？什么时候值得加？

**Reranker 是 RAG 中"性价比最高"的优化之一。**

经验数据（来自多个公开评测和实际项目）：

| 指标 | 无 Reranker | 有 Reranker | 提升 |
|------|-----------|-----------|------|
| NDCG@10 | 0.45~0.55 | 0.55~0.70 | +15~25% |
| 额外延迟 | 0ms | 50~200ms | 可接受 |

**何时值得加**：
- 你的 Top-K 结果质量不稳定（有时准有时不准）
- 你需要从大量候选（Top-20~50）中精选少量（Top-3~5）
- 你的场景对准确性要求高于对延迟的要求

**何时不需要**：
- 你的 Embedding 模型已经非常强（如 GTE-Qwen2-7B），Top-K 质量很高
- 你的候选文档本身就很少（< 10 个）
- 延迟预算极为紧张（< 100ms）

**模型推荐**：
- 轻量级：`cross-encoder/ms-marco-MiniLM-L-6-v2`（推理快，适合在线服务）
- 高精度：`BAAI/bge-reranker-v2-m3`（多语言，精度更高）
- 商业 API：Cohere Rerank（免运维，按调用付费）

---

### Q6：Agentic RAG 相比 Advanced RAG 的实际收益有多大？

**收益显著但成本也高，适合复杂查询场景。**

Agentic RAG 的核心价值是**自适应**——Agent 根据查询复杂度动态选择策略，而不是对所有查询一视同仁。

| 查询类型 | Advanced RAG | Agentic RAG | Agentic 的优势 |
|---------|-------------|-------------|---------------|
| 简单事实查询 | 表现好 | 表现好（可能跳过检索） | 节省检索开销 |
| 多跳推理 | 表现一般 | 分步检索+推理 | 显著提升 |
| 多数据源 | 固定检索所有源 | 按需路由到相关数据源 | 精度+效率提升 |
| 查询模糊 | 固定改写策略 | 迭代改写直到满意 | 鲁棒性提升 |

**代价**：每次查询可能触发 2~5 次 LLM 调用（路由 + 评估 + 重试），成本和延迟都会上升。

**建议**：先把 Advanced RAG 做好（混合检索 + Reranker），在此基础上再考虑 Agentic 化。大多数场景中，Advanced RAG 已经能覆盖 80% 的需求。

---

### Q7：如何处理 RAG 系统中的数据安全和权限控制？

**这是生产环境中最容易被忽视但最重要的问题之一。**

| 层级 | 做法 |
|------|------|
| **索引层** | 每个文档块的元数据中标记权限级别（如 `access_level: "internal"` / `"confidential"`） |
| **检索层** | 在向量搜索时增加元数据过滤条件，只返回用户有权限访问的文档 |
| **生成层** | 在 Prompt 中添加安全约束（"不要泄露敏感信息"）|
| **输出层** | 对 LLM 输出做正则检查（如检测身份证号、手机号等 PII） |
| **审计层** | 记录每次查询的检索文档和生成内容，支持事后审计 |


---

## 三、延伸资源

### 1. LangChain RAG 官方教程

**链接**：https://python.langchain.com/docs/tutorials/rag/

LangChain 官方的 RAG 教程，从最基础的 "stuff" 链到高级的多步 RAG，配有完整解释和示例。适合作为可选补充，用来观察 Document Loader、Text Splitter、Vector Store、Retriever、Chain 如何组成完整流程。

### 2. RAGAS 官方文档

**链接**：https://docs.ragas.io/

RAG 评估框架 RAGAS 的官方文档。详细介绍了 Faithfulness、Answer Relevancy、Context Precision、Context Recall 四个核心指标的计算方法和使用方式。包含与 LangChain、LlamaIndex 的集成指南。

### 3. Jerry Liu: Building Production RAG（演讲）

**来源**：LlamaIndex 创始人的技术演讲系列

介绍了生产级 RAG 系统的架构设计、常见陷阱和最佳实践。涵盖 Chunking 策略优化、检索质量提升、评估体系建立等系统实现话题。对于正在构建或优化 RAG 系统的读者很有参考价值。

---

## 四、本模块知识总图

```
RAG 与应用架构
├── 为什么需要 RAG
│   ├── 幻觉（Hallucination）
│   ├── 时效性缺失
│   └── 私有数据隔离
│
├── 三阶段流程
│   ├── 索引构建: 文档解析 → 分块 → Embedding → 向量库
│   ├── 检索: 查询编码 → 相似度搜索 → Top-K
│   └── 生成: Prompt 组装 → LLM 调用 → 后处理
│
├── Embedding 与向量检索
│   ├── 句子级 Embedding（对比学习训练）
│   ├── 相似度度量（余弦 / 内积 / 欧氏）
│   ├── ANN 算法（IVF / HNSW / PQ）
│   ├── 向量数据库（FAISS / Milvus / Chroma）
│   └── Chunking 策略（固定窗口 / 递归 / 语义 / Parent-Child）
│
├── 高级 RAG 技术
│   ├── 预检索: Query 改写 / HyDE / 多查询
│   ├── 检索: 混合检索（BM25 + 向量 + RRF）
│   ├── 后检索: Reranker / 上下文压缩
│   ├── Agentic RAG: Agent 驱动的自适应检索
│   └── 评估: RAGAS（Faithfulness / Relevancy / Precision / Recall）
│
└── 演进方向
    ├── Self-RAG（自反思检索）
    ├── Corrective RAG（纠正式检索）
    ├── Graph RAG（知识图谱增强）
    └── Multimodal RAG（多模态检索）
```
