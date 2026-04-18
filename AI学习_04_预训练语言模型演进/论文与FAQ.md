# 论文与FAQ

> 本文件用于论文脉络、误区澄清与延伸阅读，不替代主章节正文。

> 本文件汇总"预训练语言模型演进"模块的关键论文、常见误区与延伸资源。

---

## 一、关键论文

### 时间线总览

| 年份 | 作者 | 模型/方法 | 核心关键词 |
|------|------|-----------|------------|
| 2013 | Mikolov et al. | Word2Vec | 静态词向量、Skip-gram、CBOW |
| 2014 | Pennington et al. | GloVe | 全局共现矩阵、词向量 |
| 2018.02 | Peters et al. | ELMo | 上下文词向量、双向 LSTM |
| 2018.06 | Radford et al. | GPT-1 | Decoder-only 预训练、CLM |
| 2018.10 | Devlin et al. | BERT | Encoder-only 预训练、MLM |
| 2019.07 | Liu et al. | RoBERTa | BERT 训练策略优化 |
| 2019.10 | Raffel et al. | T5 | Encoder-Decoder、Text-to-Text |
| 2021 | Hu et al. | LoRA | 低秩适配、参数高效微调 |

### 逐篇简评

---

#### 1. Mikolov et al., 2013 — Word2Vec

**标题**：*Efficient Estimation of Word Representations in Vector Space*

**简评**：提出 Skip-gram 和 CBOW 两种浅层神经网络结构，首次以大规模、高效的方式训练出稠密词向量。证明了词向量的线性代数关系（如 king - man + woman ≈ queen），开启了"分布式表示"时代。

**核心贡献**：
- 将词表示从稀疏的 One-Hot 推进到稠密的低维向量
- 提出负采样（Negative Sampling）和层级 Softmax，使大规模训练成为可能
- 词向量的类比关系验证了语义信息可以被几何结构编码

---

#### 2. Pennington et al., 2014 — GloVe

**标题**：*GloVe: Global Vectors for Word Representation*

**简评**：与 Word2Vec 的"局部上下文窗口"思路不同，GloVe 显式利用全局词-词共现统计矩阵进行分解，将共现概率的对数比作为训练目标。在多项任务上取得了与 Word2Vec 相当或更优的效果。

**核心贡献**：
- 将基于计数的方法（如 LSA）和基于预测的方法（如 Word2Vec）统一到同一框架
- 证明全局统计信息对词向量质量有重要价值
- 提供了预训练词向量资源，至今仍被广泛使用

---

#### 3. Peters et al., 2018 — ELMo

**标题**：*Deep contextualized word representations*

**简评**：首次将"上下文相关"词向量引入 NLP。使用双向 LSTM 语言模型，为同一个词在不同句子中生成不同的表示。例如 "bank" 在"river bank"和"bank account"中得到不同向量，从根本上解决了静态词向量的一词多义问题。

**核心贡献**：
- 提出上下文词向量（contextualized word embeddings）概念
- 证明深层语言模型的不同层捕获了不同层次的语言信息（底层偏语法，高层偏语义）
- 开创了"预训练表示 + 下游任务"的两阶段范式雏形

---

#### 4. Radford et al., 2018 — GPT-1

**标题**：*Improving Language Understanding by Generative Pre-Training*

**简评**：首次将 Transformer Decoder 用于大规模无监督语言模型预训练，再通过有监督微调迁移到下游任务。证明了"先预训练、后微调"这一范式在多项 NLP 基准上的有效性。

**核心贡献**：
- 确立了 Decoder-only + 因果语言建模（CLM）的预训练路线
- 证明无监督预训练可以为各类下游任务提供强大的初始化
- 为后续 GPT-2、GPT-3 以至 GPT-4 的 Scaling 路线奠定了基础

---

#### 5. Devlin et al., 2018 — BERT

**标题**：*BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*

**简评**：提出掩码语言模型（MLM）和下一句预测（NSP）两个自监督目标，使用 Transformer Encoder 进行双向预训练。在 11 项 NLP 任务上刷新了当时的最佳记录，成为 NLP 领域的里程碑。

**核心贡献**：
- 提出 MLM 训练目标，使 Transformer 可以真正做到双向上下文建模
- 确立了 Encoder-only 预训练 + 下游微调的范式
- 带动了 NLP 领域从"特征工程"到"预训练微调"的范式转变

---

#### 6. Liu et al., 2019 — RoBERTa

**标题**：*RoBERTa: A Robustly Optimized BERT Pretraining Approach*

**简评**：在不改变 BERT 架构的前提下，通过系统性地调整训练策略（去掉 NSP、更大 batch size、更多数据、更长训练时间、动态 masking），大幅提升了 BERT 的性能。证明了"BERT 本身没有训练够"。

**核心贡献**：
- 证明了训练策略和数据规模对预训练效果的决定性影响
- 去除 NSP 目标后性能不降反升，澄清了 NSP 的实际价值
- 为后续预训练工作建立了更扎实的实验基线

---

#### 7. Raffel et al., 2019 — T5

**标题**：*Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*

**简评**：提出"Text-to-Text"统一框架：所有 NLP 任务（分类、翻译、摘要、问答……）都转换为"文本输入 → 文本输出"的格式，使用 Encoder-Decoder 架构统一处理。同时系统性地对比了预训练目标、模型规模、数据集等因素的影响。

**核心贡献**：
- 提出统一的 Text-to-Text 范式，消除了任务特定的输出层设计
- 发布了 C4（Colossal Clean Crawled Corpus）大规模预训练数据集
- 提供了迄今最全面的预训练消融实验，为社区提供了宝贵的工程经验

---

#### 8. Hu et al., 2021 — LoRA

**标题**：*LoRA: Low-Rank Adaptation of Large Language Models*

**简评**：当模型参数量达到数十亿乃至千亿级别时，全参数微调变得极其昂贵。LoRA 提出在冻结原始权重的同时，仅训练注入到每层的低秩分解矩阵（A 和 B），大幅降低可训练参数量，同时保持接近全参数微调的性能。

**核心贡献**：
- 提出低秩适配（Low-Rank Adaptation）方法，将微调参数量降低数个数量级
- 训练完成后，低秩矩阵可以合并回原始权重，推理时无额外延迟
- 开创了"参数高效微调"（PEFT）方向的代表性方法，被广泛应用于大模型定制

---

### 补充论文

| 年份 | 论文 | 说明 |
|------|------|------|
| 2019 | Yang et al., *XLNet: Generalized Autoregressive Pretraining for Language Understanding* | 排列语言模型，融合自回归与双向上下文的优势 |
| 2020 | Brown et al., *Language Models are Few-Shot Learners*（GPT-3） | 175B 参数，展示了 In-Context Learning 和涌现能力 |
| 2022 | Wei et al., *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models* | 思维链提示，大模型推理能力的关键解锁方式 |

---

## 二、常见误区与FAQ

### Q1：BERT 和 GPT 哪个更好？

**没有"更好"，只有"更适合"。**

这是两条不同的技术路线，解决不同类型的问题：

| 维度 | BERT（Encoder-only） | GPT（Decoder-only） |
|------|---------------------|---------------------|
| 注意力方向 | 双向（看到完整上下文） | 单向（只看到左侧） |
| 擅长任务 | 理解类：分类、NER、问答抽取 | 生成类：续写、对话、翻译 |
| 代表应用 | 搜索排序、情感分析 | ChatGPT、代码生成 |

简单说：**需要"理解"用 BERT 系，需要"生成"用 GPT 系。** 在当前大模型时代，GPT 系的 Decoder-only 架构因为在 Scaling 方面表现更好而成为主流，但这不意味着 BERT 的思路过时了——在中小规模的理解任务上，BERT 系模型仍然是高效且实用的选择。

---

### Q2：为什么不把 BERT 做大到 GPT-3 的规模？

有人尝试过，但效果不理想，原因有三：

1. **训练效率低**：MLM 每次只预测被掩码的 15% 的 token，而 CLM 每个 token 都参与预测，信号利用率差异巨大。要达到相同的"有效训练量"，BERT 需要更多的计算。
2. **Scaling 行为不同**：实验表明，Decoder-only 架构在参数量持续增大时，性能提升更平稳、更可预测（符合 Scaling Law）。Encoder-only 架构在超大规模下的收益递减更明显。
3. **应用场景限制**：BERT 的双向注意力天然不适合自回归生成，而大模型时代最核心的能力——In-Context Learning、指令跟随、思维链推理——都依赖于生成。

因此，学术界和工业界在"做大"这条路上选择了 GPT 路线。

---

### Q3：Word2Vec 还有用吗？

**有用，但使用场景发生了变化。**

Word2Vec 作为"预训练表示"的工具已经被 BERT/GPT 类模型取代，但它的价值在于：

- **教学价值**：Word2Vec 是理解分布式表示、嵌入空间、自监督学习的最佳入门素材。
- **轻量级场景**：在计算资源极度受限、数据量较小、或只需要词级别特征的场景中（如关键词聚类、简单的文本相似度计算），Word2Vec 仍然实用。
- **概念的延续**：Word2Vec 的 Embedding 思想被所有后续模型继承。BERT 和 GPT 的第一层就是一个 Embedding 层，只不过后面接了 Transformer 来生成上下文相关的表示。

一句话总结：**Word2Vec 不再是最优工具，但它奠定的思想是所有后续工作的基石。**

---

### Q4：微调和 Prompt 该选哪个？

取决于你的**模型规模、数据量和任务需求**：

| 条件 | 推荐方法 | 原因 |
|------|----------|------|
| 中小模型（< 1B）+ 有标注数据 | 微调（Fine-tuning） | 小模型的 In-Context Learning 能力弱，微调效果更好 |
| 大模型（> 10B）+ 少量/无标注数据 | Prompt / In-Context Learning | 大模型本身能力强，好的 Prompt 就能激发性能 |
| 大模型 + 有中等标注数据 | LoRA 等参数高效微调 | 在不破坏通用能力的前提下适配特定任务 |
| 需要严格控制输出格式 | 微调 | Prompt 对输出格式的控制不够稳定 |

**实践建议**：先试 Prompt（成本最低），效果不够再上 LoRA，最后才考虑全参数微调。这是当前工程实践中的主流策略。

---

### Q5：预训练需要多少数据？

**没有固定答案，但有经验规律。**

- **Word2Vec / GloVe**：数亿到数十亿词（如 Google News 约 1000 亿词、Wikipedia 约 30 亿词）即可训练出质量不错的词向量。
- **BERT-base（1.1 亿参数）**：原始论文使用约 33 亿词（BooksCorpus + English Wikipedia）。
- **GPT-3（1750 亿参数）**：使用约 3000 亿 token 的混合数据集。
- **经验法则（Chinchilla Scaling Law, 2022）**：最优的训练 token 数量约等于模型参数量的 20 倍。即 1B 参数的模型应训练约 20B token。

**关键点**：数据量不是唯一因素。数据**质量**（去重、去噪、领域多样性）和**训练策略**（学习率、batch size、训练步数）同样重要——RoBERTa 的成功就是最好的例证。

---

### Q6：预训练和"从头训练"有什么区别？用预训练模型微调和直接训一个专用模型比，谁好？

**预训练 + 微调几乎总是更好。** 原因在于：

1. **知识迁移**：预训练阶段在海量文本上学到的语言知识（语法结构、世界知识、推理模式）可以迁移到下游任务。从头训练则需要从零学习这些知识。
2. **数据效率**：微调只需要少量标注数据就能达到很好的效果。从头训练一个相同性能的专用模型，可能需要多得多的标注数据。
3. **收敛速度**：在一个好的预训练初始化基础上微调，收敛速度远快于随机初始化。

唯一的例外是当你的目标领域与预训练数据差异极大（如特定的科学符号系统），且你拥有大量领域数据时，从头预训练一个领域模型可能更合适。

---

## 三、延伸资源

1. **Hugging Face NLP Course**
   https://huggingface.co/learn/nlp-course
   免费的实践教程，覆盖 Tokenizer、预训练模型使用、微调全流程。配有 Colab 代码，适合动手学习。

2. **The Illustrated BERT, ELMo, and co.**（Jay Alammar）
   https://jalammar.github.io/illustrated-bert/
   以图解方式解释 ELMo、BERT、GPT 等模型的核心机制，是理解这些模型最直观的参考资料。

3. **Stanford CS224N: Natural Language Processing with Deep Learning**
   https://web.stanford.edu/class/cs224n/
   斯坦福经典 NLP 课程，从词向量讲到大语言模型，课程视频和讲义公开。适合系统性学习。
