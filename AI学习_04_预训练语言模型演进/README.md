# AI学习：04 预训练语言模型演进

## 本模块导读

**学什么**：从词向量到大语言模型的完整演进路线——如何让机器"理解"语言。从 Word2Vec 的静态词向量，到 ELMo 的上下文表示，到 BERT 和 GPT 开创的预训练+微调范式。

**为什么学**：理解这条演进路线，才能理解为什么 GPT 选择了 Decoder-only 架构、为什么预训练如此重要、为什么大模型能"涌现"出强大能力。

**学完能做什么**：
- 理解词向量的演进：One-Hot → Word2Vec → 上下文相关表示
- 理解自监督学习的两大范式：MLM（掩码语言模型）和 CLM（因果语言模型）
- 理解 BERT 和 GPT 分别代表的技术路线及其差异
- 理解预训练→微调的迁移学习范式
- 理解 Encoder-only / Decoder-only / Encoder-Decoder 三大架构的适用场景

**前置知识**：[03 注意力机制与 Transformer](../AI学习_03_注意力机制与Transformer/README.md)

**预估学习时间**：8~10 小时

## 目录

| 序号 | 文件 | 内容 |
|------|------|------|
| 1 | [词表示的演进](1_词表示的演进.md) | One-Hot → Word2Vec → GloVe → ELMo |
| 2 | [自监督预训练范式](2_自监督预训练范式.md) | MLM（BERT）与 CLM（GPT）、预训练目标的设计 |
| 3 | [BERT与GPT](3_BERT与GPT.md) | 两大路线的架构、训练、应用对比 |
| 4 | [微调与迁移学习](4_微调与迁移学习.md) | Fine-tuning、Prompt Tuning、三大架构路线对比 |
| - | [论文与FAQ](论文与FAQ.md) | 关键论文、常见误区、延伸资源 |

## 下一步

学完本模块后进入 [05_GPT系列](../AI学习_05_GPT系列/README.md)，深入了解 Decoder-only 路线如何从 GPT-1 演进到 GPT-4。
