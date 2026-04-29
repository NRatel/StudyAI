# AI学习：04 预训练语言模型演进

> 读本模块时，先抓住语言模型为什么从“词向量”走向“预训练大模型”。重点是路线关系和范式变化，不需要复现 Word2Vec、BERT 或 GPT 的训练。

## 本模块导读

**学什么**：从词向量到大语言模型的完整演进路线——如何让机器"理解"语言。从 Word2Vec 的静态词向量，到 ELMo 的上下文表示，到 BERT 和 GPT 开创的预训练+微调范式。

**为什么学**：理解这条演进路线，才能理解为什么预训练如此重要、为什么 BERT 和 GPT 会走向不同路线，以及为什么生成式大模型后来成为通用 AI 的主线。

**读完能看懂的主线**：
- 理解词向量的演进：One-Hot → Word2Vec → 上下文相关表示
- 理解自监督学习的两大范式：MLM（掩码语言模型）和 CLM（因果语言模型）
- 理解 BERT 和 GPT 分别代表的技术路线及其差异
- 理解预训练→微调的迁移学习范式
- 理解 Encoder-only / Decoder-only / Encoder-Decoder 三大架构的适用场景

**前置知识**：[03 注意力机制与 Transformer](../AI学习_03_注意力机制与Transformer/README.md)

**预估阅读时间**：8~10 小时

## 目录

| 序号 | 文件 | 内容 |
|------|------|------|
| 1 | [词表示的演进](1_词表示的演进.md) | One-Hot → Word2Vec → GloVe → ELMo |
| 2 | [自监督预训练范式](2_自监督预训练范式.md) | MLM（BERT）与 CLM（GPT）、预训练目标的设计 |
| 3 | [BERT与GPT](3_BERT与GPT.md) | 两大路线的核心差异和适用场景 |
| 4 | [微调与迁移学习](4_微调与迁移学习.md) | Fine-tuning、Prompt Tuning、三大架构路线对比 |
| - | [论文与FAQ](论文与FAQ.md) | 关键论文、常见误区、延伸资源 |

## 读完能解释

1. One-Hot、Word2Vec、ELMo、BERT/GPT 的表示能力为什么一代比一代强？
2. MLM 和 CLM 的训练目标有什么区别？它们分别适合什么任务？
3. BERT 的 Encoder-only 路线和 GPT 的 Decoder-only 路线为什么会分化？
4. 预训练、微调、Prompt Tuning/LoRA 分别改变了模型学习的哪个环节？

## 下一步

读完本模块后进入 [05 GPT 系列深度解析](../AI学习_05_GPT系列深度解析/README.md)，继续了解 Decoder-only 路线如何从 GPT-1 演进到 GPT-4。
