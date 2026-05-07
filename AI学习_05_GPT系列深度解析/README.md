# AI学习：05 GPT 系列深度解析

> 读本模块时，先抓住 GPT 主线如何形成：Decoder-only 负责生成底座，Scaling Laws 解释规模效应，In-Context Learning 解释上下文学习，对齐技术让模型更适合对话和任务执行。具体模型榜单和阶段性快照不需要背。

## 本模块导读

**学什么**：GPT 系列从 GPT-1 到 GPT-4 及后续模型的完整演进，以及驱动这一演进的核心理论——Scaling Laws、In-Context Learning、涌现能力、对齐技术（RLHF/DPO）。

**为什么学**：GPT 系列是大语言模型的主线剧情。理解这条路线，就理解了"为什么模型越来越大"、"为什么大模型突然变强"、"为什么需要 RLHF"等核心问题。

**读完能看懂的主线**：
- 理解 GPT-1→2→3→4 每一代的关键创新
- 理解 Scaling Laws 的含义和实际影响
- 理解 In-Context Learning 和涌现能力
- 理解 RLHF、DPO 等对齐技术的原理和流程
- 了解 GPT-4 之后的重要架构变化（截至 2026.4）

**前置知识**：[04 预训练语言模型演进](../AI学习_04_预训练语言模型演进/README.md)

**预估阅读时间**：10~12 小时

## 目录

| 序号 | 文件 | 内容 |
|------|------|------|
| 1 | [GPT系列架构演进](1_GPT系列架构演进.md) | GPT-1/2/3/4 每代关键创新、架构变化 |
| 2 | [Scaling Laws与涌现](2_ScalingLaws与涌现.md) | Kaplan/Chinchilla Scaling Laws、In-Context Learning、涌现能力 |
| 3 | [对齐技术](3_对齐技术.md) | SFT、RLHF、DPO、Constitutional AI |
| 4 | [GPT4之后的演进](4_GPT4之后的演进.md) | 截至 2026.4 的重要进展 |
| - | [论文与FAQ](论文与FAQ.md) | 关键论文、延伸资源 |

## 读完能解释

1. GPT-1、GPT-2、GPT-3、GPT-4 的关键变化分别是什么？
2. Scaling Laws 说明了什么？Chinchilla 为什么会改变“只堆参数”的直觉？
3. In-Context Learning 和涌现能力为什么会在大模型时代变得重要？
4. SFT、RLHF、DPO 各自解决什么问题？为什么预训练模型还需要对齐？
5. 对 GPT-4 之后的模型演进，哪些是稳定原理，哪些只是截至 2026 年 4 月的阶段性快照？

## 时效性提示

> 本模块中，**第 4 章（GPT-4 之后的演进）变化最快**——具体模型排名和推荐可能在数月内过时。前 3 章涵盖的核心原理（Transformer Decoder 架构、Scaling Laws 幂律关系、RLHF/DPO 对齐框架）则相对稳定，预计 2~3 年内不会有根本性变化。建议优先吃透前 3 章的原理，再将第 4 章作为"当前快照"来阅读。
