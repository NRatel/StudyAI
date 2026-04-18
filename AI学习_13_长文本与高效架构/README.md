# AI学习：13 长文本与高效架构

> **内容时效**：截至 2026 年 4 月。本模块涉及快速演进的前沿领域，部分内容可能在 6-12 个月内发生变化。

## 本模块导读

**学什么**：当序列长度从 512 增长到 128K 甚至 1M+ 时，标准 Transformer 的 O(n^2) 注意力成为不可逾越的计算与内存瓶颈。本模块系统梳理三大前沿方向——长文本技术（位置编码外推、序列并行、压缩记忆）、MoE 稀疏架构（用更少的计算调动更多的参数）、高效 Transformer 变体（线性注意力、状态空间模型）——帮你理解"如何让大模型既长又快又强"。

**为什么学**：2024-2026 年的大模型竞争焦点已从"参数规模"转向"长上下文 + 高效推理"。GPT-4 Turbo 128K、Claude 200K/1M、Gemini 1M+、Kimi 的长文本能力均依赖本模块讨论的技术。MoE 架构则是 Mixtral、DeepSeek-V2/V3、Qwen-MoE 等模型的核心设计。理解这些技术，才能跟上当下模型架构的演进节奏。

**学完能做什么**：
- 理解为什么标准注意力在长序列上不可行，以及 RoPE 扩展（PI/NTK-Aware/YaRN）如何让模型外推到超长上下文
- 理解 Ring Attention 如何实现百万级序列长度的分布式计算
- 理解 MoE（Mixture of Experts）的路由机制、稀疏激活原理，以及 Switch Transformer / Mixtral / DeepSeekMoE 的设计差异
- 理解线性注意力、稀疏注意力、状态空间模型（SSM/Mamba）与标准 Transformer 的本质区别
- 能对不同场景（长文档理解、高吞吐推理、大参数小计算）选择合适的架构方案

**前置知识**：[03 注意力机制与 Transformer](../AI学习_03_注意力机制与Transformer/README.md)（特别是 Self-Attention、Multi-Head Attention、RoPE 位置编码、FFN 的作用）

**预估学习时间**：4~6 小时

**模块定位**：前沿篇 ★★★——了解主要原理和设计思想即可，不要求推导每一个数学细节。

## 目录

| 序号 | 文件 | 内容 |
|------|------|------|
| 1 | [长文本技术](1_长文本技术.md) | 为什么长文本难（O(n^2) 瓶颈、位置编码外推问题）、RoPE 扩展方案（PI / NTK-Aware / YaRN）、Ring Attention（序列并行）、Infini-Attention（压缩记忆） |
| 2 | [MoE架构](2_MoE架构.md) | Mixture of Experts 原理（路由机制、稀疏激活）、Switch Transformer、Mixtral、DeepSeekMoE、负载均衡、MoE 的优缺点 |
| 3 | [高效Transformer](3_高效Transformer.md) | 线性注意力（Performer）、稀疏注意力（Longformer / BigBird）、状态空间模型 SSM（S4 / Mamba / Mamba-2）、RWKV、与标准 Transformer 的全面对比 |
| - | [论文与FAQ](论文与FAQ.md) | 关键论文（Reformer / Longformer / Switch / Mamba / Infini-Attention 等）、常见误区、延伸资源 |

## 学习建议

1. **先读文件 1**：长文本技术是最直观的切入点——你已经理解了标准注意力的 O(n^2) 瓶颈，自然会问"怎么办"
2. **再读文件 2**：MoE 架构解决的是另一个维度的效率问题——"参数量大但计算量小"
3. **最后读文件 3**：高效 Transformer 变体是对标准注意力的根本性替代方案，理解了前两部分再看对比会更清晰
4. **论文与 FAQ** 随时查阅，用于填补细节和解答困惑

## 与其他模块的关系

```
03_Transformer ──→ 本模块（13）
    │
    ├── 1_长文本技术：基于 RoPE（03_4_位置编码）的外推扩展
    ├── 2_MoE架构：将 Transformer 的 FFN 替换为稀疏专家层
    └── 3_高效Transformer：对 03 中标准注意力的根本性改造
```
