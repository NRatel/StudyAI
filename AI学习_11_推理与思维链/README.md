# AI学习：11 推理与思维链

> **内容时效**：截至 2026 年 4 月。推理（Reasoning）领域自 2022 年 Chain-of-Thought 论文以来进展极快，尤其是 2024-2026 年 OpenAI o 系列和开源推理模型的爆发。本模块标注了各技术的时间节点，请结合阅读时的最新进展理解。

## 本模块导读

**学什么**：大语言模型如何"思考"——从 Chain-of-Thought 提示到 Tree-of-Thought 搜索，从提示工程到专门训练的推理模型（o1/o3/o4-mini），再到 Test-time Compute Scaling 的理论基础。理解模型推理能力的来源、增强方法和工程实践。

**为什么学**：推理能力是 2024-2026 年大模型最核心的进化方向。GPT-4 之后，行业共识从"让模型更大"转向"让模型更会想"。理解推理技术，才能回答这些关键问题：
- 为什么一句 "Let's think step by step" 就能提升数学题的准确率？
- 为什么 OpenAI o1 在竞赛数学中超越人类选手？
- 为什么推理时间更长反而效果更好？
- 如何在自己的应用中利用推理能力？

**学完能做什么**：
- 掌握 Chain-of-Thought 及其变体（Zero-shot CoT、Self-Consistency）的原理与使用方法
- 理解 Tree-of-Thought、Graph-of-Thought、Reflexion 等高级推理方法
- 理解推理模型（o1/o3/o4-mini、DeepSeek-R1、QwQ）的训练范式与能力边界
- 理解 Test-time Compute Scaling 的理论基础：为什么推理时花更多计算能换来更好结果
- 在实际项目中合理选择推理策略：什么时候用 CoT 提示，什么时候用推理模型

**前置知识**：[05 GPT系列深度解析](../AI学习_05_GPT系列深度解析/README.md)（需要理解 LLM 的自回归生成、Scaling Laws、RLHF 对齐流程）

**预估学习时间**：4~6 小时

**难度标记**：★★★（前沿篇）

## 目录

| 序号 | 文件 | 内容 |
|------|------|------|
| 1 | [思维链提示](1_思维链提示.md) | Chain-of-Thought (CoT) 原始论文、Zero-shot CoT、Self-Consistency（多次采样投票） |
| 2 | [高级推理方法](2_高级推理方法.md) | Tree-of-Thought、Graph-of-Thought、Self-Reflection/Reflexion、ReAct 回顾 |
| 3 | [推理模型与TTC](3_推理模型与TTC.md) | OpenAI o1/o3/o4-mini、Test-time Compute Scaling、GRPO/PRM、DeepSeek-R1、QwQ |
| - | [论文与FAQ](论文与FAQ.md) | 关键论文、常见误区、延伸资源 |

## 模块知识图谱

```
推理与思维链（本模块）
├── 思维链提示技术
│   ├── Chain-of-Thought (CoT) — Wei 2022
│   ├── Zero-shot CoT — "Let's think step by step" — Kojima 2022
│   └── Self-Consistency — 多路径采样 + 投票 — Wang 2023
├── 高级推理方法
│   ├── Tree-of-Thought (ToT) — 搜索式推理 — Yao 2023
│   ├── Graph-of-Thought (GoT) — 图结构推理 — Besta 2023
│   ├── Reflexion — 自我反思 + 记忆 — Shinn 2023
│   └── ReAct — 推理 + 行动（→ 模块12 Agent 衔接）
├── 推理模型
│   ├── OpenAI o1 (2024.9) → o3 (2025.4) → o4-mini (2025.4)
│   ├── DeepSeek-R1 (2025.1) — 开源推理模型
│   └── QwQ (2024.11) / Qwen3 (2025.4) — 通义推理模型
└── 理论基础
    ├── Test-time Compute Scaling — 推理时计算换质量
    ├── Process Reward Model (PRM) — 过程奖励
    └── GRPO — 群组相对策略优化
```

## 学习路线建议

```
1_思维链提示  →  2_高级推理方法  →  3_推理模型与TTC
   (基础)          (进阶)            (前沿)

建议顺序学习。第 1 节是基础，理解 CoT 的原理后，
第 2 节的 ToT/GoT/Reflexion 自然是 CoT 的扩展，
第 3 节则跳到专门训练的推理模型，是另一个维度的进化。
```

## 时效性说明

推理领域处于爆发期，以下内容可能在 6~12 个月内发生显著变化：
- **推理模型迭代**：OpenAI o 系列、DeepSeek-R 系列、Qwen 推理系列均在快速迭代，模型能力和 API 接口可能发生重大变更
- **Test-time Compute**：TTC Scaling 的理论和最优实践仍在研究中，最佳推理预算分配策略尚未定论
- **开源追赶**：DeepSeek-R1 在 2025 年初引爆了开源推理模型浪潮，预计后续会有更多高质量开源推理模型
- **Hybrid 趋势**（2025-2026）：如 Qwen3 的"思考/非思考"混合模式，推理与非推理的界限正在模糊化

建议学习时结合 OpenAI、DeepSeek、Qwen 的最新文档与博客。
