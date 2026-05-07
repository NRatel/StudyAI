# 论文与FAQ

> 本文件用于论文脉络、误区澄清与延伸阅读，不替代主章节正文。

> 本文件汇总"Agent 与工具使用"模块的关键论文与延伸资源。
>
> **内容时效**：截至 2026 年 4 月

---

## 一、关键论文

### 时间线总览

| 年份 | 作者 | 论文/方法 | 核心关键词 |
|------|------|-----------|------------|
| 2022.10 | Yao et al. | ReAct | 推理+行动交替、LLM Agent |
| 2022.01 | Wei et al. | Chain-of-Thought | 思维链推理、Few-shot |
| 2023.02 | Schick et al. | Toolformer | 自学习工具调用、API 使用 |
| 2023.03 | Shinn et al. | Reflexion | 语言反思、自我改进 |
| 2023.08 | Wang et al. | Agent Survey | Agent 架构综述、规划/记忆/工具 |
| 2023.10 | Packer et al. | MemGPT | 虚拟内存管理、长上下文 |
| 2024.11 | Anthropic | MCP | Model Context Protocol、工具协议标准 |
| 2025.03 | OpenAI | Agents SDK | Agent 编排、Handoff、Guardrails |

### 逐篇简评

---

#### 1. Yao et al., 2022 — ReAct

**标题**：*ReAct: Synergizing Reasoning and Acting in Language Models*

**简评**：Agent 领域最具影响力的论文之一。提出让 LLM 在生成行动（Action）之前先生成推理（Thought），两者交替进行的范式。在知识密集型问答（HotpotQA）和决策型任务（ALFWorld）上均显著优于纯推理（CoT）和纯行动（Act-only）的方案。

**核心贡献**：
- 提出 Thought-Action-Observation 的三元组循环，成为后续几乎所有 Agent 框架的基础
- 证明了推理和行动的结合产生协同效应——推理使行动更有目的性，行动结果又修正推理
- 通过简单的 prompt 工程（无需微调）即可实现，极大降低了 Agent 构建门槛
- 为 Agent 的可解释性提供了天然支持——Thought 过程可见、可审计

**关键实验结果**：在 HotpotQA 上，ReAct 比 CoT-only 高 6%（减少了幻觉），比 Act-only 高 10%（减少了盲目行动）。

---

#### 2. Wei et al., 2022 — Chain-of-Thought Prompting

**标题**：*Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*

**简评**：虽然 CoT 本身不直接聚焦 Agent，但它是 Agent 推理能力的理论基石。证明了在 prompt 中加入中间推理步骤（"let's think step by step"）可以显著提升 LLM 在复杂推理任务上的表现。ReAct 的 Thought 机制正是 CoT 在 Agent 场景中的应用。

**核心贡献**：
- 揭示了大模型（>100B 参数）具备通过中间推理步骤解决复杂问题的"涌现"能力
- 提出 Few-shot CoT 和 Zero-shot CoT 两种范式
- 在数学推理（GSM8K）、常识推理、符号推理等任务上取得显著提升
- 启发了后续 Tree-of-Thought、Graph-of-Thought 等扩展工作

**与 Agent 的关系**：CoT 是 Agent 推理层的核心技术。没有 CoT，LLM 在复杂任务上的推理能力不足以支撑有效的规划和决策。

---

#### 3. Schick et al., 2023 — Toolformer

**标题**：*Toolformer: Language Models Can Teach Themselves to Use Tools*

**简评**：提出了一种让 LLM "自学" 何时和如何调用外部工具（计算器、搜索引擎、翻译 API 等）的方法。不同于依赖人工设计 prompt 的方案，Toolformer 通过自监督方式在训练数据中自动插入 API 调用标注，然后微调模型。

**核心贡献**：
- 提出自监督的工具调用学习方法：模型自己决定在文本的哪个位置插入 API 调用
- 训练流程：(1) 生成候选 API 调用 → (2) 执行 API → (3) 保留降低困惑度的调用 → (4) 微调
- 证明了相对较小的模型（GPT-J 6B）通过工具使用可以超越更大的模型（GPT-3 175B）
- 思想上预示了 Function Calling 等原生工具调用能力的方向

**局限性**：需要微调，不如后来的 prompt-based 方案（如 Function Calling）灵活。但其"模型自主学习使用工具"的思路仍有深远影响。

---

#### 4. Shinn et al., 2023 — Reflexion

**标题**：*Reflexion: Language Agents with Verbal Reinforcement Learning*

**简评**：提出用语言形式的反思（verbal reflection）替代传统强化学习中的标量奖励信号，让 Agent 从失败中学习。Agent 在任务失败后会生成一段自然语言的反思总结（"我在第 3 步犯了什么错？下次应该怎么做？"），并将其存入记忆供后续尝试参考。

**核心贡献**：
- 提出"语言强化学习"概念：用自然语言反思替代数值奖励，更具可解释性
- 在编程（HumanEval）、决策（ALFWorld）、推理（HotpotQA）三类任务上均有提升
- 反思记忆的引入让 Agent 在多次尝试中持续改进，而非每次从零开始
- 为 Agent 的自我改进能力提供了一种轻量级方案（无需梯度更新）

**与实际应用的关系**：Reflexion 的思路在 2024-2025 年被广泛应用于编程 Agent（如 Devin 风格的编码 Agent），Agent 在代码执行失败后反思错误并修正。

---

#### 5. Wang et al., 2023 — A Survey on LLM-based Autonomous Agents

**标题**：*A Survey on Large Language Model based Autonomous Agents*

**简评**：Agent 领域最全面的综述论文之一。系统性地将 LLM-based Agent 架构分解为四个模块——Profile（人设）、Memory（记忆）、Planning（规划）、Action（行动），并分别综述了各模块的研究进展。

**核心贡献**：
- 提出清晰的 Agent 架构分类框架：Profile + Memory + Planning + Action
- Memory 部分详细对比了短期记忆、长期记忆的各种实现方案
- Planning 部分梳理了无反馈规划（CoT/ToT/GoT）和有反馈规划（ReAct/Reflexion）的区别
- Action 部分覆盖了工具使用、代码执行、环境交互等行动类型
- 整理了单 Agent 和多 Agent 的应用场景

**阅读建议**：作为 Agent 领域的入门综述，建议至少通读一遍，建立全局认知框架。

---

### 补充论文

| 年份 | 论文 | 说明 |
|------|------|------|
| 2023.05 | Yao et al., *Tree of Thoughts* | 将 CoT 扩展为树结构搜索，支持回溯，提升复杂推理能力 |
| 2023.10 | Packer et al., *MemGPT* | 将操作系统虚拟内存概念引入 LLM，自动管理上下文窗口 |
| 2024.03 | Anthropic, *Claude 3 Tool Use* | 原生 Tool Use 能力，支持并行调用，行业标杆实现 |
| 2024.10 | Anthropic, *Computer Use* | 模型直接操作桌面 GUI（点击/输入/截图），Agent 能力边界扩展 |
| 2024.11 | Anthropic, *Model Context Protocol* | 开放协议标准，统一 Agent-工具通信接口 |
| 2025.01 | DeepSeek, *DeepSeek-R1* | 强化学习训练的推理模型，Agent 规划能力显著提升 |
| 2025.03 | OpenAI, *Agents SDK* | 官方 Agent 编排框架，Handoff + Guardrails + Tracing |

---

## 二、FAQ

### Q1：Agent 和 RAG 有什么区别？

**RAG 是 Agent 的一个子能力，而非替代关系。**

| 维度 | RAG | Agent |
|------|-----|-------|
| **核心能力** | 检索外部知识辅助回答 | 自主规划、多步执行、工具调用 |
| **交互模式** | 单次：查询 → 检索 → 回答 | 多次循环：推理 → 行动 → 观察 → ... |
| **工具使用** | 通常只有向量检索 | 可调用任意工具（搜索/数据库/API/代码执行） |
| **适用场景** | 知识密集型问答 | 复杂任务执行 |

**关系**：一个 Agent 可以将 RAG 作为它的工具之一。当 Agent 需要查找知识时调用 RAG，需要执行代码时调用代码工具，需要发邮件时调用邮件 API。Agent 是更上层的抽象。

---

### Q2：Agent 系统的最大瓶颈是什么？

**可靠性（Reliability），而非能力。**

2025-2026 年的 LLM 在能力上已经足够强，Agent 失败的主要原因不是"模型不够聪明"，而是：

1. **规划失误**：把任务分解得不合理，导致后续步骤无法执行
2. **工具调用错误**：选错工具、传错参数、误解返回结果
3. **错误累积**：多步执行中，一步的小偏差在后续步骤中被放大
4. **无限循环**：陷入重复行动，无法退出

**工程应对**：

```
提升 Agent 可靠性的手段：
├── 更好的工具描述和参数设计
├── 限制最大步数和单步 token 预算
├── 每步加入结果校验（Guardrails）
├── 关键步骤加入 Human-in-the-Loop
├── 使用推理模型（o3/Claude Opus）提升规划质量
└── 充分的日志和回放能力，便于调试
```

---

### Q3：MCP 和 Function Calling 是什么关系？

**MCP 是通信协议，Function Calling 是模型能力——两者不在同一层。**

```
层次关系：
┌──────────────────────────────────────────────┐
│  模型层：Function Calling / Tool Use          │
│  模型知道"我需要调用 search 工具"              │
├──────────────────────────────────────────────┤
│  协议层：MCP（Model Context Protocol）         │
│  定义 Agent 和工具之间的通信格式和流程          │
├──────────────────────────────────────────────┤
│  实现层：具体的 MCP Server                    │
│  真正执行搜索、数据库查询、文件操作等           │
└──────────────────────────────────────────────┘
```

Function Calling 决定**模型何时、以什么参数**调用工具。MCP 决定**这个调用请求如何传递给工具、工具如何返回结果**。在没有 MCP 的情况下，每个工具的接入方式都不同；有了 MCP，所有工具遵循统一协议。

---

### Q4：多 Agent 系统会不会比单 Agent 更贵？

**通常会贵 2-5 倍，但在适当场景下物有所值。**

成本分析：

| 成本项 | 单 Agent | 多 Agent（3 个） | 说明 |
|--------|----------|------------------|------|
| LLM 调用次数 | N 次 | 3N-5N 次 | Agent 间通信增加调用 |
| 总 token 消耗 | T | 2T-4T | 重复的上下文 + 协调消息 |
| 延迟 | 基准 | 1.5x-3x | 串行的 Agent 间通信 |
| 开发成本 | 低 | 中-高 | 协调逻辑增加复杂度 |

**何时值得**：
- 任务质量要求高（如代码生成 + 代码审查，比单 Agent 自审可靠得多）
- 任务涉及多领域专业知识（每个 Agent 的 prompt 可以更聚焦）
- 需要对抗性验证（一个 Agent 生成，另一个 Agent 挑错）
- 子任务可并行（总延迟可能反而更短）

---

### Q5：如何选择 Agent 框架？还是应该从零搭建？

**取决于项目阶段和复杂度。**

| 项目阶段 | 推荐方式 | 理由 |
|----------|----------|------|
| **学习/探索** | 从零搭建（直接用 API） | 理解底层原理，不被框架抽象遮挡 |
| **快速原型** | CrewAI / OpenAI Agents SDK | 上手快，适合验证想法 |
| **生产项目** | LangGraph / 自研框架 | 需要精确控制、可观测性、可靠性 |
| **研究实验** | AutoGen / 自定义代码 | 灵活性最高 |

**框架的两面性**：框架能加速开发，但也会引入抽象泄漏、版本升级风险、调试困难。在 Agent 领域框架迭代极快（2024-2026 年间 LangChain 的 API 经历了多次重大变更），过度依赖框架有风险。

**务实建议**：核心 Agent 循环的代码量其实不多（100-300 行），如果你的团队有能力维护，从零搭建 + 按需引入框架组件是最灵活的方案。

---

### Q6：Agent 安全性怎么保障？

Agent 比普通 LLM 应用多了"行动"能力，安全风险相应增大：

| 风险 | 场景 | 防护 |
|------|------|------|
| **Prompt 注入** | 恶意输入让 Agent 执行非预期操作 | 输入过滤、工具白名单、权限控制 |
| **工具滥用** | Agent 被诱导调用危险工具（如删除文件） | 危险操作需 Human-in-the-Loop 确认 |
| **数据泄露** | Agent 通过工具访问敏感数据后泄露 | 最小权限原则、数据脱敏 |
| **成本攻击** | 触发 Agent 进入无限循环消耗资源 | 步数限制、token 预算、超时机制 |
| **供应链攻击** | 恶意 MCP Server 返回有害内容 | 只使用受信任的 MCP Server，审查工具返回 |

**2025-2026 最佳实践**：
- OpenAI Agents SDK 内置了 Guardrails 机制，可对输入/输出进行校验
- MCP 协议支持权限声明和能力协商
- 关键操作（写入/删除/支付）必须经过人工确认

---

### Q7：推理模型（o1/o3/Claude Opus）对 Agent 有什么影响？

**推理模型显著提升了 Agent 的规划和多步推理能力，但并不改变 Agent 的基本架构。**

| 能力 | 普通模型（GPT-4o 等） | 推理模型（o3/Claude Opus 等） |
|------|----------------------|------------------------------|
| 简单工具调用 | 足够好 | 同样好，但更贵更慢 |
| 复杂规划（>5 步） | 容易失误 | 显著更可靠 |
| 错误自纠 | 需要 Reflexion 等机制辅助 | 内置的"思考"过程自带纠错 |
| 成本 | 低 | 高 3-10 倍 |

**实用策略**：在 Agent 系统中混合使用——简单步骤用快速模型（GPT-4o-mini），关键的规划和决策步骤用推理模型（o3/Claude Opus）。

---

## 三、延伸资源

### 1. Anthropic MCP 官方文档

https://modelcontextprotocol.io

MCP 协议的权威参考，包含规范文档、SDK 指南、MCP Server 列表。Agent 开发者必读。

### 2. LangGraph 官方教程

https://langchain-ai.github.io/langgraph/

从零构建 Agent 的完整教程，覆盖状态管理、工具使用、Human-in-the-Loop、多 Agent 编排。适合作为可选补充，用来观察 Agent 系统各组件如何协作。

### 3. Lilian Weng — LLM Powered Autonomous Agents

https://lilianweng.github.io/posts/2023-06-23-agent/

OpenAI 研究员 Lilian Weng 的 Agent 综述博客，对 Planning、Memory、Tool Use 等模块有清晰的梳理和图解。虽然发布于 2023 年，核心概念框架至今有效。

---

## 时效性说明

本文件内容截至 2026 年 4 月。以下注意事项：
- **论文列表**侧重于 Agent 领域的奠基性工作，2025-2026 年有大量新工作发表，建议关注 arXiv cs.AI/cs.CL 分类
- **框架推荐**基于 2026 年 4 月的生态现状，框架的活跃度和社区规模可能快速变化
- **MCP 协议**仍在演进中，部分 MCP Server 的接口可能发生变更
- **推理模型**（o3/Claude Opus 4 等）的能力仍在快速提升，Agent 架构的最优设计会随之演进——模型越强，需要的外部脚手架（Planning/Reflexion）越少
