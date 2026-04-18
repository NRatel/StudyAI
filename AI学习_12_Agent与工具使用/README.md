# AI学习：12 Agent与工具使用

> **内容时效**：截至 2026 年 4 月。Agent 领域迭代极快，本模块标注了各技术的时间节点，请结合阅读时的最新进展理解。

## 本模块导读

**学什么**：从单次问答到自主完成复杂任务——AI Agent 的完整技术栈。从 ReAct 框架的"推理+行动"循环，到 Tool Use 的函数调用机制，到多 Agent 协作与 MCP 协议，理解 Agent 如何感知、推理、规划、行动。

**为什么学**：2024-2026 年，Agent 是大模型落地最核心的方向。理解 Agent 架构，才能理解为什么模型需要调用工具、为什么需要记忆管理、为什么多 Agent 协作能解决单模型做不到的复杂任务。这是从"会用模型"到"构建智能系统"的关键跳跃。

**学完能做什么**：
- 理解 AI Agent 的核心循环：感知 → 推理 → 规划 → 行动 → 反思
- 掌握 ReAct、Tool Use、Planning 等关键设计模式
- 理解短期/长期/工作记忆的实现方案与工程取舍
- 了解多 Agent 协作的主流模式与 MCP 协议
- 能基于 LangGraph / CrewAI / OpenAI Agents SDK 等框架搭建 Agent 系统

**前置知识**：[04 预训练语言模型演进](../AI学习_04_预训练语言模型演进/README.md)（理解 LLM 的基本能力边界与 In-Context Learning）

**预估学习时间**：4~6 小时

**难度标记**：★★★（前沿篇）

## 目录

| 序号 | 文件 | 内容 |
|------|------|------|
| 1 | [Agent框架原理](1_Agent框架原理.md) | AI Agent 定义、感知→推理→行动循环、ReAct、Tool Use、Planning |
| 2 | [记忆与状态管理](2_记忆与状态管理.md) | 短期记忆（上下文窗口）、长期记忆（向量存储/摘要压缩）、工作记忆（scratch pad） |
| 3 | [多Agent与MCP](3_多Agent与MCP.md) | 多 Agent 协作模式、MCP 协议原理、Agent 框架生态 |
| - | [论文与FAQ](论文与FAQ.md) | 关键论文、常见误区、延伸资源 |

## 模块知识图谱

```
AI Agent（本模块）
├── 单 Agent 架构
│   ├── 感知层：用户输入 / 环境反馈 / 工具返回
│   ├── 推理层：ReAct（推理+行动交替）、CoT（思维链）
│   ├── 规划层：任务分解、计划修正、Reflexion（反思）
│   ├── 行动层：Tool Use（函数调用）、代码执行、API 调用
│   └── 记忆层：短期 / 长期 / 工作记忆
├── 多 Agent 协作
│   ├── 协作模式：讨论 / 分工 / 层级
│   └── 通信协议：MCP（Model Context Protocol）
└── 框架生态
    ├── LangGraph（状态机编排）
    ├── CrewAI（角色扮演协作）
    ├── OpenAI Agents SDK（原 Swarm）
    └── AutoGen（微软多 Agent 对话）
```

## 时效性说明

Agent 领域处于快速演进中，以下内容可能在 6~12 个月内发生显著变化：
- **MCP 协议**：Anthropic 于 2024 年 11 月发布，2025 年已获广泛采用，但协议细节仍在迭代
- **框架生态**：LangGraph、CrewAI、OpenAI Agents SDK 等框架的 API 可能发生重大变更
- **模型能力**：随着模型推理能力提升（如 o1/o3/Claude Opus 系列），Agent 架构的最优设计也在持续演进

建议学习时结合各框架的最新文档与 changelog。
