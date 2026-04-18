# 多 Agent 与 MCP

> **前置知识**：[Agent 框架原理](1_Agent框架原理.md)、[记忆与状态管理](2_记忆与状态管理.md)
>
> **本节目标**：从单 Agent 扩展到多 Agent 协作——理解主流的协作模式（讨论/分工/层级）、MCP（Model Context Protocol）协议的设计原理，以及 LangGraph、CrewAI、AutoGen 等框架的定位与选型。
>
> **内容时效**：截至 2026 年 4 月

---

## 1. 直觉与概述

### 1.1 核心问题：为什么一个 Agent 不够？

单 Agent 在面对复杂任务时会遇到根本性瓶颈：

```
单 Agent 的困境：

1. 角色冲突 ── 同一个 Agent 既要写代码又要审代码，自己很难发现自己的错误
2. 上下文膨胀 ── 一个 Agent 处理所有子任务，上下文窗口很快被撑满
3. 能力天花板 ── 没有一个 prompt 能让模型同时精通所有领域
4. 串行瓶颈 ── 可并行的子任务只能依次处理
```

多 Agent 的核心思路是**分治（Divide and Conquer）**：

```
复杂任务 → 拆分为子任务 → 分配给专业 Agent → 各自处理 → 汇总结果

类比：一个人做不好的事，组建一个团队来做。
      CEO 负责决策，设计师负责 UI，工程师负责代码，测试负责质量。
```

### 1.2 多 Agent 协作的三种模式

```
模式一：讨论型（Debate/Discussion）
┌──────────────────────────────────────┐
│  Agent A ←──→ Agent B ←──→ Agent C  │
│     对等讨论，互相挑战，达成共识       │
└──────────────────────────────────────┘
适用：决策、头脑风暴、观点生成

模式二：分工型（Division of Labor）
┌──────────────────────────────────────┐
│         任务分解 → 并行执行           │
│  Agent A(搜索) | Agent B(分析) | Agent C(写作) │
│         └──── 结果合并 ────┘          │
└──────────────────────────────────────┘
适用：流水线式任务、各环节独立

模式三：层级型（Hierarchical）
┌──────────────────────────────────────┐
│           Supervisor Agent           │
│          ┌───┼───┼───┐              │
│          ▼   ▼   ▼   ▼              │
│         A    B   C    D              │
│      (Worker Agents)                 │
│   Supervisor 分配任务、审核结果       │
└──────────────────────────────────────┘
适用：复杂项目管理、需要质量控制
```

### 1.3 MCP：Agent 与工具的"USB 接口"

**MCP（Model Context Protocol）** 是 Anthropic 于 2024 年 11 月提出的开放协议，目标是**标准化 Agent 与外部工具/数据源之间的通信接口**。

> **关键概念澄清：MCP 是协议层，不是框架。** MCP 定义的是 Agent 与工具之间的**通信规范**（消息格式、调用流程、能力发现机制），类似于 HTTP 定义了浏览器与服务器之间的通信规范。MCP 本身不是某个具体的 Agent 框架实现——LangGraph、CrewAI、OpenAI Agents SDK 等都可以作为 MCP 客户端使用 MCP 协议连接工具。不要把 MCP 与这些框架混为一谈：MCP 解决的是"工具怎么接"的标准化问题，框架解决的是"Agent 怎么编排"的工程问题。

```
没有 MCP 的世界：                    有 MCP 的世界：
┌────────┐  自定义 API  ┌────┐      ┌────────┐  MCP 协议  ┌────────┐
│ Agent A├──────────────┤工具1│      │        ├────────────┤MCP 服务│
│        ├──另一套 API──┤工具2│      │  Any   │  统一接口  │  工具1 │
│        ├──又一套 API──┤工具3│      │ Agent  ├────────────┤  工具2 │
└────────┘              └────┘      │        │            │  工具3 │
每个工具都要写不同的适配代码          └────────┘            └────────┘
                                    一套协议连接所有工具
```

**类比**：USB 出现之前，每个外设都有不同的接口（串口、并口、PS/2……）。USB 统一了接口标准，任何设备即插即用。MCP 对 Agent 工具生态做的就是同样的事。

---

## 2. 严谨定义与原理

### 2.1 多 Agent 协作模式详解

#### 模式一：讨论型（Debate）

多个 Agent 围绕同一问题进行多轮讨论，各自提出观点、质疑他人、修正自己，最终通过某种机制（投票/共识/裁判）得出结论。

**典型流程**：

```
Round 1:  Agent A: "我认为方案 X 最优，因为..."
          Agent B: "我不同意，方案 Y 在性能上更好..."
          Agent C: "两个方案各有优劣，但方案 X 的维护成本更低..."

Round 2:  Agent A: "B 提到的性能问题确实存在，但可以通过缓存缓解..."
          Agent B: "如果加缓存，方案 X 的复杂度会增加，接近方案 Y..."
          Agent C: "综合来看，方案 X + 缓存是最佳折衷..."

共识：    采用方案 X，辅以缓存优化
```

**关键设计点**：
- 每个 Agent 赋予不同的"人设"（角色背景、偏好、专业方向）
- 设定终止条件（最大轮数 or 达成共识）
- 可设置一个"裁判 Agent"做最终决策

#### 模式二：分工型（Pipeline / Parallel）

将任务分解为多个阶段或子任务，交给专业 Agent 分别处理。

**流水线模式**：

```
用户需求 → Agent1(需求分析) → Agent2(方案设计) → Agent3(代码实现) → Agent4(代码审查) → 输出
```

**并行模式**：

```
                   ┌→ Agent1(搜索学术论文) ──┐
用户需求 → 分解器  ├→ Agent2(搜索行业报告) ──┼→ 合并器 → 输出
                   └→ Agent3(搜索新闻资讯) ──┘
```

#### 模式三：层级型（Hierarchical / Supervisor）

一个 Supervisor Agent 负责理解任务、分配工作、审核结果；多个 Worker Agent 负责具体执行。

```
                    Supervisor
                    ├── 理解用户需求
                    ├── 制定计划
                    ├── 分配子任务给 Worker
                    ├── 审核 Worker 结果
                    ├── 必要时要求 Worker 重做
                    └── 汇总最终结果

Worker A (研究员)    Worker B (程序员)    Worker C (测试员)
├── 搜索资料         ├── 编写代码          ├── 编写测试
├── 整理报告         ├── 修复 bug          ├── 执行测试
└── 返回给 Supervisor └── 返回给 Supervisor └── 返回给 Supervisor
```

### 2.2 MCP 协议原理

#### MCP 的架构

MCP 采用**客户端-服务端**架构：

```
┌──────────────────────────────────────────────────────┐
│                    Host Application                  │
│    (IDE / Chat UI / Agent Framework)                 │
│                                                      │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐    │
│  │ MCP Client │  │ MCP Client │  │ MCP Client │    │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘    │
└────────┼───────────────┼───────────────┼────────────┘
         │ MCP Protocol  │ MCP Protocol  │ MCP Protocol
         ▼               ▼               ▼
  ┌────────────┐  ┌────────────┐  ┌────────────┐
  │ MCP Server │  │ MCP Server │  │ MCP Server │
  │  (GitHub)  │  │ (Database) │  │(Filesystem)│
  └────────────┘  └────────────┘  └────────────┘
```

**核心概念**：

| 概念 | 说明 | 方向 |
|------|------|------|
| **Tools** | 服务端暴露的可调用函数 | 客户端 → 服务端 |
| **Resources** | 服务端提供的数据/文件（类似 REST 资源） | 客户端读取 |
| **Prompts** | 服务端定义的 prompt 模板 | 客户端使用 |
| **Sampling** | 服务端请求客户端（即 LLM）进行推理 | 服务端 → 客户端 |

#### MCP 的通信协议

MCP 基于 **JSON-RPC 2.0**，支持两种传输方式：

```
传输方式 1：stdio（标准输入输出）
  适用于本地进程间通信
  Host 启动 MCP Server 作为子进程，通过 stdin/stdout 通信

传输方式 2：HTTP + SSE（Server-Sent Events）
  适用于远程服务
  MCP Server 作为 HTTP 服务运行，支持请求-响应和服务端推送
```

#### 一次 MCP Tool 调用的完整流程

```
1. 初始化：客户端连接服务端，获取可用 tool 列表
     Client → Server: initialize
     Server → Client: {tools: [{name: "search_code", ...}, ...]}

2. Agent 推理：LLM 决定需要调用某个工具
     LLM output: "需要搜索代码库中的相关函数"

3. 工具调用：客户端通过 MCP 协议调用服务端工具
     Client → Server: tools/call {name: "search_code", arguments: {"query": "..."}}

4. 结果返回：
     Server → Client: {content: [{type: "text", text: "找到 3 个相关函数..."}]}

5. 继续推理：Agent 根据工具结果继续处理
```

### 2.3 Agent 框架生态

#### 框架定位图谱（2025-2026）

```
                    高层（声明式/角色驱动）
                         ▲
                         │
                    CrewAI ── 角色扮演、自然语言定义流程
                    AutoGen ── 多 Agent 对话框架
                         │
                    ─────┼────── 中间层
                         │
                    LangGraph ── 状态机 + 图编排
                    OpenAI Agents SDK ── 轻量级 Agent 编排
                         │
                    ─────┼────── 低层
                         │
                    原始 API ── 直接调用 LLM API + 自己编排
                         │
                         ▼
                    低层（命令式/代码驱动）
```

#### 框架对比

| 维度 | LangGraph | CrewAI | AutoGen | OpenAI Agents SDK |
|------|-----------|--------|---------|-------------------|
| **核心抽象** | 状态图（StateGraph） | 角色（Agent + Crew） | 多 Agent 对话 | Agent + Handoff |
| **编排方式** | 代码定义节点和边 | 声明式任务分配 | 对话驱动 | 函数式组合 |
| **多 Agent** | 通过图节点实现 | 原生支持 Crew 协作 | 原生多 Agent 对话 | 通过 handoff 转接 |
| **状态管理** | 内置 State + Checkpointing | 基本 | 对话历史 | 上下文传递 |
| **模型支持** | 多模型 | 多模型 | 多模型 | 仅 OpenAI |
| **适用场景** | 复杂工作流、需要精确控制 | 快速原型、角色协作 | 研究、多 Agent 实验 | OpenAI 生态内的 Agent |
| **学习曲线** | 中等 | 低 | 中等 | 低 |

---

## 3. Python 代码实践

### 3.1 多 Agent 讨论系统（纯 API 实现）

```python
"""
多 Agent 讨论系统：多个角色围绕一个问题进行多轮讨论
不依赖任何 Agent 框架，展示多 Agent 协作的核心机制
"""
from openai import OpenAI
from dataclasses import dataclass

client = OpenAI()

@dataclass
class AgentRole:
    name: str
    system_prompt: str

def create_discussion_agents() -> list[AgentRole]:
    """创建具有不同视角的讨论 Agent"""
    return [
        AgentRole(
            name="架构师",
            system_prompt=(
                "你是一位资深软件架构师。你关注系统的可扩展性、可维护性和长期演进。"
                "你倾向于选择成熟、稳定的技术方案。讨论时请从架构角度提出观点，"
                "同时认真考虑其他参与者的意见。发言简洁（100字以内）。"
            ),
        ),
        AgentRole(
            name="产品经理",
            system_prompt=(
                "你是一位经验丰富的产品经理。你关注用户体验、开发速度和交付时间。"
                "你倾向于选择能最快交付价值的方案。讨论时请从产品和用户角度提出观点，"
                "同时认真考虑其他参与者的意见。发言简洁（100字以内）。"
            ),
        ),
        AgentRole(
            name="安全工程师",
            system_prompt=(
                "你是一位安全工程师。你关注系统的安全性、合规性和风险控制。"
                "你倾向于对任何可能引入安全隐患的方案提出质疑。讨论时请从安全角度提出观点，"
                "同时认真考虑其他参与者的意见。发言简洁（100字以内）。"
            ),
        ),
    ]

def run_discussion(topic: str, max_rounds: int = 3) -> str:
    """运行多 Agent 讨论"""
    agents = create_discussion_agents()
    discussion_history: list[dict] = []

    print(f"讨论主题：{topic}\n{'='*60}")

    for round_num in range(1, max_rounds + 1):
        print(f"\n--- 第 {round_num} 轮讨论 ---")

        for agent in agents:
            # 构建该 Agent 看到的消息
            messages = [
                {"role": "system", "content": agent.system_prompt},
                {"role": "user", "content": f"讨论主题：{topic}"},
            ]

            # 加入之前的讨论历史
            for entry in discussion_history:
                if entry["speaker"] == agent.name:
                    messages.append({"role": "assistant", "content": entry["content"]})
                else:
                    messages.append({
                        "role": "user",
                        "content": f"[{entry['speaker']}]: {entry['content']}"
                    })

            # 在最后一轮要求总结
            if round_num == max_rounds:
                messages.append({
                    "role": "user",
                    "content": "这是最后一轮讨论，请给出你的最终观点和建议。"
                })

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.7,
                max_tokens=200,
            )
            content = response.choices[0].message.content
            discussion_history.append({"speaker": agent.name, "content": content})
            print(f"\n[{agent.name}]: {content}")

    # ── 裁判 Agent 汇总 ──────────────────────────────────
    summary_messages = [
        {
            "role": "system",
            "content": "你是讨论的主持人。请根据所有参与者的讨论，给出公正的总结和最终建议。"
        },
        {
            "role": "user",
            "content": f"讨论主题：{topic}\n\n讨论记录：\n" + "\n".join(
                f"[{e['speaker']}]: {e['content']}" for e in discussion_history
            ) + "\n\n请总结讨论结论。"
        }
    ]

    summary = client.chat.completions.create(
        model="gpt-4o",
        messages=summary_messages,
        temperature=0,
    )
    conclusion = summary.choices[0].message.content
    print(f"\n{'='*60}")
    print(f"讨论结论：\n{conclusion}")
    return conclusion


if __name__ == "__main__":
    run_discussion("是否应该将公司核心系统从单体架构迁移到微服务？")
```

### 3.2 MCP Server 实现（Python SDK）

```python
"""
MCP Server 示例：暴露一组工具供 Agent 调用
使用官方 Python SDK（mcp 包）

安装：pip install mcp
"""
# 注意：MCP SDK 的 API 仍在迭代中，以下代码基于 2025 年初版本
# 请参考 https://modelcontextprotocol.io 获取最新文档

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import json
import sqlite3
from datetime import datetime

# ── 创建 MCP Server ──────────────────────────────────────
app = Server("demo-mcp-server")

# ── 定义工具 ─────────────────────────────────────────────

@app.list_tools()
async def list_tools() -> list[Tool]:
    """声明该 Server 提供的所有工具"""
    return [
        Tool(
            name="query_database",
            description="执行 SQL 查询并返回结果。仅支持 SELECT 语句。",
            inputSchema={
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "SQL SELECT 查询语句"
                    }
                },
                "required": ["sql"]
            }
        ),
        Tool(
            name="get_current_time",
            description="获取当前服务器时间",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="read_file",
            description="读取指定路径的文本文件内容",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "文件路径"
                    },
                    "max_lines": {
                        "type": "integer",
                        "description": "最多读取的行数，默认 100",
                        "default": 100
                    }
                },
                "required": ["path"]
            }
        ),
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """执行工具调用"""
    if name == "query_database":
        sql = arguments["sql"].strip()
        # 安全检查：只允许 SELECT
        if not sql.upper().startswith("SELECT"):
            return [TextContent(type="text", text="错误：仅允许 SELECT 查询")]
        try:
            conn = sqlite3.connect("app.db")
            cursor = conn.execute(sql)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = cursor.fetchall()
            conn.close()
            result = {"columns": columns, "rows": rows, "row_count": len(rows)}
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]
        except Exception as e:
            return [TextContent(type="text", text=f"查询错误：{e}")]

    elif name == "get_current_time":
        now = datetime.now().isoformat()
        return [TextContent(type="text", text=f"当前时间：{now}")]

    elif name == "read_file":
        path = arguments["path"]
        max_lines = arguments.get("max_lines", 100)
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()[:max_lines]
            return [TextContent(type="text", text="".join(lines))]
        except Exception as e:
            return [TextContent(type="text", text=f"读取文件错误：{e}")]

    return [TextContent(type="text", text=f"未知工具：{name}")]

# ── 启动 Server ──────────────────────────────────────────
async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### 3.3 LangGraph 层级 Agent 系统

```python
"""
使用 LangGraph 构建层级 Agent 系统
Supervisor Agent 分配任务给 Worker Agent

安装：pip install langgraph langchain-openai
"""
from typing import Annotated, TypedDict, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState, END, START

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# ── Worker Agent 定义 ────────────────────────────────────

def researcher_agent(state: MessagesState) -> dict:
    """研究员 Agent：负责搜索和整理信息"""
    messages = [
        SystemMessage(content=(
            "你是一位研究员。你的任务是根据要求搜索和整理相关信息。"
            "请直接提供研究结果，格式清晰。"
        )),
    ] + state["messages"]

    response = llm.invoke(messages)
    return {"messages": [AIMessage(content=f"[研究员] {response.content}")]}

def coder_agent(state: MessagesState) -> dict:
    """程序员 Agent：负责编写代码"""
    messages = [
        SystemMessage(content=(
            "你是一位 Python 程序员。你的任务是根据要求编写高质量的代码。"
            "请提供完整可运行的代码，包含注释。"
        )),
    ] + state["messages"]

    response = llm.invoke(messages)
    return {"messages": [AIMessage(content=f"[程序员] {response.content}")]}

def reviewer_agent(state: MessagesState) -> dict:
    """审核员 Agent：负责审查和质量控制"""
    messages = [
        SystemMessage(content=(
            "你是一位代码审核员。你的任务是审查代码质量、检查潜在问题、"
            "提出改进建议。请给出具体的审查意见。"
        )),
    ] + state["messages"]

    response = llm.invoke(messages)
    return {"messages": [AIMessage(content=f"[审核员] {response.content}")]}

# ── Supervisor Agent ────────────────────────────────────

WORKERS = ["researcher", "coder", "reviewer"]

def supervisor_agent(state: MessagesState) -> dict:
    """Supervisor：决定下一步交给哪个 Worker 或结束"""
    messages = [
        SystemMessage(content=(
            "你是一个团队主管。你的团队有以下成员：\n"
            "- researcher（研究员）：搜索和整理信息\n"
            "- coder（程序员）：编写代码\n"
            "- reviewer（审核员）：审查代码质量\n\n"
            "根据对话历史，决定下一步应该交给哪个成员，或者任务已完成。\n"
            "请只回复一个词：researcher / coder / reviewer / FINISH"
        )),
    ] + state["messages"]

    response = llm.invoke(messages)
    decision = response.content.strip().lower()

    # 标准化决策
    if "finish" in decision:
        next_worker = "FINISH"
    elif "researcher" in decision:
        next_worker = "researcher"
    elif "coder" in decision:
        next_worker = "coder"
    elif "reviewer" in decision:
        next_worker = "reviewer"
    else:
        next_worker = "FINISH"  # 无法识别时默认结束

    return {
        "messages": [AIMessage(content=f"[主管] 分配给：{next_worker}")],
    }

def route_after_supervisor(state: MessagesState) -> Literal["researcher", "coder", "reviewer", "__end__"]:
    """根据 Supervisor 的决策路由到对应 Worker"""
    last_msg = state["messages"][-1].content
    if "researcher" in last_msg:
        return "researcher"
    elif "coder" in last_msg:
        return "coder"
    elif "reviewer" in last_msg:
        return "reviewer"
    return "__end__"

# ── 构建图 ───────────────────────────────────────────────

def build_hierarchical_graph():
    graph = StateGraph(MessagesState)

    # 添加节点
    graph.add_node("supervisor", supervisor_agent)
    graph.add_node("researcher", researcher_agent)
    graph.add_node("coder", coder_agent)
    graph.add_node("reviewer", reviewer_agent)

    # 入口 → Supervisor
    graph.add_edge(START, "supervisor")

    # Supervisor 根据决策路由
    graph.add_conditional_edges("supervisor", route_after_supervisor)

    # 每个 Worker 完成后回到 Supervisor
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("coder", "supervisor")
    graph.add_edge("reviewer", "supervisor")

    return graph.compile()


if __name__ == "__main__":
    app = build_hierarchical_graph()

    result = app.invoke({
        "messages": [
            HumanMessage(content="帮我写一个 Python 脚本，用于监控网站可用性并在宕机时发送邮件通知。")
        ]
    })

    print("\n最终结果：")
    for msg in result["messages"]:
        print(f"\n{msg.content[:300]}")
```

---

## 4. 工程视角

### 4.1 多 Agent vs 单 Agent：何时该用多 Agent？

```
使用单 Agent：                        使用多 Agent：
├── 任务简单、步骤少（< 5 步）         ├── 任务复杂、涉及多个领域
├── 不需要不同视角                    ├── 需要多角色协作（如写+审）
├── 上下文窗口足够                    ├── 单 Agent 上下文不够
├── 延迟敏感（多 Agent 更慢）          ├── 质量优先于速度
└── 系统简单、易于维护                └── 有成熟的框架支持
```

**常见误区**：不是所有复杂任务都需要多 Agent。先确认单 Agent + 好的 Planning 无法解决，再考虑多 Agent。多 Agent 引入了额外的通信开销、协调复杂度和调试难度。

### 4.2 MCP 的工程实践

#### MCP Server 配置示例（Claude Desktop）

```json
{
    "mcpServers": {
        "filesystem": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/workspace"]
        },
        "github": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
            "env": {
                "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_xxx"
            }
        },
        "custom-db": {
            "command": "python",
            "args": ["my_mcp_server.py"],
            "env": {
                "DB_CONNECTION_STRING": "postgresql://..."
            }
        }
    }
}
```

#### MCP 生态现状（2026 年 4 月）

| 类别 | 代表性 MCP Server | 状态 |
|------|-------------------|------|
| 文件系统 | @modelcontextprotocol/server-filesystem | 官方，稳定 |
| GitHub | @modelcontextprotocol/server-github | 官方，稳定 |
| 数据库 | PostgreSQL / SQLite MCP Servers | 社区，活跃 |
| 搜索 | Brave Search / Tavily MCP Servers | 社区，活跃 |
| 浏览器 | Playwright MCP Server | 社区，迭代中 |
| Slack/Email | 多个社区实现 | 社区，质量参差 |

### 4.3 框架选型决策树

```
你的需求是什么？

├── 快速原型、多角色协作
│   └── CrewAI ── 声明式定义角色和任务，上手最快
│
├── 复杂工作流、需要精确控制流程
│   └── LangGraph ── 状态图编排，支持条件分支、循环、checkpointing
│
├── 研究多 Agent 对话和辩论
│   └── AutoGen ── 原生多 Agent 对话框架
│
├── OpenAI 生态、轻量级 Agent
│   └── OpenAI Agents SDK ── 官方 SDK，简洁易用
│
└── 极致控制、不想依赖框架
    └── 原始 API ── 直接用 OpenAI/Anthropic API + 自己写循环
        （本模块第 1 节的方式）
```

### 4.4 多 Agent 系统的调试与可观测性

| 挑战 | 说明 | 工具/方案 |
|------|------|----------|
| **对话追踪** | 多 Agent 之间的消息流难以追踪 | LangSmith、Braintrust、Arize Phoenix |
| **成本监控** | 多 Agent 的总 token 消耗难以预测 | 每个 Agent 单独计量，设预算上限 |
| **死循环检测** | Agent 之间互相等待或无限传递 | 最大轮次限制 + 循环检测 |
| **结果归因** | 最终错误由哪个 Agent 引起 | 结构化日志，标记每个 Agent 的输出 |
| **回放调试** | 复现问题需要重放完整流程 | LangGraph Checkpointing、对话快照 |

### 4.5 Agent 框架的 2025-2026 重要变化

**LangGraph（LangChain 团队）**：
- 从 LangChain 的 AgentExecutor 演进为独立的图编排引擎
- 2025 年成为 LangChain 生态最推荐的 Agent 构建方式
- 内置 Persistence（checkpointing）、Human-in-the-loop、Streaming

**CrewAI**：
- 2024 年中发布，以"角色扮演"为核心抽象，快速获得社区关注
- 2025 年加入 CrewAI Flows（工作流编排），向更通用的 Agent 框架演进
- 适合非技术背景用户快速搭建多 Agent 系统

**OpenAI Agents SDK**（原 Swarm）：
- 2025 年 3 月从实验性项目升级为正式 SDK
- 核心概念：Agent（带指令和工具的 LLM）+ Handoff（Agent 间任务转接）
- 内置 Guardrails（护栏）和 Tracing（追踪）
- 限制：仅支持 OpenAI 模型

**AutoGen（微软）**：
- 2023 年发布，专注于多 Agent 对话场景
- 2025 年发布 AutoGen 0.4，完全重写架构
- 引入 Event-driven 模型，支持更灵活的 Agent 通信

---

## 时效性说明

本节内容截至 2026 年 4 月。以下内容可能快速变化：
- **MCP 协议**仍在积极演进中（版本号、新能力如 Elicitation/OAuth），以 https://modelcontextprotocol.io 为准
- **框架 API**：LangGraph、CrewAI、OpenAI Agents SDK 均处于活跃开发期，代码示例可能需要根据最新版本调整
- **多 Agent 最佳实践**尚未成熟——业界仍在探索最有效的协作模式，不同任务类型的最优方案可能差异很大
- **新框架不断涌现**：Google ADK（Agent Development Kit, 2025.04）等新入局者持续丰富生态
