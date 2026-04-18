# Agent 框架原理

> **前置知识**：[预训练语言模型演进](../AI学习_04_预训练语言模型演进/README.md)（理解 LLM 的 In-Context Learning 与指令跟随能力）
>
> **本节目标**：理解 AI Agent 的核心定义与架构——从最朴素的"模型即 Agent"出发，逐步引入 ReAct、Tool Use、Planning 等关键机制，建立对 Agent 系统的完整认知。
>
> **内容时效**：截至 2026 年 4 月

---

## 1. 直觉与概述

### 1.1 核心问题：模型为什么需要"动手"？

一个纯粹的 LLM 只能做一件事：**给定上文，预测下一个 token**。它不能查数据库、不能调 API、不能执行代码、不能浏览网页。就像一个知识渊博但被锁在房间里的专家——能回答问题，但无法亲手做任何事。

Agent 的本质就是给这个专家**一双手和一双眼睛**：

```
纯 LLM：          用户提问 → 模型生成回答 → 结束
                  （一次性，只靠模型内部知识）

Agent：           用户提问 → 模型思考 → 调用工具 → 观察结果 → 继续思考 → ... → 最终回答
                  （多步循环，利用外部工具和环境反馈）
```

### 1.2 什么是 AI Agent？

AI Agent 的定义没有统一标准，但核心要素是一致的：

> **AI Agent 是一个以 LLM 为核心推理引擎，能够自主感知环境、制定计划、调用工具、执行行动，并根据反馈迭代调整的系统。**

关键词拆解：

| 要素 | 含义 | 类比 |
|------|------|------|
| **感知（Perception）** | 接收用户输入、工具返回、环境状态 | 人的眼睛和耳朵 |
| **推理（Reasoning）** | 分析当前状态，决定下一步 | 人的大脑思考 |
| **规划（Planning）** | 将复杂任务分解为子步骤 | 人制定工作计划 |
| **行动（Action）** | 调用工具、执行代码、生成输出 | 人的双手操作 |
| **反思（Reflection）** | 评估行动结果，修正策略 | 人的复盘总结 |

### 1.3 Agent 的核心循环

```
         ┌──────────────────────────────────────┐
         │         Agent 核心循环               │
         │                                      │
         │   感知(Perceive)                     │
         │     │                                │
         │     ▼                                │
         │   推理(Reason) ←── 记忆(Memory)      │
         │     │                                │
         │     ▼                                │
         │   规划(Plan)                         │
         │     │                                │
         │     ▼                                │
         │   行动(Act) ───→ 工具/环境            │
         │     │                                │
         │     ▼                                │
         │   反思(Reflect)                      │
         │     │                                │
         │     └──→ 任务完成？──否──→ 回到感知   │
         │              │                       │
         │             是                       │
         │              │                       │
         │              ▼                       │
         │         输出最终结果                  │
         └──────────────────────────────────────┘
```

---

## 2. 严谨定义与原理

### 2.1 ReAct：推理与行动的交替

**ReAct（Reasoning + Acting）** 是 Yao et al.（2022）提出的关键框架，核心思想极其简洁：

> 让模型在生成行动之前先生成一段推理（Thought），行动之后观察结果（Observation），再进入下一轮推理。

#### ReAct 的三元组循环

```
Thought  →  Action  →  Observation  →  Thought  →  Action  →  Observation  →  ...
 (推理)     (行动)      (观察)        (推理)      (行动)      (观察)
```

**对比其他范式**：

| 范式 | 过程 | 问题 |
|------|------|------|
| 纯推理（CoT） | Thought → Thought → ... → Answer | 无法获取外部信息，容易幻觉 |
| 纯行动（Act-only） | Action → Observation → Action → ... | 无目标导向，盲目试错 |
| **ReAct** | **Thought → Action → Observation → ...** | **推理指导行动，观察修正推理** |

#### ReAct 的 Prompt 模板

```
Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
```

### 2.2 Tool Use（函数调用）

Tool Use 是 Agent 执行行动的核心机制。从 2023 年 OpenAI 引入 Function Calling 开始，到 2025 年已成为所有主流模型的标配能力。

#### 工作原理

```
用户消息 + 工具定义（JSON Schema）
        │
        ▼
    LLM 推理
        │
        ├── 直接回答（不需要工具）
        │
        └── 生成工具调用请求
                │
                ▼
            应用层执行工具
                │
                ▼
            工具返回结果
                │
                ▼
            LLM 综合结果，继续推理或回答
```

#### 工具定义的 JSON Schema

```json
{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "获取指定城市的当前天气信息",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称，如 '北京'、'上海'"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "温度单位，默认摄氏度"
                }
            },
            "required": ["city"]
        }
    }
}
```

**关键设计决策**：

| 决策点 | 选项 | 权衡 |
|--------|------|------|
| 工具数量 | 少量精选 vs 大量全面 | 工具越多，模型选择越困难；通常 10-20 个为宜 |
| 描述质量 | 简洁 vs 详尽 | 描述是模型选择工具的唯一依据，需要准确且有区分度 |
| 参数设计 | 简单类型 vs 复杂嵌套 | 模型对简单参数更可靠，深嵌套结构容易出错 |
| 并行调用 | 串行 vs 并行 | 2024 年起主流模型支持并行工具调用，提升效率 |

### 2.3 Planning：任务分解与计划生成

复杂任务需要分解为可执行的子步骤。Planning 是 Agent 从"能回答问题"到"能完成任务"的关键能力。

#### 主流 Planning 策略

**1. 预先规划（Plan-then-Execute）**

```
用户任务 → 一次性生成完整计划 → 按步骤逐个执行
优点：全局视野，步骤清晰
缺点：无法根据中间结果动态调整
```

**2. 动态规划（Interleaved Planning）**

```
用户任务 → 规划第一步 → 执行 → 根据结果规划下一步 → 执行 → ...
优点：灵活，能根据中间结果调整
缺点：可能陷入局部最优，缺乏全局视野
```

**3. 混合规划（Plan-and-Replan）**——2025 年的主流做法

```
用户任务 → 生成初步计划 → 执行第一步 → 观察结果
                                           │
                                    需要修改计划？
                                    ├── 否 → 继续执行下一步
                                    └── 是 → 重新规划剩余步骤
```

### 2.4 Reflexion：反思与自我修正

Shinn et al.（2023）提出的 Reflexion 框架，让 Agent 能从失败中学习：

```
               ┌──────────────────────────┐
               │      Episode 1           │
               │  尝试 → 失败 → 反思      │
               │  "我忽略了边界条件"       │
               └──────────┬───────────────┘
                          │ 反思经验存入记忆
                          ▼
               ┌──────────────────────────┐
               │      Episode 2           │
               │  利用反思 → 改进 → 成功  │
               └──────────────────────────┘
```

Reflexion 的核心并非重试，而是**结构化的自我反思**——生成文字形式的经验教训，作为后续尝试的上下文。

---

## 3. Python 代码实践

### 3.1 从零实现一个最小 ReAct Agent

```python
"""
最小 ReAct Agent 实现
不依赖任何 Agent 框架，仅使用 OpenAI API，展示核心循环
"""
import json
import re
from openai import OpenAI

client = OpenAI()  # 需要设置 OPENAI_API_KEY 环境变量

# ── 工具定义 ──────────────────────────────────────────────
def search_web(query: str) -> str:
    """模拟网络搜索"""
    # 实际项目中替换为真实搜索 API（如 Tavily、SerpAPI）
    mock_results = {
        "Python 3.12 新特性": "Python 3.12 于 2023 年 10 月发布，主要特性包括：改进的错误消息、"
                              "f-string 语法增强、类型参数语法（PEP 695）、per-interpreter GIL（PEP 684）。",
        "2026年AI趋势": "2026 年 AI 主要趋势：Agent 系统成为主流应用形态，"
                        "多模态模型能力持续提升，推理模型（如 o3、Claude Opus）成为标配。",
    }
    for key, value in mock_results.items():
        if key in query or any(word in query for word in key.split()):
            return value
    return f"未找到关于 '{query}' 的相关结果。"

def calculator(expression: str) -> str:
    """安全的数学计算"""
    try:
        # 仅允许数学运算，防止代码注入
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "错误：包含不允许的字符"
        result = eval(expression)  # 生产环境应使用 ast.literal_eval 或专用库
        return str(result)
    except Exception as e:
        return f"计算错误：{e}"

# 工具注册表
TOOLS = {
    "search_web": search_web,
    "calculator": calculator,
}

TOOL_DESCRIPTIONS = """
可用工具：
1. search_web(query) - 搜索网络获取信息
2. calculator(expression) - 执行数学计算
"""

# ── ReAct 核心循环 ─────────────────────────────────────────
REACT_SYSTEM_PROMPT = f"""你是一个能使用工具的 AI 助手。请使用以下格式回答问题：

Thought: <你的思考过程>
Action: <工具名称>
Action Input: <工具输入>

当你观察到工具返回结果后，继续思考。当你得到最终答案时，使用：

Thought: 我已经得到了足够的信息
Final Answer: <最终答案>

{TOOL_DESCRIPTIONS}

重要规则：
- 每次只调用一个工具
- 仔细分析工具返回的结果
- 如果第一次搜索没找到想要的信息，尝试换个关键词
- 最多执行 5 轮工具调用
"""

def run_react_agent(question: str, max_steps: int = 5) -> str:
    """运行 ReAct Agent 循环"""
    messages = [
        {"role": "system", "content": REACT_SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}"},
    ]

    for step in range(max_steps):
        # 调用 LLM
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0,
        )
        assistant_msg = response.choices[0].message.content
        messages.append({"role": "assistant", "content": assistant_msg})
        print(f"\n{'='*60}")
        print(f"Step {step + 1}:")
        print(assistant_msg)

        # 检查是否有 Final Answer
        if "Final Answer:" in assistant_msg:
            final = assistant_msg.split("Final Answer:")[-1].strip()
            return final

        # 解析 Action 和 Action Input
        action_match = re.search(r"Action:\s*(.+)", assistant_msg)
        input_match = re.search(r"Action Input:\s*(.+)", assistant_msg)

        if action_match and input_match:
            tool_name = action_match.group(1).strip()
            tool_input = input_match.group(1).strip()

            # 执行工具
            if tool_name in TOOLS:
                observation = TOOLS[tool_name](tool_input)
            else:
                observation = f"错误：未知工具 '{tool_name}'"

            # 将观察结果加入对话
            obs_msg = f"Observation: {observation}"
            messages.append({"role": "user", "content": obs_msg})
            print(f"\n{obs_msg}")
        else:
            # 无法解析 Action，要求模型重新生成
            messages.append({
                "role": "user",
                "content": "请按照 Thought/Action/Action Input 或 Final Answer 格式继续。"
            })

    return "达到最大步数限制，未能得出最终答案。"

# ── 使用示例 ──────────────────────────────────────────────
if __name__ == "__main__":
    answer = run_react_agent("Python 3.12 有哪些新特性？帮我算一下 3.12 * 100 是多少。")
    print(f"\n最终答案：{answer}")
```

### 3.2 使用 OpenAI Function Calling（原生 Tool Use）

```python
"""
使用 OpenAI 原生 Function Calling 实现 Tool Use
这是 2024-2026 年最主流的 Agent 工具调用方式
"""
import json
from openai import OpenAI

client = OpenAI()

# ── 工具定义（JSON Schema）─────────────────────────────────
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "获取指定股票的当前价格",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "股票代码，如 AAPL、GOOGL"
                    }
                },
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "发送电子邮件",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "收件人邮箱"},
                    "subject": {"type": "string", "description": "邮件主题"},
                    "body": {"type": "string", "description": "邮件正文"}
                },
                "required": ["to", "subject", "body"]
            }
        }
    }
]

# ── 工具执行函数 ──────────────────────────────────────────
def execute_tool(name: str, arguments: dict) -> str:
    """根据工具名和参数执行对应功能"""
    if name == "get_stock_price":
        # 模拟返回
        prices = {"AAPL": 198.50, "GOOGL": 175.30, "MSFT": 420.80}
        symbol = arguments["symbol"]
        price = prices.get(symbol, None)
        if price:
            return json.dumps({"symbol": symbol, "price": price, "currency": "USD"})
        return json.dumps({"error": f"未找到股票 {symbol}"})

    elif name == "send_email":
        return json.dumps({"status": "sent", "message_id": "msg_12345"})

    return json.dumps({"error": f"未知工具: {name}"})

# ── Agent 循环 ────────────────────────────────────────────
def run_tool_agent(user_message: str) -> str:
    messages = [
        {"role": "system", "content": "你是一个有用的助手，可以查询股票价格和发送邮件。"},
        {"role": "user", "content": user_message}
    ]

    while True:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            tool_choice="auto",  # 模型自行决定是否调用工具
        )

        msg = response.choices[0].message

        # 情况 1：模型直接回答，不调用工具
        if msg.tool_calls is None:
            return msg.content

        # 情况 2：模型请求调用工具（可能是多个并行调用）
        messages.append(msg)  # 把 assistant 消息（含 tool_calls）加入历史

        for tool_call in msg.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)
            print(f"  调用工具: {func_name}({func_args})")

            result = execute_tool(func_name, func_args)

            # 将工具结果加入消息历史
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    # 继续循环，让模型根据工具结果生成回答或继续调用工具

if __name__ == "__main__":
    answer = run_tool_agent("帮我查一下苹果公司的股价，然后发邮件给 boss@company.com 告诉他最新价格。")
    print(f"\n最终回答：{answer}")
```

### 3.3 Plan-and-Execute Agent 模式

```python
"""
Plan-and-Execute Agent：先规划后执行，支持动态重规划
"""
from openai import OpenAI

client = OpenAI()

def plan_and_execute(task: str) -> str:
    """Plan-and-Execute 模式的 Agent"""

    # ── 第一阶段：生成计划 ────────────────────────────────
    plan_prompt = f"""你是一个任务规划专家。请将以下任务分解为 3-7 个具体步骤。
每个步骤应该是可独立执行的，且按逻辑顺序排列。

输出格式（严格 JSON）：
{{"steps": ["步骤1", "步骤2", ...]}}

任务：{task}"""

    plan_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": plan_prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    import json
    plan = json.loads(plan_response.choices[0].message.content)
    steps = plan["steps"]

    print("生成的计划：")
    for i, step in enumerate(steps, 1):
        print(f"  {i}. {step}")

    # ── 第二阶段：逐步执行 ────────────────────────────────
    results = []
    for i, step in enumerate(steps):
        exec_prompt = f"""你正在执行一个多步任务。

原始任务：{task}
完整计划：{json.dumps(steps, ensure_ascii=False)}
当前步骤（第 {i+1} 步）：{step}
之前步骤的结果：{json.dumps(results, ensure_ascii=False) if results else "无"}

请执行当前步骤并返回结果。如果发现计划需要调整，请在结果中说明。"""

        exec_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": exec_prompt}],
            temperature=0,
        )
        result = exec_response.choices[0].message.content
        results.append({"step": step, "result": result})
        print(f"\n步骤 {i+1} 完成：{step}")
        print(f"  结果：{result[:200]}...")

    # ── 第三阶段：汇总 ───────────────────────────────────
    summary_prompt = f"""请汇总以下任务的执行结果：
任务：{task}
各步骤结果：{json.dumps(results, ensure_ascii=False)}

请给出简洁完整的最终结果。"""

    summary = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": summary_prompt}],
        temperature=0,
    )
    return summary.choices[0].message.content
```

---

## 4. 工程视角

### 4.1 ReAct vs Function Calling：该用哪个？

| 维度 | ReAct（Prompt 驱动） | Function Calling（原生支持） |
|------|----------------------|------------------------------|
| 实现方式 | 通过 prompt 约定格式，解析文本 | 模型原生输出结构化调用 |
| 可靠性 | 依赖模型遵循格式，可能出错 | 结构化输出，极少格式错误 |
| 灵活性 | 高——可自定义任何格式 | 受限于 API 支持的模式 |
| 推理可见性 | Thought 过程完全可见 | 思考过程通常不暴露 |
| 推荐场景 | 研究/教学、需要可解释性 | 生产环境，追求稳定性 |

**工程建议**：2025-2026 年的生产项目，优先使用 Function Calling；在需要可解释性或模型不支持原生 Tool Use 时，使用 ReAct。

### 4.2 工具设计的工程原则

```
好的工具设计                          坏的工具设计
─────────                            ─────────
✓ 名称自描述                          ✗ 名称含糊
  "search_database"                    "do_thing"

✓ 描述包含何时使用                    ✗ 描述缺失或误导
  "当需要查询用户历史                   "一个搜索函数"
   订单时使用"

✓ 参数简单、类型明确                  ✗ 参数复杂嵌套
  {"query": "string"}                  {"config": {"sub": {"deep": ...}}}

✓ 返回结构化、可解读                  ✗ 返回原始 dump
  {"status": "ok", "data": [...]}      "<html>...10KB...</html>"

✓ 包含错误处理                        ✗ 异常直接抛出
  {"error": "未找到用户"}               raise Exception(...)
```

### 4.3 Agent 系统的常见陷阱

| 陷阱 | 说明 | 应对 |
|------|------|------|
| **无限循环** | Agent 反复调用同一工具，得到相同结果 | 设置最大步数限制（通常 5-15 步） |
| **工具选择错误** | 模型选错工具或传错参数 | 改善工具描述，添加示例，减少工具数量 |
| **幻觉行动** | 模型"假装"已执行工具，编造结果 | 严格校验工具调用格式，分离思考和行动 |
| **成本爆炸** | 多轮循环导致 token 消耗巨大 | 监控每次请求的 token 数，设置预算上限 |
| **上下文溢出** | 工具返回内容过长，超出上下文窗口 | 对工具返回进行摘要或截断 |

### 4.4 Agent 架构的演进趋势（2024-2026）

```
2023: ReAct + 手动 Prompt 工程
        │
        ▼
2024: Function Calling 成为标配 + LangChain/LangGraph 框架化
        │
        ▼
2025: MCP 协议统一工具接口 + 推理模型（o1/o3）提升规划能力
      OpenAI Agents SDK 发布 + Computer Use（Claude）
        │
        ▼
2026: Agent-native 应用架构 + 多 Agent 协作成为主流
      "Vibe coding" 催生大量 Agent 驱动的开发工具
```

**2025-2026 重要进展**：
- **Anthropic Claude Computer Use**（2024.10）：模型直接操作桌面 GUI，标志着 Agent 从 API 调用扩展到 GUI 交互
- **OpenAI Agents SDK**（2025.03）：从实验性的 Swarm 框架升级为正式 SDK，内置 handoff、guardrails、tracing
- **MCP 协议广泛采用**（2025）：Anthropic 提出的开放标准，统一了 Agent-工具通信接口（详见第 3 节）
- **推理模型驱动 Agent**（2025-2026）：o3/Claude Opus 等推理模型显著提升 Agent 的规划和多步推理能力

---

## 时效性说明

本节内容截至 2026 年 4 月。以下内容可能快速变化：
- Function Calling 的 API 格式各厂商仍在迭代，以官方文档为准
- ReAct 作为概念框架仍有效，但具体实现方式已逐渐被原生 Tool Use 替代
- Agent 的最佳实践随模型能力提升持续演进——更强的模型需要更少的 Prompt 工程技巧
