# 推理模型与 Test-time Compute Scaling

> **前置知识**：[1_思维链提示](1_思维链提示.md)、[2_高级推理方法](2_高级推理方法.md)
>
> **内容时效**：截至 2026 年 4 月。本节涉及大量 2024-2026 年的最新进展，更新速度快，请注意各小节标注的时间。

---

## 一、直觉：两种 Scaling 路线

### 1.1 回顾：传统 Scaling（训练时计算）

模块 05 讲过的 Scaling Laws 描述的是一条清晰的路线：

> **Train-time Compute Scaling**：花更多计算在**训练阶段**（更大的模型 + 更多的数据 + 更多的训练步数），得到更强的模型。

```
GPT-3 (175B) → GPT-4 (~1.8T?) → ...
每一代: 更大的模型、更多的数据、更多的训练计算
```

但这条路线遇到了瓶颈：
- **数据快耗尽**：高质量文本数据的增量在放缓
- **成本天花板**：GPT-4 训练成本估计过亿美元
- **收益递减**：模型参数从 175B → 1.8T，但在很多任务上的提升并不成比例

### 1.2 新路线：Test-time Compute Scaling

2024 年，一条新路线被验证有效：

> **Test-time Compute (TTC) Scaling**：模型大小不变，在**推理阶段**花更多计算（让模型"想更久"），得到更好的结果。

```
传统模型:  输入 → [快速生成] → 输出      （固定计算量）
推理模型:  输入 → [长时间思考...] → 输出   （可变计算量，越想越好）

人类类比:
- 传统模型 ≈ 看一眼就答题（系统 1，快速直觉）
- 推理模型 ≈ 仔细想很久再答题（系统 2，慢速推理）
```

**核心发现**（Snell et al., 2024 等）：

在推理阶段增加计算量的效果，可以**等价于**增加训练阶段的计算量。具体来说：

```
一个 14B 的推理模型 + 充足的推理时间
  ≈ 一个 ~100B+ 的传统模型 + 快速推理

在某些数学/编程任务上，这个等价关系已经被实验验证。
```

这意味着：**模型不必更大，只需要更会"想"**。

> **但 test-time compute 有上限**：不是无限思考就一定更好，TTC 存在**收益递减**（diminishing returns）。Snell et al. 的实验明确显示：推理计算量从 1x 增加到 8x 时，准确率提升显著；从 32x 增加到 128x 时，提升几乎可以忽略。更重要的是，如果一个问题超出了模型的知识范围（模型根本不"知道"相关事实），再多的推理时间也无法凭空"想"出正确答案——TTC 只能帮助模型更好地利用已有的知识，不能创造新知识。实际工程中，为 reasoning_effort 设置合理的上限（而非一律 "high"）是性价比的关键。

---

## 二、OpenAI o 系列推理模型

### 2.1 o1：开创推理模型范式（2024.9）

**发布时间**：2024 年 9 月

OpenAI o1 是第一个大规模商用的"推理模型"。它不是在架构上做了什么改变，而是在训练方法上引入了一个关键创新：

> **通过强化学习（RL）训练模型在回答前进行长链推理。**

#### o1 的工作方式

```
传统模型（GPT-4o）:
用户: "证明 √2 是无理数"
模型: "假设 √2 = p/q..."     ← 直接开始生成答案

推理模型（o1）:
用户: "证明 √2 是无理数"
模型 [内部思考，用户不可见]:
  "这是一道经典证明题。我来想想用什么方法..."
  "方法 1: 反证法。假设 √2 是有理数..."
  "让我检查一下推导是否严谨..."
  "第 3 步到第 4 步的推导有漏洞，让我修正..."
  [可能产生数千 token 的内部思维链]
模型 [最终输出]:
  "证明：我们使用反证法。假设 √2 = p/q，其中 p, q 互质..."
```

**关键特征**：

| 特征 | 说明 |
|------|------|
| 内部思维链 | 模型在输出答案前，先在内部进行长链推理（hidden chain-of-thought） |
| 思考时间可变 | 简单问题思考几秒，复杂问题可能思考几十秒到几分钟 |
| 训练方法 | 大规模 RL（强化学习）训练模型学会在推理中自我纠错 |
| 思维链不可见 | 出于安全和商业考虑，用户只能看到最终答案，不能看到思维过程的原始内容（API 中提供摘要） |

#### o1 的惊人表现

| 基准测试 | GPT-4o | o1-preview | o1 (正式版) |
|----------|--------|------------|-------------|
| AIME 2024（竞赛数学） | 13.4% | 56.7% | 83.3% |
| GPQA Diamond（博士级科学问题） | 53.6% | 73.3% | 78.0% |
| Codeforces（竞赛编程） | ~90th | ~93rd | ~99th 百分位 |

### 2.2 o3 与 o4-mini（2025.4）

**发布时间**：2025 年 4 月

OpenAI 在 2025 年 4 月发布了 o3 和 o4-mini，代表推理模型的进一步进化。

#### o3 的主要改进

```
o1 → o3 的进化:
1. 推理能力更强: 在 AIME/Codeforces 等基准上进一步提升
2. 工具使用能力: o3 可以在推理过程中调用工具（搜索、代码执行等）
3. 多模态推理: 支持图像输入的推理（如几何题看图推理）
4. 更高效: 相同推理质量下，token 消耗和延迟有所降低
```

#### o4-mini：高性价比推理模型

o4-mini 是 OpenAI 针对成本敏感场景推出的轻量推理模型：

| 维度 | o3 | o4-mini |
|------|-----|---------|
| 推理能力 | 最强 | 略低于 o3，但远超 GPT-4o |
| 成本 | 高（思考 token 计费） | 约 o3 的 1/5~1/10 |
| 延迟 | 较高 | 较低 |
| 适用场景 | 顶尖推理任务 | 日常推理、成本敏感场景 |
| 工具使用 | 支持 | 支持 |

#### reasoning_effort 参数

o 系列模型提供了 `reasoning_effort` 参数，让开发者控制推理深度：

```python
"""
OpenAI o 系列模型的使用示例
展示 reasoning_effort 参数对推理深度的影响
"""
from openai import OpenAI

client = OpenAI()

def solve_with_reasoning(
    question: str,
    model: str = "o4-mini",
    effort: str = "medium"  # "low", "medium", "high"
) -> dict:
    """
    使用推理模型求解

    reasoning_effort 控制思考深度:
    - "low":    快速回答，适合简单问题。内部思考链短。
    - "medium": 平衡模式。大多数问题推荐此选项。
    - "high":   深度思考，适合竞赛级难题。延迟可能很长。
    """
    response = client.chat.completions.create(
        model=model,
        reasoning_effort=effort,
        messages=[{"role": "user", "content": question}]
    )

    result = response.choices[0].message
    usage = response.usage

    return {
        "answer": result.content,
        "reasoning_tokens": usage.completion_tokens_details.reasoning_tokens,
        "output_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens
    }

# ----- 对比不同 reasoning_effort -----
question = "求证：对所有正整数 n，有 1³+2³+...+n³ = (1+2+...+n)²"

for effort in ["low", "medium", "high"]:
    result = solve_with_reasoning(question, effort=effort)
    print(f"\n=== effort={effort} ===")
    print(f"推理 tokens: {result['reasoning_tokens']}")
    print(f"输出 tokens: {result['output_tokens']}")
    print(f"答案前 200 字: {result['answer'][:200]}...")

# 典型结果:
# effort=low:    推理 ~200 tokens,  可能给出不完整的证明
# effort=medium: 推理 ~1000 tokens, 给出完整但简洁的证明
# effort=high:   推理 ~5000 tokens, 给出详细严谨的证明，可能探索多种方法
```

---

## 三、Test-time Compute Scaling 理论

### 3.1 核心问题

TTC Scaling 回答的核心问题是：

> **在推理阶段，如何最优地分配额外的计算预算来最大化输出质量？**

### 3.2 两种 TTC 策略

推理阶段花更多计算的方式主要有两种：

```
策略 1：并行采样 + 验证（Self-Consistency 路线）
┌──────────────────────────────┐
│ 同一个问题，独立采样 N 次     │
│                              │
│  采样 1 → 答案 A             │
│  采样 2 → 答案 B             │
│  采样 3 → 答案 A             │
│  ...                         │
│  采样 N → 答案 A             │
│                              │
│  验证/投票 → 最终选 A         │
└──────────────────────────────┘

策略 2：顺序深入思考（o1/o3 路线）
┌──────────────────────────────┐
│ 模型自动产生更长的思维链       │
│                              │
│  想法 → 推导 → 检查 →        │
│  发现问题 → 回溯 → 重新推导 → │
│  验证 → 确认 → 输出答案       │
│                              │
│  思维链越长，答案质量越高      │
└──────────────────────────────┘
```

### 3.3 Scaling 曲线

Snell et al. (2024) 的研究表明，TTC 与 Train-time Compute 之间存在有趣的对比关系：

```python
"""
Test-time Compute Scaling 曲线可视化
基于 Snell et al. (2024) "Scaling LLM Test-Time Compute" 的结论
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ----- 模拟 TTC Scaling 曲线 -----
# 推理计算量（相对于基础推理的倍数）
ttc_multiplier = np.array([1, 2, 4, 8, 16, 32, 64, 128])

# 不同策略在数学任务上的准确率（模拟数据，基于论文趋势）
majority_voting = np.array([45, 55, 62, 67, 70, 72, 73, 73.5])  # Self-Consistency 路线
best_of_n = np.array([45, 58, 66, 72, 76, 79, 81, 82])          # Best-of-N + 验证器
sequential_revision = np.array([45, 56, 65, 73, 78, 82, 85, 87])  # 顺序修正（o1 路线）

print("TTC 倍数 | 多数投票 | Best-of-N | 顺序修正（o1风格）")
print("-" * 60)
for i in range(len(ttc_multiplier)):
    print(f"  {ttc_multiplier[i]:>4}x   | {majority_voting[i]:>5.1f}%  | {best_of_n[i]:>5.1f}%  | {sequential_revision[i]:>5.1f}%")

# 输出:
# TTC 倍数 | 多数投票 | Best-of-N | 顺序修正（o1风格）
# ------------------------------------------------------------
#     1x   |  45.0%  |  45.0%  |  45.0%
#     2x   |  55.0%  |  58.0%  |  56.0%
#     4x   |  62.0%  |  66.0%  |  65.0%
#     8x   |  67.0%  |  72.0%  |  73.0%
#    16x   |  70.0%  |  76.0%  |  78.0%
#    32x   |  72.0%  |  79.0%  |  82.0%
#    64x   |  73.0%  |  81.0%  |  85.0%
#   128x   |  73.5%  |  82.0%  |  87.0%

# 关键观察:
# 1. 所有策略都呈现 log-linear 的 scaling 趋势
# 2. 顺序修正（o1 风格）的 scaling 效率最高
# 3. 多数投票最快饱和（收益递减最明显）
# 4. 验证器的质量是 Best-of-N 策略的瓶颈
```

### 3.4 TTC vs Train-time Compute 的互换关系

```
核心发现（简化表述）:

在数学推理任务上：
- 14B 模型 + 128x TTC ≈ 70B+ 传统模型
- 这意味着可以用 "小模型 + 更多推理时间" 替代 "大模型 + 快速推理"

但这个互换不是无条件的：
- 对于需要广泛知识的任务（如世界知识问答），TTC 帮助有限
  （模型不知道的知识，想再久也想不出来）
- 对于推理密集型任务（数学、编程、逻辑），TTC 效果最好
  （知识已在模型中，只需要更多计算来"推导"出来）
```

### 3.5 工程意义

```python
"""
TTC Scaling 的工程决策：什么时候用推理模型？
"""

# 决策框架
def choose_model_strategy(task_type, latency_budget_ms, cost_budget):
    """
    根据任务类型、延迟预算、成本预算选择模型策略

    实际工程中的考虑：
    - 推理模型的 token 成本更高（思考 token 也计费）
    - 推理模型的延迟更高（需要等待"思考"完成）
    - 但在推理密集型任务上，准确率提升可能值回票价
    """
    strategies = {
        "simple_qa": {
            "model": "gpt-4o-mini",
            "reason": "简单问答不需要深度推理",
            "expected_latency": "200-500ms",
            "expected_cost": "低"
        },
        "moderate_reasoning": {
            "model": "o4-mini (effort=low)",
            "reason": "中等推理用轻量推理模型",
            "expected_latency": "1-5s",
            "expected_cost": "中"
        },
        "hard_math_coding": {
            "model": "o3 (effort=high)",
            "reason": "竞赛级难题需要深度推理",
            "expected_latency": "10-60s",
            "expected_cost": "高"
        },
        "cost_sensitive_reasoning": {
            "model": "o4-mini (effort=medium)",
            "reason": "推理需求 + 成本敏感",
            "expected_latency": "2-10s",
            "expected_cost": "中低"
        }
    }

    # 实际中可以更精细，考虑批处理、缓存、降级策略等
    return strategies.get(task_type, strategies["moderate_reasoning"])

# ----- 成本估算示例 -----
print("场景: 每天处理 10,000 道数学题")
print()
print("方案 A: GPT-4o + Self-Consistency (5 次采样)")
print("  - 每题 5 次调用 × ~500 output tokens = ~2,500 tokens")
print("  - 准确率: ~70%")
print("  - 日成本: ~$37.5  (10K × 2.5K tokens × $1.5/M output tokens)")
print()
print("方案 B: o4-mini (effort=medium)")
print("  - 每题 1 次调用, ~1,000 推理 tokens + ~200 输出 tokens")
print("  - 准确率: ~80%")
print("  - 日成本: ~$48  (10K × 1.2K tokens × $4/M tokens, 推理 token 更贵)")
print()
print("方案 C: o3 (effort=high)")
print("  - 每题 1 次调用, ~5,000 推理 tokens + ~500 输出 tokens")
print("  - 准确率: ~90%")
print("  - 日成本: ~$330  (10K × 5.5K tokens × $6/M tokens)")
print()
print("选择取决于: 准确率提升 10-20% 值多少钱？")
```

---

## 四、强化学习训练推理能力

### 4.1 从 RLHF 到推理 RL

模块 05 介绍了 RLHF：用人类偏好训练模型输出更好的回答。推理模型的训练方法是 RLHF 的一个变种，但目标不同：

```
RLHF（模块 05）:
  目标: 让输出符合人类偏好（有帮助、安全、诚实）
  奖励: 人类标注的偏好排序

推理 RL（本节）:
  目标: 让模型在推理任务上给出正确答案
  奖励: 答案是否正确（数学题有标准答案，代码题有测试用例）

关键区别:
  RLHF 的奖励是主观的（人类偏好）
  推理 RL 的奖励是客观的（对/错）→ 更容易规模化
```

### 4.2 两种奖励模型：ORM vs PRM

**ORM（Outcome Reward Model，结果奖励模型）**：
- 只看最终答案是否正确
- 简单直接："答案对了奖励 +1，错了 -1"
- 问题：无法区分"推理过程正确但答案碰巧错了" vs "推理过程全错但答案碰巧对了"

**PRM（Process Reward Model，过程奖励模型）**：
- 评估推理过程中**每一步**是否正确
- 更精细："第 1 步 +1，第 2 步 +1，第 3 步 -1（这步错了），第 4 步..."
- 训练数据更贵：需要标注每一步的正确性

```
ORM:  问题 → [整个推理过程] → 最终答案 → 正确？ → 奖励
       ↑                                         │
       └─────────── 只看最终结果 ─────────────────┘

PRM:  问题 → 步骤1 → 步骤2 → 步骤3 → ... → 最终答案
              ↓ +1    ↓ +1    ↓ -1          ↓ 取决于每步
              └─────── 每一步都有奖励 ────────┘
```

**PRM 的关键论文**：Lightman et al. (2023) *Let's Verify Step by Step*（OpenAI）

### 4.3 GRPO：群组相对策略优化

**GRPO（Group Relative Policy Optimization）** 是 DeepSeek 在训练 DeepSeek-R1 时使用的核心算法。

#### 传统 PPO 的问题

```
传统 PPO（用于 RLHF）:
  需要 4 个模型同时在内存中:
  1. 策略模型（正在训练的模型）
  2. 参考模型（冻结的基线模型）
  3. 奖励模型（评估输出质量）
  4. 价值模型（估计未来奖励）

  → 内存占用巨大，训练不稳定
```

#### GRPO 的简化

```
GRPO 的核心简化:
  1. 去掉价值模型（最占内存的那个）
  2. 用 "组内相对排名" 替代绝对奖励

具体做法:
  对同一个问题，采样一组（比如 8 个）回答:
  - 回答 1: 正确 → 相对奖励 +1
  - 回答 2: 错误 → 相对奖励 -1
  - 回答 3: 正确 → 相对奖励 +1
  - 回答 4: 错误 → 相对奖励 -1
  ...

  不需要绝对的奖励分数，只需要组内的相对好坏！
  → 去掉了价值模型，大幅降低内存需求
```

```python
"""
GRPO 核心概念的伪代码实现
（实际训练需要 GPU 集群和完整的 RL 框架）
"""

def grpo_training_step(
    policy_model,
    reference_model,
    question_batch,
    group_size: int = 8,
    temperature: float = 0.7,
    kl_coeff: float = 0.01
):
    """
    GRPO 一个训练步的伪代码

    与 PPO 的关键区别：
    1. 不需要 value model
    2. 用组内相对排名替代绝对奖励
    3. 训练更稳定，内存占用更小
    """
    total_loss = 0

    for question in question_batch:
        # 第 1 步：对同一问题采样 group_size 个回答
        responses = []
        for _ in range(group_size):
            response = policy_model.generate(question, temperature=temperature)
            responses.append(response)

        # 第 2 步：计算每个回答的奖励（数学题：答案对=1，错=0）
        rewards = []
        for response in responses:
            answer = extract_answer(response)
            correct_answer = get_ground_truth(question)
            reward = 1.0 if answer == correct_answer else 0.0
            rewards.append(reward)

        # 第 3 步：计算组内相对优势（GRPO 的核心）
        # 归一化：均值为 0，标准差为 1
        import numpy as np
        rewards = np.array(rewards)
        mean_reward = rewards.mean()
        std_reward = rewards.std() + 1e-8
        advantages = (rewards - mean_reward) / std_reward
        # advantages > 0: 比组内平均好 → 应该鼓励
        # advantages < 0: 比组内平均差 → 应该抑制

        # 第 4 步：计算策略梯度损失
        for response, advantage in zip(responses, advantages):
            # 计算 log 概率比
            log_prob = policy_model.log_prob(question, response)
            ref_log_prob = reference_model.log_prob(question, response)

            # 策略梯度 + KL 正则化
            ratio = (log_prob - ref_log_prob).exp()
            loss = -(ratio * advantage).mean()  # 鼓励 advantage > 0 的回答
            kl_penalty = kl_coeff * (log_prob - ref_log_prob).mean()
            total_loss += loss + kl_penalty

    return total_loss / len(question_batch)
```

---

## 五、开源推理模型

### 5.1 DeepSeek-R1（2025.1）

**发布时间**：2025 年 1 月

DeepSeek-R1 是开源推理模型的标志性突破，其影响力堪比 LLaMA 对开源基础模型的意义。

#### 训练流程

```
DeepSeek-R1 的训练分为两个阶段:

阶段 1: 纯 RL 冷启动（DeepSeek-R1-Zero）
  ─────────────────────────────────
  直接从 DeepSeek-V3 基础模型出发
  不使用任何 SFT 数据
  只用 GRPO + 数学/代码的正确性奖励
  ↓
  惊人发现: 模型自发学会了:
  - 长链推理（"让我一步步想..."）
  - 自我验证（"让我检查一下这个结果..."）
  - 回溯修正（"等等，这一步好像有问题..."）
  - 类似 "Aha moment": 模型学会说 "wait, let me reconsider..."

阶段 2: 精化训练（DeepSeek-R1）
  ─────────────────────────────────
  用 R1-Zero 生成高质量推理数据
  SFT + RL 结合训练
  增加可读性和多语言支持
  ↓
  最终性能:
  - AIME 2024: 79.8%（接近 o1 的 83.3%）
  - MATH-500: 97.3%
  - Codeforces: ~97th 百分位
```

#### R1 的核心贡献

| 贡献 | 说明 |
|------|------|
| 证明"纯 RL 训练推理"可行 | 不需要人工编写推理示例，RL 就能让模型学会推理 |
| 开源权重和论文 | 完整公开了模型权重、训练细节 |
| 蒸馏到小模型 | 提供了 1.5B/7B/8B/14B/32B/70B 的蒸馏版本 |
| GRPO 的实践验证 | 证明 GRPO 可以替代 PPO 训练推理模型 |

### 5.2 QwQ 与 Qwen3（2024.11 / 2025.4）

**QwQ (Qwen with Questions)**：阿里通义团队于 2024 年 11 月发布的实验性推理模型。

**Qwen3**：2025 年 4 月发布，将推理和非推理能力统一到一个模型中。

#### Qwen3 的"思考模式"创新

Qwen3 引入了一个工程上很优雅的设计：**同一个模型支持"思考"和"非思考"两种模式**。

```python
"""
Qwen3 的思考/非思考模式示例
展示 Hybrid Thinking 的工程实现
"""
from openai import OpenAI

# 假设使用兼容 OpenAI 格式的 API（如 vLLM 部署的 Qwen3）
client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

def qwen3_thinking_mode(question: str, enable_thinking: bool = True) -> dict:
    """
    Qwen3 支持通过 /think 和 /no_think 标签控制推理模式

    enable_thinking=True:  模型会先在 <think>...</think> 中推理，再给出答案
    enable_thinking=False: 模型直接给出答案，不进行内部推理
    """
    if enable_thinking:
        # 思考模式：适合数学、逻辑、编程等推理任务
        messages = [
            {"role": "user", "content": f"{question} /think"}
        ]
    else:
        # 非思考模式：适合翻译、摘要、简单问答等
        messages = [
            {"role": "user", "content": f"{question} /no_think"}
        ]

    response = client.chat.completions.create(
        model="qwen3-235b-a22b",  # MoE 架构，总参数 235B，激活 22B
        messages=messages,
        temperature=0.6 if enable_thinking else 0.3
    )

    content = response.choices[0].message.content

    # 解析思考过程和最终答案
    if "<think>" in content and "</think>" in content:
        thinking = content.split("<think>")[1].split("</think>")[0]
        answer = content.split("</think>")[1].strip()
        return {"thinking": thinking, "answer": answer, "mode": "thinking"}
    else:
        return {"thinking": None, "answer": content, "mode": "non-thinking"}

# ----- 对比两种模式 -----
math_question = "解方程: 3x² - 12x + 9 = 0"

# 思考模式
result = qwen3_thinking_mode(math_question, enable_thinking=True)
print("=== 思考模式 ===")
print(f"思考过程: {result['thinking'][:200]}...")
print(f"最终答案: {result['answer']}")

# 非思考模式
result = qwen3_thinking_mode(math_question, enable_thinking=False)
print("\n=== 非思考模式 ===")
print(f"直接答案: {result['answer']}")
```

### 5.3 开源推理模型对比（截至 2026.4）

| 模型 | 发布时间 | 基座大小 | 推理能力（AIME 2024） | 许可证 | 关键特点 |
|------|----------|----------|---------------------|--------|----------|
| DeepSeek-R1 | 2025.1 | 671B (MoE) | 79.8% | MIT | 首个顶级开源推理模型 |
| DeepSeek-R1-Distill-7B | 2025.1 | 7B | 55.5% | MIT | 小模型蒸馏版 |
| DeepSeek-R1-Distill-32B | 2025.1 | 32B | 72.6% | MIT | 中等蒸馏版 |
| QwQ-32B | 2024.11 | 32B | ~60% | Apache 2.0 | 阿里通义推理模型 |
| Qwen3-235B-A22B | 2025.4 | 235B (MoE, 激活22B) | ~75%+ | Apache 2.0 | Hybrid 思考模式 |
| Qwen3-32B | 2025.4 | 32B | ~70% | Apache 2.0 | 思考/非思考双模式 |

---

## 六、Python 实践：本地推理模型使用

```python
"""
使用开源推理模型的实践示例
通过 vLLM 或 Ollama 本地部署
"""

# ===== 方案 1: 使用 Ollama 部署 DeepSeek-R1 蒸馏版 =====
# 安装: https://ollama.com/
# 拉取模型: ollama pull deepseek-r1:14b

import requests

def ollama_reasoning(question: str, model: str = "deepseek-r1:14b") -> dict:
    """
    通过 Ollama API 使用本地推理模型
    """
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": question,
            "stream": False,
            "options": {
                "temperature": 0.6,
                "num_predict": 4096  # 推理模型需要更多 token
            }
        }
    )
    result = response.json()
    content = result["response"]

    # DeepSeek-R1 的思考过程在 <think>...</think> 中
    if "<think>" in content and "</think>" in content:
        thinking = content.split("<think>")[1].split("</think>")[0]
        answer = content.split("</think>")[1].strip()
    else:
        thinking = None
        answer = content

    return {
        "thinking": thinking,
        "answer": answer,
        "total_duration_ms": result.get("total_duration", 0) / 1_000_000,
        "eval_count": result.get("eval_count", 0)
    }

# ===== 方案 2: 使用 vLLM 部署 + OpenAI 兼容 API =====
# 启动: vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --tensor-parallel-size 2

def vllm_reasoning(question: str) -> dict:
    """
    通过 vLLM 的 OpenAI 兼容 API 使用推理模型
    """
    from openai import OpenAI

    client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        messages=[{"role": "user", "content": question}],
        temperature=0.6,
        max_tokens=4096
    )

    content = response.choices[0].message.content
    # 解析 <think> 标签...
    return {"content": content, "usage": response.usage}


# ===== 方案 3: 推理模型 vs 普通模型的效果对比 =====
def compare_reasoning_vs_standard():
    """对比推理模型和普通模型在数学任务上的表现"""

    questions = [
        "如果 a + b = 10, a × b = 21, 求 a² + b²",
        "一个水池有进水管和出水管。进水管 3 小时注满，出水管 5 小时排空。两管同开，多久注满？",
        "有 100 阶楼梯，每次可以走 1 阶或 2 阶。前 10 种走法分别是什么？总共有多少种走法？"
    ]

    for q in questions:
        print(f"\n问题: {q}")

        # 普通模型
        standard = ollama_reasoning(q, model="qwen2.5:14b")
        print(f"普通模型答案: {standard['answer'][:200]}")

        # 推理模型
        reasoning = ollama_reasoning(q, model="deepseek-r1:14b")
        print(f"推理模型思考: {reasoning['thinking'][:200] if reasoning['thinking'] else 'N/A'}")
        print(f"推理模型答案: {reasoning['answer'][:200]}")
        print(f"推理 token 数: {reasoning['eval_count']}")

# compare_reasoning_vs_standard()
```

---

## 七、工程视角：推理模型的实际应用

### 7.1 何时使用推理模型

```
推理模型适合:
├── 数学/科学计算（多步推导）
├── 代码生成与调试（复杂逻辑）
├── 法律/医疗分析（严谨推理）
├── 数据分析与统计推断
└── 竞赛/考试类任务

推理模型不适合:
├── 简单问答（浪费推理 token）
├── 翻译/摘要（不需要逻辑推理）
├── 创意写作（推理链可能限制创造力）
├── 延迟敏感的实时交互
└── 大批量处理（成本太高）
```

### 7.2 成本优化策略

```python
"""
推理模型的成本优化策略
"""

class ReasoningRouter:
    """
    智能路由：根据问题复杂度选择模型

    核心思路：
    - 简单问题 → 快速模型（GPT-4o-mini / Qwen2.5）
    - 中等问题 → 轻量推理模型（o4-mini / DeepSeek-R1-7B）
    - 复杂问题 → 完整推理模型（o3 / DeepSeek-R1）
    """

    def __init__(self, client):
        self.client = client

    def classify_and_route(self, question: str) -> dict:
        """
        两阶段路由：
        1. 用小模型快速分类问题难度
        2. 根据难度选择合适的模型
        """
        # 第 1 阶段：快速分类（成本极低）
        classification = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"判断以下问题的推理复杂度(1=简单直接,2=中等推理,3=复杂多步推理):\n{question}\n只回答数字。"
            }],
            temperature=0,
            max_tokens=1
        )
        complexity = int(classification.choices[0].message.content.strip())

        # 第 2 阶段：路由到合适的模型
        model_map = {
            1: {"model": "gpt-4o-mini", "reason": "简单问题，无需推理"},
            2: {"model": "o4-mini", "params": {"reasoning_effort": "low"}},
            3: {"model": "o4-mini", "params": {"reasoning_effort": "high"}}
        }

        config = model_map.get(complexity, model_map[2])
        model = config["model"]
        params = config.get("params", {})

        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": question}],
            **params
        )

        return {
            "complexity": complexity,
            "model_used": model,
            "answer": response.choices[0].message.content,
            "tokens": response.usage.total_tokens
        }

    def batch_process(self, questions: list[str]) -> list[dict]:
        """
        批量处理优化：
        - 先分类所有问题
        - 简单问题批量发给快速模型
        - 复杂问题发给推理模型
        - 减少不必要的推理 token 消耗
        """
        results = []
        for q in questions:
            results.append(self.classify_and_route(q))
        return results
```

### 7.3 推理模型的注意事项

| 注意事项 | 说明 |
|----------|------|
| 思考 token 计费 | o 系列的思考 token 也算入计费，费用可能远超输出 token |
| 延迟不可预测 | 复杂问题可能思考几十秒，需要设置合理的超时 |
| 不要叠加 CoT | 推理模型已经自带推理过程，再加 "Let's think step by step" 可能反而干扰 |
| 流式输出体验 | 推理模型的思考过程通常不支持流式输出（先思考完，再一次性输出） |
| 内容过滤 | o1/o3 的思维链不可见，无法审计推理过程（开源模型可以看到） |

---

## 八、推理技术全景图

```
推理技术演进（2022-2026）:

2022 ─── CoT Prompting (Wei)          提示模型写步骤
  │      Zero-shot CoT (Kojima)       "Let's think step by step"
  │      Self-Consistency (Wang)       多次采样投票
  │
2023 ─── Tree-of-Thought (Yao)        搜索式推理
  │      Reflexion (Shinn)             自我反思循环
  │      ReAct (Yao)                   推理 + 行动
  │      PRM (Lightman/OpenAI)         过程奖励模型
  │
2024 ─── OpenAI o1 (2024.9)           首个商用推理模型
  │      QwQ (2024.11)                 阿里推理模型实验版
  │      TTC Scaling 理论 (Snell)      推理时计算 = 训练时计算
  │
2025 ─── DeepSeek-R1 (2025.1)         开源推理模型突破
  │      Qwen3 (2025.4)               Hybrid 思考模式
  │      OpenAI o3/o4-mini (2025.4)   推理模型升级
  │
趋势 ─── 推理与工具使用的深度融合
         开源推理模型的能力快速追赶闭源
         Hybrid 模式：同一模型可选择性推理
         推理能力正在成为基础模型的标配
```

---

## 本节小结

| 概念 | 一句话解释 |
|------|-----------|
| Test-time Compute | 推理时花更多计算换更好结果，与训练时计算互补 |
| OpenAI o 系列 | 通过 RL 训练的推理模型，自动产生长链推理 |
| reasoning_effort | 控制推理深度的参数：low/medium/high |
| GRPO | 去掉价值模型的简化 RL 算法，用组内相对排名做奖励 |
| PRM | 过程奖励模型，评估推理的每一步而非只看最终答案 |
| DeepSeek-R1 | 首个顶级开源推理模型，证明纯 RL 可以训练出推理能力 |
| Qwen3 Hybrid | 同一模型支持思考/非思考双模式，工程上最灵活 |

---

## 时效性说明

> 本节是整个模块中时效性最强的部分，以下内容在 2026 年内可能发生重大变化：
>
> - **OpenAI o 系列**：o3 和 o4-mini 于 2025 年 4 月刚发布，API 和定价可能调整。后续可能有 o5 或新命名系列
> - **DeepSeek-R1**：2025 年 1 月发布，后续版本（R2?）预计在 2025-2026 年推出
> - **Qwen3**：2025 年 4 月发布，混合思考模式是新趋势，但 API 和最佳实践仍在完善
> - **TTC Scaling 理论**：仍处于活跃研究中，最优的推理计算分配策略尚无定论
> - **行业趋势**：推理能力正从"高端功能"变为"基础能力"，预计 2026 年大多数主流模型都将具备推理模式
>
> 建议关注：OpenAI Blog、DeepSeek GitHub、Qwen Blog、Anthropic Research 的最新发布。

---

> 返回 [本模块 README](README.md) | 上一节 [高级推理方法](2_高级推理方法.md) | 下一节 [论文与FAQ](论文与FAQ.md)
