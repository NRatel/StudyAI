# GPT-4 之后的演进（截至 2026.4）

> **前置知识**：[GPT 系列架构演进](1_GPT系列架构演进.md)（GPT-1/2/3/4 的架构与关键创新）、[Scaling Laws 与涌现](2_ScalingLaws与涌现.md)（Chinchilla 配比、涌现能力）、[对齐技术](3_对齐技术.md)（RLHF/DPO）

---

## 直觉与概述

### GPT-4 之后发生了什么？

GPT-4（2023.3）发布后，大模型领域进入了一个**多路线并行爆发**的阶段。如果用一句话概括：

> GPT-4 证明了 scaling 的天花板还远未到来，但行业发现"变大"不是唯一出路——变聪明、变高效、变开放同样重要。

核心趋势可以用四条线索串联：

```
GPT-4 之后的四条演进主线
================================================

1. 开源追赶    LLaMA → Mistral → DeepSeek → Qwen
   ─────────  开源模型从"能用"到"好用"到"某些场景超越闭源"

2. 推理增强    o1 → o3 → DeepSeek-R1 → QwQ
   ─────────  模型学会"想一想再回答"，推理时间换准确率

3. 多模态融合  GPT-4V → GPT-4o → Gemini → LLaMA 4
   ─────────  从文本插件到原生多模态统一架构

4. 效率优化    GQA → MLA → MoE → 长上下文
   ─────────  更少的显存、更快的推理、更长的上下文
```

这四条线索并非独立的——它们交叉融合。例如 DeepSeek-V3 同时在开源、效率（MLA + MoE）和推理（R1）三条线上推进。

### 为什么工程师需要关注这些？

- **模型选型**：闭源 API 和开源部署各有最优解，需要理解各模型的架构差异才能做出选择
- **成本控制**：GQA、MLA、MoE 等效率优化直接影响推理成本
- **能力边界**：推理模型与通用模型的能力分布不同，选错模型事倍功半

---

## 严谨定义与原理

### 一、开源模型的崛起

GPT-4 是闭源的（架构、数据、训练细节均未公开），这在学术界和工业界引发了强烈的开源需求。2023~2026 年间，开源社区的追赶速度超出了大多数人的预期。

#### 1.1 LLaMA 系列（Meta）

LLaMA 是开源大模型的分水岭——Meta 用实际行动证明了开源权重的价值。

**LLaMA 1（2023.2）**

| 配置 | 参数量 | 训练 token 数 | 特点 |
|------|--------|-------------|------|
| LLaMA-7B | 7B | 1T | 比 GPT-3 小得多，性能接近 |
| LLaMA-13B | 13B | 1T | 超越 GPT-3（175B） |
| LLaMA-33B | 33B | 1.4T | — |
| LLaMA-65B | 65B | 1.4T | 接近 Chinchilla-70B |

关键创新：
- 遵循 Chinchilla 最优配比：**用更多数据训练更小的模型**
- 架构：Pre-Norm（RMSNorm）、SwiGLU 激活、RoPE 位置编码
- 证明了"数据比参数量更重要"

**LLaMA 2（2023.7）**

| 改进 | 细节 |
|------|------|
| 训练数据 | 1T → 2T tokens |
| 上下文 | 2K → 4K |
| GQA | 70B 版本使用 GQA（8 个 KV 头，64 个 Q 头） |
| 对齐 | 提供 Chat 版本（SFT + RLHF） |
| 许可 | 商用友好许可证 |

**LLaMA 3 / 3.1（2024.4 / 2024.7）**

这是开源模型的里程碑——LLaMA 3 在多数基准上接近或达到 GPT-4 水平。

```
LLaMA 3 关键创新
================================================================
词表大小:   32K → 128K（更好的多语言支持和编码效率）
训练数据:   2T → 15T+ tokens（7 倍以上的增长）
上下文:     4K → 8K (LLaMA 3) → 128K (LLaMA 3.1)
GQA:        全系列使用 GQA
架构:       与 LLaMA 2 相同的基础架构，胜在数据和规模
尺寸:       8B / 70B / 405B
```

LLaMA 3 的启示：**在 Transformer 架构已经成熟的今天，数据质量和数据规模可能比架构创新更重要**。405B 的 LLaMA 3.1 在 MMLU、HumanEval、GSM8K 等主流基准上与 GPT-4 Turbo 相当。

**LLaMA 4（2025.4）**

LLaMA 4 标志着 Meta 对开源模型战略的重大升级——首次采用 MoE 架构和原生多模态。

| 模型 | 总参数 | 激活参数 | 专家数 | 上下文 | 特点 |
|------|--------|---------|-------|--------|------|
| Scout | ~109B | ~17B | 16 | 10M tokens | 超长上下文 |
| Maverick | ~400B | ~17B | 128 | — | 高性能 |
| Behemoth | 更大 | — | — | — | 教师模型 |

关键架构变化：
- **MoE（Mixture of Experts）**：从 dense 模型转向稀疏 MoE，激活参数仅 17B 但总参数远超
- **原生多模态**：文本和图像的 early fusion（从底层即融合，而非后期拼接）
- **训练数据**：30T+ tokens
- 训练时使用 Behemoth 作为教师模型进行蒸馏

#### 1.2 Mistral 系列（Mistral AI，法国）

核心理念："用更小的模型达到更好的效果"。

| 模型 | 时间 | 参数 | 关键创新 |
|------|------|------|---------|
| Mistral 7B | 2023.9 | 7B | 滑动窗口注意力（SWA）+ GQA，超越 LLaMA 2 13B |
| Mixtral 8x7B | 2023.12 | 47B 总 / 13B 激活 | 开源首个高质量 MoE，top-2 路由 |
| Mixtral 8x22B | 2024.4 | 8x22B / ~44B 激活 | 128K 上下文 |
| Mistral Large 2 | 2024.7 | 123B (dense) | 竞争 GPT-4o 级别 |

**滑动窗口注意力（SWA）**：每个 token 只关注窗口内 token（如 4096），通过多层堆叠扩大感受野（类似 CNN），复杂度从 $O(n^2)$ 降至 $O(n \cdot w)$。

#### 1.3 DeepSeek 系列（中国深度求索）

DeepSeek 在 2024~2025 年以极高的性价比和架构创新引起全球关注。

**DeepSeek-V2（2024.5）**

- 架构创新：**MLA（Multi-head Latent Attention）** + **DeepSeekMoE**
- 236B 总参数，21B 激活参数
- MLA 将 KV Cache 压缩 ~93%，详见后文效率优化章节

**DeepSeek-V3（2024.12）**

| 维度 | 细节 |
|------|------|
| 总参数 | 671B |
| 激活参数 | 37B |
| 架构 | MLA + DeepSeekMoE |
| 上下文 | 128K tokens |
| 训练数据 | 14.8T tokens |
| 训练成本 | ~$5.6M（H800 GPU） |
| 性能 | 多项基准与 GPT-4o 相当 |

$5.6M 的训练成本震惊了业界——这比 LLaMA 3 405B 的训练成本低一个数量级以上。核心原因：
1. MoE 的稀疏激活大幅降低计算量
2. MLA 降低显存占用
3. FP8 混合精度训练
4. 精细的工程优化

**DeepSeek-R1（2025.1）**

- 基于 V3 架构，通过 RL（强化学习）训练推理能力
- 在数学、代码、科学推理上与 OpenAI o1 竞争
- 开源权重，引发了推理模型的开源浪潮
- 详见后文"推理模型"章节

#### 1.4 Qwen 系列（阿里巴巴通义千问）

**Qwen 2 / 2.5（2024）**

- 参数规模覆盖 0.5B ~ 72B
- Qwen 2.5 在多语言（尤其中文）、代码、数学上表现突出
- 开源许可友好，被社区广泛采用和微调

**QwQ-32B（2025 初）**

- 阿里的推理模型，对标 o1/R1
- 32B 参数即可在推理任务上与更大模型竞争
- 链式思考 + RL 训练

#### 1.5 其他重要玩家

| 模型 | 组织 | 时间 | 要点 |
|------|------|------|------|
| Grok-1 (314B) | xAI | 2024.3 | 开源 MoE，Apache 2.0 |
| Grok-3 | xAI | 2025.2 | 10 万张 H100 训练 |
| Gemma 2 | Google | 2024 | 2B/9B/27B，优秀的小模型 |
| Phi-3/4 | Microsoft | 2024~2025 | "小模型大智慧"路线 |
| Yi 系列 | 零一万物 | 2023~2024 | 中文能力突出 |

---

### 二、GPT-4o 与实时多模态

#### 2.1 从 GPT-4V 到 GPT-4o

GPT-4 本身是文本模型。GPT-4V（2023.9）和 GPT-4o（2024.5）代表了两种不同的多模态路径：

```
GPT-4V（插件式多模态）:
┌──────────┐     ┌──────────┐
│ 视觉编码器 │────→│          │
│ (独立模型) │     │  GPT-4   │ → 文本输出
└──────────┘     │ (文本模型) │
     图像 ──────→│          │
                 └──────────┘
图像理解是"嫁接"上去的，视觉和语言在后期融合

GPT-4o（原生多模态）:
                 ┌──────────────────┐
  文本 ──────────→│                  │
  图像 ──────────→│  统一多模态模型    │ → 文本/音频/图像
  音频 ──────────→│  (end-to-end)    │
                 └──────────────────┘
所有模态在同一个模型中端到端训练
```

**GPT-4o 的关键特征**：
- **"o" = omni**（全模态）
- 文本、图像、音频在同一个模型中原生处理
- 实时语音对话：延迟低至 ~300ms（接近人类对话节奏）
- 音频直入直出——不需要先 ASR（语音识别）再 LLM 再 TTS（语音合成）的管线
- 性能与 GPT-4 Turbo 相当，推理速度更快，成本更低

#### 2.2 GPT-4.1（2025.4）

OpenAI 在 2025.4 发布了 GPT-4.1 系列，定位为**开发者优先的 API 模型**：

| 模型 | 上下文 | 定位 |
|------|--------|------|
| GPT-4.1 | 1M tokens | 旗舰，代码和长文本能力强 |
| GPT-4.1 mini | 1M tokens | 平衡性能和成本 |
| GPT-4.1 nano | — | 最低延迟、最低成本 |

关键点：
- **非推理模型**——专注通用能力，不走 o 系列的"慢思考"路线
- 1M token 上下文窗口
- 在代码生成、指令遵循方面显著优于 GPT-4o
- 通过 API 而非 ChatGPT 消费端发布

#### 2.3 Google Gemini 的多模态路线

Google 走的也是原生多模态路线：

| 模型 | 时间 | 要点 |
|------|------|------|
| Gemini 1.0 | 2023.12 | Ultra/Pro/Nano 三档 |
| Gemini 1.5 Pro | 2024.2 | 1M token 上下文，MoE 架构 |
| Gemini 2.0 Flash | 2024.12 | 快速高效多模态 |
| Gemini 2.5 Pro | 2025 | "思考型"模型，融合推理能力 |
| Gemini 2.5 Flash | 2025 | 轻量推理模型 |

Gemini 的特色是从一开始就按多模态设计，且与 Google 生态（搜索、Workspace、Android）深度集成。

---

### 三、推理模型（o 系列路线）

这是 GPT-4 之后最重要的范式变化之一。

#### 3.1 核心思想：Test-time Compute Scaling

传统的 Scaling Law 关注**训练时**的算力投入——更多参数、更多数据、更多 GPU 小时。推理模型则开辟了第二个维度：**推理时**也可以投入更多算力来提升性能。

```
两个维度的 Scaling
================================================================

维度 1: Train-time Scaling（传统）
  更大模型 + 更多数据 + 更多训练 → 更强的基础能力
  例: GPT-3 → GPT-4（参数量和数据量扩大）

维度 2: Test-time Scaling（新范式）
  相同模型 + 推理时更多"思考" → 更准确的答案
  例: o1 在回答前生成长链式推理

核心洞察:
  对于困难问题，让模型"想一想"比单纯增大模型更高效。
  类比：人类面对难题时，多想一会儿比换一个更大的脑子更有用。
```

数学上，如果 $C_{\text{train}}$ 是训练算力，$C_{\text{test}}$ 是推理算力，传统方法只优化 $C_{\text{train}}$，而推理模型同时优化两者的分配。

#### 3.2 OpenAI o 系列

**o1（2024.9）**

- 第一个公开的"推理模型"
- 在回答前进行内部链式思考（chain-of-thought），用户看到的是"思考"过程的摘要
- 在数学（AIME、MATH）、代码（Codeforces）、科学推理上大幅超越 GPT-4o
- 代价：延迟更高、成本更高（消耗更多 output token 进行思考）

**o3 / o3-mini（2025 初）**

- o3 在 ARC-AGI 基准上取得突破性成绩
- o3-mini 以更低成本提供接近 o3 的推理能力
- 跳过了 o2（据传因商标问题）

**o4-mini（2025.4）**

- 更高效的推理模型
- 对标 Gemini 2.5 Flash、Claude Haiku 等轻量模型

#### 3.3 开源推理模型

OpenAI 的 o 系列是闭源的。但开源社区快速跟进：

| 模型 | 时间 | 方法 | 要点 |
|------|------|------|------|
| DeepSeek-R1 | 2025.1 | RL 训练 | 开源，671B MoE，与 o1 竞争 |
| QwQ-32B | 2025 初 | RL + CoT | 32B 即可做推理 |
| 各种蒸馏版 | 2025 | 从 R1 蒸馏 | R1-distill-qwen-7B 等 |

DeepSeek-R1 的意义在于：它证明了推理能力可以通过 RL 在开源模型上训练出来，不需要 OpenAI 的闭源技术。

#### 3.4 推理模型的技术本质

推理模型并不是一种新架构——底层仍然是 Transformer。关键区别在于**训练方法**和**推理策略**：

```
普通模型:
  输入: "123 + 456 = ?"
  输出: "579"     （直觉反应，一步到位）

推理模型:
  输入: "123 + 456 = ?"
  思考: "个位: 3+6=9，十位: 2+5=7，百位: 1+4=5 → 579"
  输出: "579"     （分步推理后作答）
```

训练方法的核心：
1. **SFT 阶段**：用带有详细推理过程的数据训练模型"展示思考过程"
2. **RL 阶段**：用强化学习（如 GRPO）奖励正确答案，让模型自主探索更好的推理策略
3. **关键发现**（DeepSeek-R1 论文）：纯 RL 训练即可让模型涌现出链式推理能力，不一定需要 SFT 阶段

> **与模块 11（推理与思维链）的关联**：本节概述推理模型的发展脉络和定位，技术细节（CoT/ToT/RL 方法论）在模块 11 中深入展开。

---

### 四、效率与架构创新

GPT-4 之后，架构创新的核心目标从"让模型更大"转向"让推理更高效"。以下是三个最重要的技术。

#### 4.1 GQA（Grouped Query Attention）

**问题**：标准 Multi-Head Attention（MHA）中，每个注意力头都有独立的 K、V 投影。在推理时，KV Cache 的显存占用与头数成正比，成为长序列推理的瓶颈。

**解决思路**：让多个 Q 头共享同一组 KV 头。

```
MHA（Multi-Head Attention）:     H 个 Q 头, H 个 K 头, H 个 V 头
  Q_1 → K_1, V_1
  Q_2 → K_2, V_2
  ...
  Q_H → K_H, V_H
  KV Cache 大小 ∝ H

MQA（Multi-Query Attention）:    H 个 Q 头, 1 个 K 头, 1 个 V 头
  Q_1 ┐
  Q_2 ├→ K_1, V_1          （所有 Q 共享同一个 KV）
  ... │
  Q_H ┘
  KV Cache 大小 ∝ 1         质量有所下降

GQA（Grouped Query Attention）:  H 个 Q 头, G 个 K 头, G 个 V 头
  Q_1 ┐→ K_1, V_1          （每组内共享 KV）
  Q_2 ┘
  Q_3 ┐→ K_2, V_2
  Q_4 ┘
  ...
  KV Cache 大小 ∝ G          G 通常远小于 H
```

**GQA 的配置记法**：GQA-G 表示有 G 个 KV 组。
- GQA-1 = MQA（所有 Q 头共享 1 组 KV）
- GQA-H = MHA（每个 Q 头有独立 KV）
- 常见配置：LLaMA 2 70B 使用 GQA-8（64 个 Q 头分成 8 组）

**实际效果**：

| 方法 | KV Cache（相对） | 质量（相对 MHA） | 采用者 |
|------|-----------------|------------------|--------|
| MHA | 1x | 基准 | GPT-3, BERT |
| MQA | 1/H | 有退化 | PaLM, StarCoder |
| GQA-8 | 8/H | 接近 MHA | LLaMA 2/3, Mistral |

> **从 MHA 迁移到 GQA**：可以将已有 MHA checkpoint 的 KV 权重在组内做 mean-pooling，然后用原始训练量 5% 的数据 uptrain，即可获得 GQA 模型。

#### 4.2 MLA（Multi-head Latent Attention）

MLA 是 DeepSeek-V2 提出的注意力机制，核心思想是**用低秩压缩取代 KV Cache 的直接存储**。

**标准 MHA 的 KV Cache**：

```
每个 token 缓存:  K ∈ R^{n_heads × d_head},  V ∈ R^{n_heads × d_head}
总缓存:  2 × n_heads × d_head × seq_len   （per layer）
```

**MLA 的做法**：

```
步骤 1: 压缩（Down Projection）
  h_t → c_t = W_DKV · h_t    c_t ∈ R^{d_c}    d_c ≪ n_heads × d_head

  KV Cache 只存 c_t（极小的压缩向量）

步骤 2: 解压并计算注意力（Up Projection）
  K_t = W_UK · c_t           恢复出 Key
  V_t = W_UV · c_t           恢复出 Value

  然后正常做 Attention
```

**关键问题：RoPE 怎么办？**

RoPE（旋转位置编码）需要作用在 K 上，但如果 K 是从 $c_t$ 恢复出来的，RoPE 会破坏低秩结构（无法将位置信息吸收进权重矩阵）。DeepSeek 的解决方案是**解耦 RoPE**：

```
K = [K_content ; K_rope]
       ↑              ↑
  从 c_t 恢复      独立计算，带 RoPE
  (不带位置信息)   (额外少量 KV 头专门携带位置信息)
```

**MLA 的效果**：

| 指标 | MHA | GQA | MLA |
|------|-----|-----|-----|
| KV Cache | 基准 | 降低数倍 | 降低 ~93% |
| 性能 | 基准 | 接近 MHA | **优于 MHA** |

MLA 不仅省显存，性能还更好——因为低秩压缩相当于一种正则化，且压缩/解压过程本身增加了模型表达能力。

#### 4.3 MoE（Mixture of Experts）

MoE 不是新概念（Shazeer 2017），但在 GPT-4 之后成为大模型的主流架构选择。

**核心思想**：FFN 层从"一个大网络"变为"多个小专家 + 路由器"，每次只激活部分专家。

```
标准 Transformer FFN:
  x → FFN(x)                    所有参数都参与计算

MoE FFN:
  x → Router(x) → top-k 专家    只有被选中的专家参与计算

  Router: x → softmax(W_r · x) → 选择 top-k 个专家
  输出:   Σ(gate_i × Expert_i(x))   i ∈ top-k
```

**MoE 的关键挑战**：
1. **负载均衡**：辅助损失（Auxiliary Loss）惩罚路由不均
2. **通信开销**：不同专家在不同 GPU 上，需要 all-to-all 通信
3. **训练不稳定**：路由是离散决策，需用直通估计器或软路由

**主要 MoE 模型**：

| 模型 | 总参数 | 激活参数 | 专家数 | top-k | 年份 |
|------|--------|---------|-------|-------|------|
| GPT-4（传闻） | ~1.8T | ~280B | 16 | 2 | 2023 |
| Mixtral 8x7B | 47B | 13B | 8 | 2 | 2023 |
| DeepSeek-V3 | 671B | 37B | 256 | 8 | 2024 |
| LLaMA 4 Scout | 109B | 17B | 16 | — | 2025 |
| LLaMA 4 Maverick | 400B+ | 17B | 128 | — | 2025 |

#### 4.4 更长的上下文窗口

上下文窗口的扩展也是 GPT-4 之后的重要趋势：

演进路径：GPT-3(4K) → GPT-4(32K) → GPT-4 Turbo(128K) → Claude 3(200K) → Gemini 1.5(1M) → GPT-4.1(1M) → LLaMA 4 Scout(10M)

关键技术：**RoPE 外推**（NTK-aware Scaling、YaRN）、**Ring Attention**（长序列环状分布到多 GPU）、**稀疏注意力**（Sliding Window 等，降低 $O(n^2)$）。

---

### 五、大模型发展时间线（2023.3 ~ 2026.4）

| 时间 | 事件 |
|------|------|
| 2023.02 | LLaMA 1 开源（Meta）——开源大模型元年 |
| 2023.03 | GPT-4 发布（OpenAI） |
| 2023.07 | LLaMA 2 开源 + Claude 2 |
| 2023.09 | Mistral 7B 开源 |
| 2023.11 | GPT-4 Turbo（128K 上下文） |
| 2023.12 | Mixtral 8x7B + Gemini 1.0 |
| 2024.02 | Gemini 1.5 Pro（1M 上下文） |
| 2024.03 | Claude 3 系列 + Grok-1 开源 |
| 2024.04 | LLaMA 3 + Mixtral 8x22B |
| 2024.05 | GPT-4o（原生多模态）+ DeepSeek-V2（MLA） |
| 2024.06 | Claude 3.5 Sonnet（超越 Claude 3 Opus） |
| 2024.07 | LLaMA 3.1（405B, 128K）+ Mistral Large 2 |
| 2024.09 | o1 发布（OpenAI 推理模型） |
| 2024.10 | Claude 3.5 Sonnet 更新 + Computer Use |
| 2024.12 | DeepSeek-V3 + Gemini 2.0 |
| 2025.01 | DeepSeek-R1（开源推理模型） |
| 2025.02 | Grok-3（xAI, 10 万 H100） |
| 2025.03 | Claude 3.7 Sonnet + o3/o3-mini |
| 2025.04 | GPT-4.1 + LLaMA 4 + o4-mini + Gemini 2.5 Pro/Flash + Claude Opus 4/Sonnet 4 |
| 2026 至今 | 多模态 + 推理 + Agent 融合阶段，各家持续迭代 |

---

## Python 代码示例

### 示例 1：使用 HuggingFace 加载开源模型进行推理

以 Qwen 2.5 为例（切换 LLaMA 只需改 model_name）：

```python
# pip install transformers torch accelerate
# GPU 显存需求: 7B 模型约 14GB (FP16) / 4GB (INT4)
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen2.5-7B-Instruct"  # 或 "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
)

# 构造 chat 输入并生成
messages = [
    {"role": "system", "content": "你是一个有帮助的AI助手。"},
    {"role": "user", "content": "解释什么是 Mixture of Experts"},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7, do_sample=True)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

**INT4 量化版**（显存降至 ~4GB）：

```python
from transformers import BitsAndBytesConfig
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4",
    ),
)
```

### 示例 2：GQA 的 PyTorch 实现

```python
"""GQA 实现：展示 MHA/GQA/MQA 的统一框架"""
import torch, torch.nn as nn, torch.nn.functional as F, math

class GroupedQueryAttention(nn.Module):
    """
    n_kv_heads=1        → MQA
    n_kv_heads=n_q_heads → 标准 MHA
    其他值              → GQA
    """
    def __init__(self, d_model: int, n_q_heads: int, n_kv_heads: int):
        super().__init__()
        assert n_q_heads % n_kv_heads == 0
        self.n_q_heads, self.n_kv_heads = n_q_heads, n_kv_heads
        self.d_head = d_model // n_q_heads
        self.n_groups = n_q_heads // n_kv_heads

        self.W_q = nn.Linear(d_model, n_q_heads * self.d_head, bias=False)
        self.W_k = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)   # KV 头更少
        self.W_v = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        B, S, _ = x.shape
        Q = self.W_q(x).view(B, S, self.n_q_heads,  self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, S, self.n_kv_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, S, self.n_kv_heads, self.d_head).transpose(1, 2)

        # 核心: 将 KV 头重复 n_groups 次以匹配 Q 头数
        K = K.unsqueeze(2).expand(-1,-1,self.n_groups,-1,-1).reshape(B,self.n_q_heads,S,self.d_head)
        V = V.unsqueeze(2).expand(-1,-1,self.n_groups,-1,-1).reshape(B,self.n_q_heads,S,self.d_head)

        scores = (Q @ K.transpose(-2,-1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        out = F.softmax(scores, dim=-1) @ V
        return self.W_o(out.transpose(1,2).contiguous().view(B, S, -1))

# 验证: 观察不同配置下 KV 参数量的变化
for name, nq, nkv in [("MHA",32,32), ("GQA-8",32,8), ("GQA-4",32,4), ("MQA",32,1)]:
    attn = GroupedQueryAttention(512, nq, nkv)
    kv_p = sum(p.numel() for n,p in attn.named_parameters() if "W_k" in n or "W_v" in n)
    print(f"{name:8s}  KV params: {kv_p:>8,}  ratio: {nkv/nq:.0%}")
# MHA       KV params:  524,288  ratio: 100%
# GQA-8     KV params:  131,072  ratio: 25%
# GQA-4     KV params:   65,536  ratio: 12%
# MQA       KV params:   16,384  ratio: 3%
```

### 示例 3：MLA 原理的简化实现

```python
"""MLA 简化实现: 展示低秩 KV 压缩的核心思想（省略 RoPE 解耦等工程细节）"""
import torch, torch.nn as nn, torch.nn.functional as F, math

class SimplifiedMLA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_latent: int):
        super().__init__()
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        self.d_latent = d_latent
        self.W_q   = nn.Linear(d_model,  n_heads * self.d_head, bias=False)  # Q 正常投影
        self.W_dkv = nn.Linear(d_model,  d_latent, bias=False)               # 压缩 (Down)
        self.W_uk  = nn.Linear(d_latent, n_heads * self.d_head, bias=False)  # 恢复 K (Up)
        self.W_uv  = nn.Linear(d_latent, n_heads * self.d_head, bias=False)  # 恢复 V (Up)
        self.W_o   = nn.Linear(d_model,  d_model, bias=False)

    def forward(self, x, mask=None):
        B, S, _ = x.shape
        reshape = lambda t, h: t.view(B, S, h, self.d_head).transpose(1, 2)
        Q = reshape(self.W_q(x), self.n_heads)

        c = self.W_dkv(x)                          # (B, S, d_latent) ← KV Cache 只存这个!
        K = reshape(self.W_uk(c), self.n_heads)     # 从 c 恢复 K
        V = reshape(self.W_uv(c), self.n_heads)     # 从 c 恢复 V

        scores = (Q @ K.transpose(-2,-1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        out = (F.softmax(scores, -1) @ V).transpose(1,2).contiguous().view(B, S, -1)
        return self.W_o(out), c

# 对比 KV Cache 大小: d_model=4096, n_heads=32, d_head=128
mha_cache = 2 * 32 * 128   # 标准 MHA: 存 K + V = 8192 floats/token
mla_cache = 512             # MLA: 只存压缩向量 c_t = 512 floats/token
print(f"MHA: {mha_cache}  MLA: {mla_cache}  节省: {1 - mla_cache/mha_cache:.1%}")
# MHA: 8192  MLA: 512  节省: 93.8%
```

---

## 工程师视角

### 开源 vs 闭源的选择

| 维度 | 闭源 API | 开源部署 |
|------|---------|---------|
| **最强性能** | 通常领先（GPT-4o, Claude Opus 4） | 接近（LLaMA 3.1 405B, DeepSeek-V3） |
| **数据隐私** | 数据发送到第三方 | 完全自控 |
| **定制化** | 有限（微调 API） | 完全自由（LoRA/全量微调） |
| **成本结构** | 按 token 计费，线性增长 | 固定 GPU 成本，规模效益 |
| **运维** | 零 | 需要 GPU 集群管理 |

**经验法则**：日调用 < 1 万次用闭源 API；> 10 万次考虑开源部署；数据敏感必选开源；推理任务用 o3/R1。

### 模型选型指南（2026.4 快照）

| 场景 | 推荐模型 | 理由 |
|------|---------|------|
| 通用对话 | GPT-4o / Claude Sonnet 4 | 性价比最优的闭源选择 |
| 复杂推理 | o3 / Claude Opus 4 | 数学、代码竞赛级任务 |
| 代码生成 | GPT-4.1 / Claude Sonnet 4 | 代码专项优化 |
| 开源部署（性能优先） | DeepSeek-V3 / LLaMA 3.1 405B | 最强开源选择 |
| 开源部署（效率优先） | Qwen 2.5-7B / LLaMA 3.1-8B | 单卡可跑 |
| 开源推理 | DeepSeek-R1 / QwQ-32B | 推理能力强的开源模型 |
| 多模态 | GPT-4o / Gemini 2.5 Pro | 图像+文本+音频 |
| 超长上下文 | Gemini 2.5 Pro / GPT-4.1 | 1M token 级别 |
| 中文场景 | Qwen 2.5 / DeepSeek-V3 | 中文训练数据充足 |
| 边缘设备 | Phi-4 / Qwen 2.5-1.5B | 极小模型 |

> **注意**：以上推荐是截至 2026.4 的快照。模型排名可能在数月内显著变化。

### 评测基准速览

工程师需要理解常见基准的含义，以正确解读模型评测结果：

| 基准 | 测试什么 | 满分 | 说明 |
|------|---------|------|------|
| **MMLU** | 多学科知识 | 100% | 57 个学科的多选题，测试"知识广度" |
| **HumanEval** | 代码生成 | 100% | 164 个 Python 编程题，pass@1 |
| **GSM8K** | 数学推理 | 100% | 8.5K 小学数学应用题 |
| **MATH** | 高等数学 | 100% | 竞赛级数学题 |
| **ARC-AGI** | 抽象推理 | 100% | 视觉图案推理，测试泛化能力 |
| **AIME** | 数学竞赛 | 30 | 美国数学邀请赛题目 |
| **Codeforces** | 编程竞赛 | ELO 分 | 竞赛编程排名 |
| **MT-Bench** | 对话质量 | 10 | 多轮对话打分 |
| **Arena ELO** | 人类偏好 | ELO 分 | LMSYS Chatbot Arena 人类盲评 |

**注意**：不要只看单一基准；关注评测协议（0-shot vs 5-shot 差异大）；Arena ELO 最接近真实偏好；警惕基准污染（训练数据包含测试题导致分数虚高）。

---

## 时效性说明

- **核心架构原理**（GQA/MLA/MoE/推理范式）：有效期 **2~3 年以上**
- **具体模型排名**：有效期 **3~6 个月**，大模型领域变化极快
- **代码示例**：HuggingFace API 可能在 1~2 年内变化

本文内容截至 **2026 年 4 月**。建议通过 [LMSYS Chatbot Arena](https://chat.lmsys.org/) 和 [HuggingFace Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) 查看最新排名。

---

## 本章小结

1. **开源追赶**：LLaMA/Mistral/DeepSeek/Qwen 在 2024~2025 接近甚至部分超越闭源
2. **多模态融合**：GPT-4V(插件) → GPT-4o(原生) → LLaMA 4(开源)，趋势是 early fusion
3. **推理模型**：o1 → o3 → R1 → QwQ，核心是 test-time compute scaling
4. **效率创新**：GQA(共享KV头) → MLA(低秩压缩,省93%) → MoE(稀疏激活)
5. **上下文扩展**：4K → 128K → 1M → 10M tokens

> **下一步学习**：
> - 模型训练细节 → [07 大模型训练技术](../AI学习_07_大模型训练技术/README.md)
> - 推理部署优化 → [08 大模型推理与部署](../AI学习_08_大模型推理与部署/README.md)
> - 推理模型深入 → [11 推理与思维链](../AI学习_11_推理与思维链/README.md)
> - 多模态深入 → [10 多模态模型](../AI学习_10_多模态模型/README.md)
