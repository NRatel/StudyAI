# Mixture of Experts（MoE）架构

> **前置知识**：[Transformer 架构](../AI学习_03_注意力机制与Transformer/3_Transformer架构.md)（特别是 FFN 层的结构与参数量估算）
>
> **内容时效**：截至 2026 年 4 月

---

## 为什么只激活部分专家
### 核心问题

> **关键澄清**：MoE 经常被描述为"训练省算力"，但这个说法需要精确理解。MoE 的"训练省算力"指的是：**在同样的计算预算下，可以训练参数量更大的模型**（因为每个 token 只激活少数专家，FLOPs 与激活参数成正比而非总参数）。但在推理时，**显存占用并不省**——所有专家的参数都要加载到 GPU 内存中，即使每个 token 只用到其中几个。例如 DeepSeek-V3 总参数 671B，推理时每个 token 只激活 37B 的计算量，但 671B 的参数必须全部驻留在内存中。这意味着 MoE 模型对推理硬件的**内存要求**并不比同总参数的 Dense 模型低，它节省的是**计算量**（FLOPs），不是**内存**。

大模型的性能与参数量高度正相关。但参数越多，推理计算量越大。有没有一种方法能**拥有巨大的参数量，但推理时只使用一小部分**？

这就是 MoE 的核心思想：一个模型包含许多"专家"（Expert），但每个输入只激活其中少数几个。

{{img:ch13_02_moe_router_experts}}

### 直觉类比

想象一家大型医院：

- **Dense 模型**（标准 Transformer）：只有一个全科医生，所有病人都由这一个医生看全部科目
- **MoE 模型**：医院有 64 个专科医生（专家），每个病人根据症状被分配到 2 个最相关的专科医生

结果：医院的总"专业知识"（参数量）是全科医生的 64 倍，但每个病人的就诊时间（推理计算量）只相当于 2 个医生的工作量。

### MoE 模型一览

| 模型 | 总参数量 | 激活参数量 | 专家数 | 每 token 激活 | 发布时间 |
|------|----------|-----------|--------|--------------|---------|
| Switch Transformer | 1.6T | ~T5-Base | 128 | 1 | 2021.01 |
| Mixtral 8x7B | 46.7B | 12.9B | 8 | 2 | 2023.12 |
| DeepSeek-V2 | 236B | 21B | 160 | 6 | 2024.05 |
| DeepSeek-V3 | 671B | 37B | 256 | 8 | 2024.12 |
| Qwen1.5-MoE-A2.7B | 14.3B | 2.7B | 60 | 4 | 2024.03 |

**关键数据**：DeepSeek-V3 总参数 671B，每个 token 仅激活 37B（约 5.5%），用接近 Llama-2-70B 的推理成本获得远超之的性能。

---

## MoE 的结构、路由与负载均衡
### MoE 的基本结构

MoE 替换的是 Transformer 中的 **FFN 层**（注意力层保持不变）：

```
标准 Transformer 层:           MoE Transformer 层:
  Self-Attention                 Self-Attention         (所有 token 共享)
       ↓                              ↓
  单个 FFN                        路由器 (Router)
  (所有 token 共享)                    ↓
                              ┌──┬──┬──┬──┬──┬──┬──┬──┐
                              │E₀│E₁│E₂│E₃│E₄│E₅│E₆│E₇│  ← N 个 Expert FFN
                              └──┴──┴──┴──┴──┴──┴──┴──┘
                              每个 token 只激活其中 Top-K 个
```

每个 Expert 就是一个独立的 FFN，结构与标准 FFN 完全相同：

$$\text{Expert}_i(x) = W_{2,i} \cdot \text{Activation}(W_{1,i} \cdot x) + b_{2,i}$$

### 路由机制（Router / Gate）

路由器是一个简单的线性层，输出每个专家的"相关度"得分：

$$\text{scores}(x) = W_g \cdot x \in \mathbb{R}^{N}$$

$$\text{gates}(x) = \text{Softmax}(\text{scores}(x))$$

选择得分最高的 Top-K 个专家，用归一化后的权重对专家输出做加权求和：

$$\text{output}(x) = \sum_{i \in \text{TopK}} \hat{g}_i(x) \cdot \text{Expert}_i(x)$$

其中 $\hat{g}_i$ 是 Top-K 内部重新归一化后的权重。

**一个具体例子（8 专家，Top-2）**：

```
Token "猫" → Router → [0.05, 0.02, 0.41, 0.03, 0.01, 0.30, 0.06, 0.12]
                              ↑                       ↑
                        Expert 2 (0.41)          Expert 5 (0.30)
                        归一化后: 0.577            归一化后: 0.423

output = 0.577 * Expert_2("猫") + 0.423 * Expert_5("猫")
```

### Switch Transformer（Google, Fedus et al. 2021）

Switch Transformer 是首个成功扩展到超大规模的 MoE 模型。核心简化：**Top-1 路由**——每个 token 只送给 1 个专家。

**为什么 Top-1 就够了？** 虽然每个 token 只经过一个专家，但不同 token 被路由到不同专家。句子中的不同 token 分别利用不同专家的知识，整体上模型仍然调动了多个专家。同时 Top-1 避免了 All-to-All 通信的复杂性，大幅简化了分布式训练。

### Mixtral 8x7B（Mistral AI, 2023）

Mixtral 是首个性能比肩 GPT-3.5 的开源 MoE 模型：

| 设计选择 | 值 | 说明 |
|----------|-----|------|
| 专家数 | 8 | 每层 8 个独立 FFN |
| Top-K | 2 | 每个 token 激活 2 个专家 |
| 单专家规模 | 与 Mistral-7B 的 FFN 相同 | 因此叫 "8x7B" |
| 总参数 | 46.7B | 8 个 FFN + 共享注意力层 |
| 激活参数 | 12.9B | 2 个 FFN + 注意力层 |

Mixtral 的实际性能接近 Llama-2-70B，但推理速度快数倍——因为每个 token 只激活 12.9B 参数。

### DeepSeekMoE（2024-2025）

DeepSeek 团队在 MoE 架构上做出了两项关键创新：

#### 创新 1：细粒度专家（Fine-Grained Experts）

将大专家拆成更多小专家，提供更灵活的组合能力：

```
Mixtral:      8 个大专家, 选 2 个  → 组合方式 C(8,2) = 28 种
DeepSeek-V2: 160 个小专家, 选 6 个 → 组合方式 C(160,6) ≈ 2.1 亿种
```

小专家 + 更多选择 = 更精细的知识调度。

#### 创新 2：共享专家（Shared Experts）

保留 1~2 个"共享专家"，所有 token 都经过它们：

$$\text{output} = \underbrace{\text{SharedExpert}(x)}_{\text{通用知识}} + \underbrace{\sum_{i \in \text{TopK}} g_i \cdot \text{Expert}_i(x)}_{\text{专业知识}}$$

共享专家学通用知识（语法、常识），路由专家学专业知识。避免每个路由专家都冗余地学习相同的通用能力。

### 负载均衡——MoE 的核心挑战

如果路由器将大多数 token 都送给少数"热门"专家，会导致：
1. 其他专家得不到训练信号（参数浪费）
2. 热门专家成为计算瓶颈
3. 分布式训练中持有热门专家的 GPU 过载

#### 辅助负载均衡损失（Switch Transformer）

在训练损失上加一个辅助项，惩罚负载不均：

$$\mathcal{L}_{\text{balance}} = \alpha \cdot N \sum_{i=1}^{N} f_i \cdot p_i$$

- $f_i$：batch 中被分配到专家 $i$ 的 token 比例
- $p_i$：路由器分配给专家 $i$ 的平均概率
- 当负载完全均匀时（$f_i = p_i = 1/N$），损失取最小值
- $\alpha$ 通常设为 0.01，确保不破坏主任务损失

#### DeepSeek-V3 的无辅助损失负载均衡（2024）

DeepSeek-V3 提出更优雅的方案：给每个专家一个偏置项 $b_i$，加到路由 logits 上，但 $b_i$ **不参与梯度更新**，而是用简单规则调整：

- 专家 $i$ 负载过高 → 减小 $b_i$
- 专家 $i$ 负载过低 → 增大 $b_i$

优势：不引入额外的损失函数，不需要调 $\alpha$ 超参数，负载均衡与主任务训练完全解耦。

---

## 时效性说明

本文内容截至 **2026 年 4 月**。以下方面仍在快速演进：

- **路由策略**：Top-K 值、共享专家比例、路由算法持续优化
- **负载均衡**：DeepSeek-V3 的无辅助损失方法可能催生更多新方案
- **MoE 推理优化**：专家卸载（offloading 到 CPU/SSD）、专家剪枝/合并正在发展
- **端侧 MoE**：如何在有限硬件高效运行 MoE 是 2025-2026 热点方向

**建议**：关注 DeepSeek 和 Mistral AI 的技术博客获取 MoE 最新进展。

**下一章**：[高效Transformer](3_高效Transformer.md)——除了 MoE（稀疏参数），还有哪些方法可以从根本上降低注意力机制的计算复杂度（稀疏/线性计算）？
