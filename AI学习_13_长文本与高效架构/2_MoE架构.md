# Mixture of Experts（MoE）架构

> **前置知识**：[Transformer 架构](../AI学习_03_注意力机制与Transformer/3_Transformer架构.md)（特别是 FFN 层的结构与参数量估算）
>
> **内容时效**：截至 2026 年 4 月

---

## 直觉与概述

### 核心问题

> **常见误区前置**：MoE 经常被描述为"训练省算力"，但这个说法需要精确理解。MoE 的"训练省算力"指的是：**在同样的计算预算下，可以训练参数量更大的模型**（因为每个 token 只激活少数专家，FLOPs 与激活参数成正比而非总参数）。但在推理时，**显存占用并不省**——所有专家的参数都要加载到 GPU 内存中，即使每个 token 只用到其中几个。例如 DeepSeek-V3 总参数 671B，推理时每个 token 只激活 37B 的计算量，但 671B 的参数必须全部驻留在内存中。这意味着 MoE 模型对推理硬件的**内存要求**并不比同总参数的 Dense 模型低，它节省的是**计算量**（FLOPs），不是**内存**。

大模型的性能与参数量高度正相关。但参数越多，推理计算量越大。有没有一种方法能**拥有巨大的参数量，但推理时只使用一小部分**？

这就是 MoE 的核心思想：一个模型包含许多"专家"（Expert），但每个输入只激活其中少数几个。

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

## 严谨定义与原理

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

## Python 代码示例

### 示例 1：Top-K MoE 层的完整实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    """单个专家 = 标准 FFN。"""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)


class TopKMoELayer(nn.Module):
    """
    Top-K MoE 层：替换 Transformer 中的 FFN。
    支持可选的共享专家（DeepSeek 风格）。
    """
    def __init__(self, dim, hidden_dim, num_experts=8, top_k=2,
                 num_shared_experts=0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # 路由器
        self.router = nn.Linear(dim, num_experts, bias=False)

        # 路由专家
        self.experts = nn.ModuleList(
            [Expert(dim, hidden_dim) for _ in range(num_experts)]
        )

        # 共享专家（可选）
        self.shared_experts = nn.ModuleList(
            [Expert(dim, hidden_dim) for _ in range(num_shared_experts)]
        )

    def forward(self, x):
        B, S, D = x.shape
        flat_x = x.view(-1, D)  # (B*S, D)

        # 1. 路由决策
        logits = self.router(flat_x)             # (B*S, num_experts)
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_idx = probs.topk(self.top_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        # 2. 分发到各专家并计算
        output = torch.zeros_like(flat_x)
        for k in range(self.top_k):
            idx_k = topk_idx[:, k]
            weight_k = topk_probs[:, k]
            for e in range(self.num_experts):
                mask = (idx_k == e)
                if mask.any():
                    tokens_in = flat_x[mask]
                    tokens_out = self.experts[e](tokens_in)
                    output[mask] += weight_k[mask].unsqueeze(-1) * tokens_out

        # 3. 共享专家
        for shared in self.shared_experts:
            output = output + shared(flat_x)

        # 4. 负载均衡辅助损失
        aux_loss = self._balance_loss(probs, topk_idx)

        return output.view(B, S, D), aux_loss

    def _balance_loss(self, probs, topk_idx):
        """Switch Transformer 风格的负载均衡损失。"""
        N = self.num_experts
        num_tokens = probs.shape[0]
        # f_i: 每个专家实际被分配的 token 比例
        one_hot = F.one_hot(topk_idx, N).float()
        f = one_hot.sum(dim=(0, 1)) / (num_tokens * self.top_k)
        # p_i: 路由器输出的平均概率
        p = probs.mean(dim=0)
        return N * (f * p).sum()


# --- 演示 ---
torch.manual_seed(42)
dim, hidden, bs, seq = 64, 256, 2, 16

moe = TopKMoELayer(dim, hidden, num_experts=8, top_k=2, num_shared_experts=1)
x = torch.randn(bs, seq, dim)
out, loss = moe(x)

print(f"输入: {list(x.shape)}, 输出: {list(out.shape)}")
print(f"负载均衡损失: {loss.item():.4f}")

# 参数量对比
moe_params = sum(p.numel() for p in moe.parameters())
dense_params = dim * hidden * 2 + dim + hidden  # 单个 FFN（近似）
print(f"\nMoE 总参数: {moe_params:,}")
print(f"Dense FFN 参数: {dense_params:,}")
print(f"参数量比: {moe_params / dense_params:.1f}x")
print(f"但每 token 只用 2+1=3 个专家, 计算量约 Dense 的 3x")

# 路由分布
with torch.no_grad():
    logits = moe.router(x.view(-1, dim))
    idx = F.softmax(logits, dim=-1).topk(2, dim=-1).indices
    counts = torch.bincount(idx.flatten(), minlength=8)
    print(f"\n路由分布:")
    for i, c in enumerate(counts):
        bar = "#" * c.item()
        print(f"  Expert {i}: {c.item():>3d} tokens  {bar}")
```

### 示例 2：负载均衡分析——均匀 vs 不均匀路由

```python
import torch
import torch.nn.functional as F

def analyze_routing(probs, num_experts, top_k, capacity_factor=1.25):
    """分析路由分布和 token dropping 情况。"""
    num_tokens = probs.shape[0]
    _, topk_idx = probs.topk(top_k, dim=-1)

    # 每个专家的容量上限
    capacity = int(capacity_factor * num_tokens * top_k / num_experts)

    # 模拟分配
    load = torch.zeros(num_experts, dtype=torch.long)
    dropped = 0
    for t in range(num_tokens):
        for k in range(top_k):
            eid = topk_idx[t, k].item()
            if load[eid] < capacity:
                load[eid] += 1
            else:
                dropped += 1

    drop_rate = dropped / (num_tokens * top_k) * 100
    return load, drop_rate

# --- 场景对比 ---
torch.manual_seed(42)
N_tok, N_exp, K = 1024, 8, 2

# 场景 1: 均匀路由
uniform = F.softmax(torch.randn(N_tok, N_exp) * 0.1, dim=-1)

# 场景 2: 严重倾斜路由（专家 0 和 1 热门）
skewed_logits = torch.randn(N_tok, N_exp)
skewed_logits[:, 0] += 3.0
skewed_logits[:, 1] += 2.0
skewed = F.softmax(skewed_logits, dim=-1)

print("路由均衡性分析")
print("=" * 55)
for name, probs in [("均匀路由", uniform), ("严重倾斜", skewed)]:
    load, drop = analyze_routing(probs, N_exp, K, capacity_factor=1.25)
    print(f"\n--- {name} (容量因子=1.25) ---")
    print(f"  Token Drop 率: {drop:.1f}%")
    print(f"  负载 std: {load.float().std():.1f}")
    print(f"  负载分布:")
    for i in range(N_exp):
        bar = "#" * (load[i].item() // 10)
        print(f"    Expert {i}: {load[i].item():>4d} {bar}")

    # 计算辅助损失
    f = torch.zeros(N_exp)
    _, idx = probs.topk(K, dim=-1)
    for i in range(N_exp):
        f[i] = (idx == i).float().sum() / (N_tok * K)
    p = probs.mean(dim=0)
    aux = N_exp * (f * p).sum().item()
    print(f"  辅助损失 L_balance: {aux:.4f} (理想值=1.0)")
```

### 示例 3：MoE 参数量与计算量估算

```python
def estimate_moe_model(
    d_model: int,
    n_layers: int,
    n_experts: int,
    top_k: int,
    n_shared_experts: int = 0,
    d_ff_ratio: float = 4.0,
    vocab_size: int = 32000,
    name: str = "Model"
):
    """估算 MoE 模型的参数量和计算量。"""
    d_ff = int(d_model * d_ff_ratio)

    # 注意力层参数（所有 token 共享）
    attn_per_layer = 4 * d_model * d_model  # Q, K, V, O

    # FFN 专家参数
    ffn_per_expert = 2 * d_model * d_ff  # W1 + W2
    moe_per_layer = n_experts * ffn_per_expert
    shared_per_layer = n_shared_experts * ffn_per_expert

    # 路由器参数
    router_per_layer = d_model * n_experts

    # 总参数
    per_layer = attn_per_layer + moe_per_layer + shared_per_layer + router_per_layer
    total = per_layer * n_layers + vocab_size * d_model  # + embedding

    # 激活参数（每 token）
    active_ffn = (top_k + n_shared_experts) * ffn_per_expert
    active_per_layer = attn_per_layer + active_ffn
    active_total = active_per_layer * n_layers + vocab_size * d_model

    print(f"\n{'='*50}")
    print(f"{name}")
    print(f"{'='*50}")
    print(f"  d_model={d_model}, layers={n_layers}")
    print(f"  experts={n_experts}, top_k={top_k}, shared={n_shared_experts}")
    print(f"  总参数量:    {total/1e9:.1f}B")
    print(f"  激活参数量:  {active_total/1e9:.1f}B")
    print(f"  激活比例:    {active_total/total*100:.1f}%")
    print(f"  参数效率:    {total/active_total:.1f}x (总/激活)")

# --- 对比几个代表模型 ---
# Mixtral 8x7B (近似)
estimate_moe_model(4096, 32, 8, 2, 0, 3.5, 32000, "Mixtral 8x7B (近似)")

# DeepSeek-V2 (简化)
estimate_moe_model(5120, 60, 160, 6, 2, 1.5, 100000, "DeepSeek-V2 (简化)")

# Dense 对比: Llama-2-7B
d = 4096
layers = 32
dense_total = 12 * layers * d * d + 32000 * d
print(f"\n{'='*50}")
print(f"Dense 对比: Llama-2-7B (近似)")
print(f"{'='*50}")
print(f"  总参数 = 激活参数 = {dense_total/1e9:.1f}B")
print(f"  激活比例: 100% (Dense 模型每个 token 用全部参数)")
```

---

## 工程师视角

### MoE 的核心优缺点

| 优点 | 缺点 |
|------|------|
| 总参数大 → 知识容量大 | 总参数大 → 显存占用大（需加载全部专家） |
| 激活参数少 → 推理计算高效 | 小 batch 时，内存带宽（而非计算）成为瓶颈 |
| 易于通过增加专家数扩展 | 路由不均会浪费参数 |
| 已被 GPT-4/Mixtral/DeepSeek 验证 | 分布式训练需 Expert Parallelism，通信复杂 |

### 专家并行（Expert Parallelism）

MoE 引入了一种特有的并行策略：

```
传统并行:
  数据并行(DP):   每 GPU 持有完整模型，处理不同 batch
  张量并行(TP):   每 GPU 持有模型的部分参数（矩阵横切或纵切）
  流水线并行(PP): 每 GPU 持有不同的层

MoE 特有:
  专家并行(EP):   每 GPU 持有不同的专家
    GPU 0: Expert 0~3     GPU 1: Expert 4~7
    Token 路由到哪个专家 → 发送到对应 GPU → 计算 → 发回结果
    通信模式: All-to-All（每个 GPU 同时向所有其他 GPU 收发数据）
```

**All-to-All** 是 MoE 分布式训练的主要通信瓶颈。DeepSeek-V3 通过限制每个 token 最多被路由到有限数量的节点（而非任意节点），减少了跨节点通信。

### 什么时候选 MoE vs Dense

```
场景适合 MoE 吗？
├─ GPU 显存充足？（需要加载全部专家参数）
│   ├─ 否 → 用 Dense 模型更实际
│   └─ 是
│       ├─ 高 QPS / 大 batch？
│       │   └─ MoE 优势明显（大容量 + 高吞吐）
│       └─ 低 QPS / 小 batch？
│           └─ MoE 的计算优势被内存带宽抵消，Dense 可能更合适
└─ 追求极致性价比？
    └─ MoE（如 DeepSeek-V3：671B 总参数，训练成本仅 $5.6M）
```

### 2025-2026 重要进展

| 进展 | 时间 | 关键点 |
|------|------|--------|
| DeepSeek-V3 | 2024.12 | 671B/37B，无辅助损失均衡，多 Token 预测，训练仅 $5.6M |
| Mixtral → Mistral Large 2 | 2024-2025 | MoE 在中等规模的工程实践 |
| Qwen-MoE 系列 | 2024-2025 | 14.3B (A2.7B) 到更大规模，细粒度 + 共享专家 |
| 端侧 MoE | 2025-2026 | 小规模 MoE 在手机/边缘设备上的部署探索 |
| MoE + 推理链 | 2025 | DeepSeek-R1 使用 MoE 架构进行长链推理 |

---

## 本章小结

| 概念 | 一句话总结 |
|------|-----------|
| **MoE 核心思想** | 多专家、稀疏激活——总参数大但每 token 只用少数几个 |
| **路由器** | 线性层 + Softmax，决定每个 token 去哪些专家 |
| **Top-K 路由** | 选概率最高的 K 个专家，加权求和 |
| **Switch Transformer** | Top-1 路由，首次成功扩展到 1T+ 规模 |
| **Mixtral** | 8 专家选 2，首个媲美 GPT-3.5 的开源 MoE |
| **DeepSeekMoE** | 细粒度专家 + 共享专家，组合灵活度大幅提升 |
| **负载均衡** | MoE 核心挑战，通过辅助损失或自适应偏置解决 |
| **Expert Parallelism** | MoE 特有的分布式策略，All-to-All 通信为主要瓶颈 |

---

## 时效性说明

本文内容截至 **2026 年 4 月**。以下方面仍在快速演进：

- **路由策略**：Top-K 值、共享专家比例、路由算法持续优化
- **负载均衡**：DeepSeek-V3 的无辅助损失方法可能催生更多新方案
- **MoE 推理优化**：专家卸载（offloading 到 CPU/SSD）、专家剪枝/合并正在发展
- **端侧 MoE**：如何在有限硬件高效运行 MoE 是 2025-2026 热点方向

**建议**：关注 DeepSeek 和 Mistral AI 的技术博客获取 MoE 最新进展。

**下一章**：[高效Transformer](3_高效Transformer.md)——除了 MoE（稀疏参数），还有哪些方法可以从根本上降低注意力机制的计算复杂度（稀疏/线性计算）？
