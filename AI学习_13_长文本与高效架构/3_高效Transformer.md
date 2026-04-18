# 高效 Transformer 变体

> **前置知识**：[注意力机制](../AI学习_03_注意力机制与Transformer/1_注意力机制.md)、[Transformer 架构](../AI学习_03_注意力机制与Transformer/3_Transformer架构.md)
>
> **内容时效**：截至 2026 年 4 月

---

## 直觉与概述

### 核心问题

前一章讨论了如何让标准 Transformer 支持更长的序列（位置编码外推、序列并行、压缩记忆）。但这些方案都没有改变一个根本事实：**标准注意力的计算量是 O(n^2)**。

本章探讨另一条路：**从注意力机制本身下手**，设计计算复杂度更低的替代方案。

### 三大类方案

| 类别 | 核心思路 | 复杂度 | 代表模型 |
|------|----------|--------|----------|
| **稀疏注意力** | 不让每个 token 关注所有 token，只关注一部分 | $O(n \sqrt{n})$ 或 $O(n \log n)$ | Longformer, BigBird |
| **线性注意力** | 用核函数近似 softmax，将 $QK^T$ 分解为 $Q(K^T V)$ | $O(n \cdot d^2)$ | Performer |
| **状态空间模型 (SSM)** | 完全抛弃注意力，用递归状态方程建模序列 | $O(n \cdot d)$ | S4, Mamba, RWKV |

### 直觉对比

```
标准注意力:  每个 token 都"看"所有其他 token
             n=1000 时，100 万次交互
             ↓
稀疏注意力:  每个 token 只看"附近的"和"远处采样的"token
             n=1000 时，约 3-5 万次交互
             ↓
线性注意力:  用数学技巧避免显式计算 n×n 矩阵
             n=1000 时，不需要 n×n 矩阵
             ↓
SSM (Mamba): 完全不用注意力，用隐状态逐步处理序列
             类似"高效版 RNN"，但可以并行训练
```

---

## 严谨定义与原理

### 一、稀疏注意力

#### Longformer（Beltagy et al., Allen AI, 2020）

**核心思想**：绝大多数 NLP 任务中，token 主要关注自己附近的上下文，偶尔关注远距离的特殊 token。Longformer 用两种稀疏模式组合来近似全注意力：

```
全注意力矩阵 (n=12):          Longformer 稀疏模式:
■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■      ■ ■ ■ · · · · · · · · ■   ← [CLS] 做全局
■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■      ■ ■ ■ ■ · · · · · · · ·
■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■      ■ ■ ■ ■ ■ · · · · · · ·
■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■      · ■ ■ ■ ■ ■ · · · · · ·   ← 滑动窗口
■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■      · · ■ ■ ■ ■ ■ · · · · ·      (局部注意力)
■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■      · · · ■ ■ ■ ■ ■ · · · ·
■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■      · · · · ■ ■ ■ ■ ■ · · ·
■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■      · · · · · ■ ■ ■ ■ ■ · ·
■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■      · · · · · · ■ ■ ■ ■ ■ ·
■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■      · · · · · · · ■ ■ ■ ■ ■
■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■      · · · · · · · · ■ ■ ■ ■
■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■      ■ · · · · · · · · ■ ■ ■
O(n^2)                         O(n * w)
■=计算注意力  ·=跳过
```

两种注意力模式：

1. **滑动窗口（Local）注意力**：每个 token 只关注左右 $w/2$ 个 token（窗口大小 $w$）。复杂度 $O(n \cdot w)$。
2. **全局（Global）注意力**：少数特殊 token（如 [CLS]、问题 token）可以关注所有 token。这些全局 token 起到"信息中继站"的作用。

**效果**：$L$ 层堆叠后，信息可以通过窗口的逐层扩展传播到整个序列。感受野 = $L \times w$。

#### BigBird（Google, 2020）

BigBird 在 Longformer 的基础上加入了**随机注意力**：

| 模式 | 描述 | 作用 |
|------|------|------|
| 滑动窗口 | 与 Longformer 相同 | 捕捉局部依赖 |
| 全局 token | 少数 token 关注所有 | 信息汇聚和传播 |
| 随机连接 | 每个 token 额外随机关注 $r$ 个 token | 缩短任意两点间的"最短路径" |

随机连接的理论动机来自图论：在一个规则图中加入少量随机边，可以将任意两点间的平均路径长度从 $O(n)$ 降到 $O(\log n)$——这就是"小世界网络"效应。

### 二、线性注意力

#### 核心思想：重排矩阵乘法顺序

标准注意力：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) V$$

关键瓶颈：$QK^T$ 产生 $n \times n$ 矩阵。

如果我们把 softmax 替换为一个可分解的核函数 $\phi$：

$$\text{softmax}(QK^T) \approx \phi(Q) \cdot \phi(K)^T$$

那么计算顺序可以改变：

$$\text{Attention} = \phi(Q) \cdot \underbrace{(\phi(K)^T \cdot V)}_{\text{先算这个！}} $$

先计算 $\phi(K)^T V$，结果是 $d \times d$ 矩阵（与 $n$ 无关！），再乘以 $\phi(Q)$。总复杂度从 $O(n^2 d)$ 降到 $O(n d^2)$。

**当 $d \ll n$ 时（长序列场景），这就是线性复杂度**。

#### Performer（Google, Choromanski et al. 2020）

Performer 使用随机特征（Random Features）来近似 softmax 核：

$$\phi(x) = \frac{1}{\sqrt{m}} \left[\exp(w_1^T x - \|x\|^2/2), \ldots, \exp(w_m^T x - \|x\|^2/2)\right]$$

其中 $w_1, \ldots, w_m$ 是从标准正态分布采样的随机向量。

**问题**：近似质量依赖随机特征数量 $m$，实际中 $m$ 需要很大才能接近标准 softmax，导致常数因子大。因此 Performer 在实际应用中未被广泛采用。

### 三、状态空间模型（SSM）

#### 从 RNN 到 SSM 的演进

回顾 RNN 的核心公式：$h_t = f(h_{t-1}, x_t)$。RNN 的问题是非线性递归导致无法并行训练。

SSM 的核心洞察：如果递归是**线性**的，就可以展开为卷积，从而并行计算。

**连续时间 SSM**（线性状态方程）：

$$h'(t) = Ah(t) + Bx(t), \quad y(t) = Ch(t)$$

离散化后：

$$h_t = \bar{A} h_{t-1} + \bar{B} x_t, \quad y_t = C h_t$$

其中 $\bar{A}, \bar{B}$ 是连续参数 $A, B$ 的离散化版本。

**关键特性**：由于是线性递归，整个序列的输出可以展开为一个卷积：

$$y = \bar{K} * x, \quad \bar{K} = (C\bar{B}, C\bar{A}\bar{B}, C\bar{A}^2\bar{B}, \ldots)$$

训练时用 FFT 加速卷积，$O(n \log n)$；推理时用递归，$O(1)$ 每步。

#### S4（Structured State Space, Gu et al. 2022）

S4 是 SSM 的关键突破。核心贡献：发现矩阵 $A$ 的初始化方式极其重要。使用 **HiPPO 矩阵**（一种特殊结构的矩阵）初始化 $A$，可以让状态 $h$ 有效地记忆长期历史。

S4 的 $A$ 矩阵结构使得状态 $h$ 在数学上等价于对历史输入做**正交多项式基的最优逼近**——直觉上，它在"用有限的状态维度尽可能准确地记住过去"。

#### Mamba（Gu & Dao, 2023）

Mamba 是 SSM 领域最重要的工作，首次让 SSM 在语言建模上比肩同规模的 Transformer。核心创新：**选择性 SSM（Selective SSM）**。

**S4 的局限**：$A, B, C$ 是固定参数，对所有输入相同。这意味着 S4 无法根据输入内容动态决定"记住什么、忘记什么"。

**Mamba 的解决方案**：让 $B, C$ 和步长 $\Delta$ 成为输入的函数：

$$B_t = \text{Linear}_B(x_t), \quad C_t = \text{Linear}_C(x_t), \quad \Delta_t = \text{softplus}(\text{Linear}_\Delta(x_t))$$

这样模型可以根据当前输入**选择性地**决定：
- $\Delta_t$ 大：多关注当前输入（"这个信息重要，要记住"）
- $\Delta_t$ 小：多保留历史状态（"当前输入不重要，保持记忆"）

**硬件优化**：Mamba 配合了类似 FlashAttention 的 IO-aware 计算策略，在 GPU 上实现了高效的选择性扫描（selective scan）。

#### Mamba-2（Dao & Gu, 2024）

Mamba-2 揭示了 SSM 与注意力之间更深层的联系：

- 将 Mamba 的选择性 SSM 重新表述为一种**半可分离矩阵**（semiseparable matrix）的结构化矩阵乘法
- 证明了线性注意力实际上是 SSM 的一个特例
- 利用这种等价性设计了更高效的 GPU 算法，速度比 Mamba-1 提升 2-8 倍

#### RWKV（Peng et al., 2023-2025）

RWKV 是另一条"去注意力"路线，名字代表其四个核心向量：Receptance、Weight、Key、Value。

**核心思想**：用**线性递归**替代注意力，但保留类似 attention 的 QKV 交互形式：

$$wkv_t = \frac{\sum_{i=1}^{t} e^{-(t-i)w + k_i} v_i}{\sum_{i=1}^{t} e^{-(t-i)w + k_i}}$$

其中 $w$ 是可学习的衰减权重。这可以用递归高效计算（$O(1)$ 每步），同时保留了对历史 token 的加权聚合能力。

**RWKV 演进**：RWKV-4 (2023) → RWKV-5/Eagle (2024, 引入多头) → RWKV-6/Finch (2024, 14B 接近同规模 Transformer) → RWKV-7 (2025, 持续优化)。

---

## Python 代码示例

### 示例 1：稀疏注意力（滑动窗口）的实现

```python
import torch
import torch.nn.functional as F
import math

def sliding_window_attention(Q, K, V, window_size=4):
    """
    滑动窗口注意力：每个 token 只关注左右 window_size//2 个 token。
    复杂度 O(n * w)，其中 w 是窗口大小。
    """
    n, d = Q.shape
    half_w = window_size // 2

    # 构建稀疏掩码
    mask = torch.zeros(n, n, dtype=torch.bool)
    for i in range(n):
        lo = max(0, i - half_w)
        hi = min(n, i + half_w + 1)
        mask[i, lo:hi] = True

    # 计算注意力（掩码外的位置设为 -inf）
    scores = Q @ K.T / math.sqrt(d)
    scores = scores.masked_fill(~mask, float('-inf'))
    attn = F.softmax(scores, dim=-1)
    return attn @ V, mask

def full_attention(Q, K, V):
    """标准全注意力，用于对比。"""
    d = Q.shape[-1]
    scores = Q @ K.T / math.sqrt(d)
    return F.softmax(scores, dim=-1) @ V

# --- 对比 ---
torch.manual_seed(42)
n, d = 16, 32
Q = torch.randn(n, d)
K = torch.randn(n, d)
V = torch.randn(n, d)

out_full = full_attention(Q, K, V)
out_sparse, mask = sliding_window_attention(Q, K, V, window_size=6)

sparsity = 1 - mask.float().mean().item()
print(f"序列长度: {n}, 窗口大小: 6")
print(f"全注意力计算量: {n*n} = {n}^2")
print(f"稀疏注意力计算量: {mask.sum().item()} (非零元素)")
print(f"稀疏率: {sparsity:.1%}")
print(f"近似误差 (L2): {(out_full - out_sparse).norm():.4f}")
```

### 示例 2：线性注意力 vs 标准注意力

```python
import torch
import torch.nn.functional as F
import math
import time

def standard_attention(Q, K, V):
    """标准 softmax 注意力: O(n^2 * d)"""
    n, d = Q.shape
    scores = Q @ K.T / math.sqrt(d)  # O(n^2 * d)
    attn = F.softmax(scores, dim=-1)  # n x n 矩阵
    return attn @ V  # O(n^2 * d)

def linear_attention(Q, K, V):
    """
    线性注意力（简化版）: O(n * d^2)

    核心技巧：用 ELU+1 替代 softmax 作为核函数，
    然后利用结合律改变计算顺序。

    标准: (Q K^T) V  →  先算 n*n 矩阵
    线性: Q (K^T V)  →  先算 d*d 矩阵
    """
    # 核函数（用 ELU+1 近似 softmax）
    Q_prime = F.elu(Q) + 1  # 保证非负
    K_prime = F.elu(K) + 1

    # 关键：先算 K^T @ V (d x d)，再乘 Q
    KV = K_prime.T @ V       # (d, d) — 与 n 无关！
    Z = K_prime.sum(dim=0)   # (d,) 归一化项

    # Q @ KV / Q @ Z
    numerator = Q_prime @ KV       # (n, d)
    denominator = Q_prime @ Z      # (n,) → 用于归一化
    return numerator / denominator.unsqueeze(-1)

# --- 正确性对比 ---
torch.manual_seed(42)
n, d = 64, 32
Q, K, V = torch.randn(n, d), torch.randn(n, d), torch.randn(n, d)

out_std = standard_attention(Q, K, V)
out_lin = linear_attention(Q, K, V)

# 线性注意力是近似，不会完全相等
cos_sim = F.cosine_similarity(out_std.flatten(), out_lin.flatten(), dim=0)
print(f"标准 vs 线性注意力:")
print(f"  余弦相似度: {cos_sim.item():.4f} (1.0=完全相同)")
print(f"  L2 距离: {(out_std - out_lin).norm().item():.4f}")

# --- 复杂度对比 ---
print(f"\n计算复杂度对比 (d={d}):")
for n in [64, 256, 1024, 4096, 16384]:
    flops_std = 2 * n * n * d  # QK^T + AV
    flops_lin = 2 * n * d * d  # K^TV + Q(K^TV)
    speedup = flops_std / flops_lin
    winner = "线性更快" if speedup > 1 else "标准更快"
    print(f"  n={n:>6d}: 标准={flops_std/1e6:>8.1f}M, "
          f"线性={flops_lin/1e6:>8.1f}M, "
          f"加速比={speedup:>5.1f}x ({winner})")

print(f"\n关键观察: 当 n > d 时线性注意力更快, 当 n < d 时标准更快")
print(f"实际场景: d≈128, n>128 时线性注意力有优势 → 长序列场景")
```

### 示例 3：Mamba 选择性 SSM 的核心机制

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelectiveSSM(nn.Module):
    """
    Mamba 的选择性 SSM 核心机制（简化教学版）。

    与传统 SSM 的关键区别：B, C, Delta 都是输入的函数，
    使得模型可以根据内容"选择性"地记忆或遗忘。
    """
    def __init__(self, dim, state_dim=16):
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim  # SSM 状态维度 N

        # 固定的 A 矩阵（HiPPO 初始化的简化版）
        # 实际 Mamba 使用更精心设计的初始化
        A = torch.arange(1, state_dim + 1).float()
        self.log_A = nn.Parameter(torch.log(A))  # 参数化为 log 保证正数

        # 输入相关的投影（这是"选择性"的来源）
        self.proj_B = nn.Linear(dim, state_dim, bias=False)  # B = f(x)
        self.proj_C = nn.Linear(dim, state_dim, bias=False)  # C = f(x)
        self.proj_delta = nn.Linear(dim, dim, bias=True)     # Delta = f(x)

        # 输出投影
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        """
        x: (batch, seq_len, dim)
        使用递归模式（推理高效，训练时实际用并行扫描）
        """
        B_size, L, D = x.shape
        N = self.state_dim

        # 计算输入相关的参数
        A = -torch.exp(self.log_A)         # (N,) 负数保证衰减
        B_seq = self.proj_B(x)             # (B, L, N) — 每步不同！
        C_seq = self.proj_C(x)             # (B, L, N) — 每步不同！
        delta = F.softplus(self.proj_delta(x))  # (B, L, D) — 步长

        # 离散化 A, B（使用零阶保持 ZOH）
        # deltaA = exp(delta * A)
        # deltaB = delta * B

        # 递归计算
        h = torch.zeros(B_size, D, N, device=x.device)  # 隐状态
        outputs = []

        for t in range(L):
            x_t = x[:, t, :]           # (B, D)
            delta_t = delta[:, t, :]   # (B, D)
            B_t = B_seq[:, t, :]       # (B, N)
            C_t = C_seq[:, t, :]       # (B, N)

            # 离散化
            dA = torch.exp(delta_t.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
            dB = delta_t.unsqueeze(-1) * B_t.unsqueeze(1)  # (B, D, N)

            # 状态更新: h = dA * h + dB * x
            h = dA * h + dB * x_t.unsqueeze(-1)  # (B, D, N)

            # 输出: y = C * h
            y_t = (h * C_t.unsqueeze(1)).sum(dim=-1)  # (B, D)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # (B, L, D)
        return self.out_proj(y)


# --- 演示 ---
torch.manual_seed(42)
dim, seq_len, state_dim = 32, 64, 16

model = SelectiveSSM(dim=dim, state_dim=state_dim)
x = torch.randn(2, seq_len, dim)
y = model(x)

print(f"输入: {list(x.shape)}")
print(f"输出: {list(y.shape)}")
print(f"SSM 状态维度: {state_dim}")

# 参数量对比
ssm_params = sum(p.numel() for p in model.parameters())
attn_params = 4 * dim * dim  # Q,K,V,O 投影
print(f"\n参数量:")
print(f"  Selective SSM: {ssm_params:,}")
print(f"  Self-Attention (QKV+O): {attn_params:,}")

# 计算复杂度对比
print(f"\n推理复杂度 (每个新 token):")
print(f"  SSM:       O(d * N) = O({dim} * {state_dim}) = {dim * state_dim}")
print(f"  Attention: O(n * d) = O(seq_len * {dim}), 随序列增长!")
print(f"  当 seq_len > N={state_dim} 时, SSM 更高效")

# 展示选择性机制
with torch.no_grad():
    test_input = torch.randn(1, 10, dim)
    deltas = F.softplus(model.proj_delta(test_input))
    print(f"\n选择性机制 — Delta 值 (控制 '记忆' vs '遗忘'):")
    delta_means = deltas[0, :, 0].tolist()
    for t, d in enumerate(delta_means):
        bar = "#" * int(d * 20)
        print(f"  step {t}: delta={d:.3f} {bar}")
    print(f"  delta 大 → 多关注当前输入; delta 小 → 多保留历史")
```

---

## 工程师视角

### 全面对比：标准 Transformer vs 各高效变体

| 维度 | 标准 Transformer | Longformer | Performer | Mamba | RWKV |
|------|-----------------|------------|-----------|-------|------|
| 注意力复杂度 | $O(n^2)$ | $O(n \cdot w)$ | $O(n \cdot d)$ | $O(n \cdot d)$ | $O(n \cdot d)$ |
| 是否精确注意力 | 精确 | 局部精确 | 近似 | 无注意力 | 无注意力 |
| 训练并行性 | 完全并行 | 完全并行 | 完全并行 | 并行扫描 | 并行扫描 |
| 推理效率 | KV Cache $O(n)$ | KV Cache $O(w)$ | 可递推 $O(1)$ | 递推 $O(1)$ | 递推 $O(1)$ |
| 长文本能力 | 受限于 $n^2$ | 窗口限制 | 理论无限 | 理论无限 | 理论无限 |
| 实际 LLM 表现 | 最成熟 | 适合分类 | 未被采用 | 接近但仍有差距 | 接近但仍有差距 |
| 代表性大模型 | GPT-4, LLaMA | - | - | Mamba-2 | RWKV-7 |

### Mamba vs Transformer：当前状态（2026 年）

截至 2026 年 4 月，Mamba 和 RWKV 等线性复杂度模型的现状：

**Mamba 优势**：推理速度快（无 KV Cache 膨胀）、显存占用低（状态固定）、长序列/DNA/音频等任务表现优秀。

**Mamba 劣势**：精确回忆能力弱（in-context learning 中的精确复制）、信息必须经有限隐状态传递（瓶颈）、生态和预训练资源远少于 Transformer。

**混合架构趋势（2024-2026）**：Jamba (AI21) = Mamba+Attention+MoE；Zamba (Zyphra) = Mamba+共享注意力；Samba (Microsoft) = Mamba+滑动窗口注意力。共同思路：大部分用 Mamba（高效），关键位置插入注意力层（精确检索）。

### 选型建议

```
你的场景适合哪种架构？

长序列 + 需要精确回忆（如 RAG、文档问答）？
  └─ 标准 Transformer + FlashAttention + RoPE 扩展
     （精确注意力不可替代）

超长序列 + 主要是分类/理解（如文档分类）？
  └─ Longformer / BigBird
     （稀疏注意力即可）

需要极低推理延迟 / 边缘部署？
  └─ Mamba 或 RWKV
     （固定状态，无 KV Cache 膨胀）

追求最佳综合性能？
  └─ 混合架构（Mamba + Attention）
     （2025-2026 的新趋势）
```

### 2025-2026 重要进展

| 进展 | 时间 | 要点 |
|------|------|------|
| Mamba-2 | 2024.05 | 揭示 SSM 与注意力的统一框架，速度提升 2-8x |
| Jamba | 2024.03 | 首个 Mamba+Attention+MoE 混合架构的大模型 |
| RWKV-6/7 | 2024-2025 | 持续改进，14B 规模接近同规模 Transformer |
| 混合架构成为共识 | 2025 | 多个团队验证 Mamba+Attention 混合优于纯 Mamba |
| State Space Duality | 2024 | Mamba-2 论文证明 SSM = 结构化注意力的理论等价 |
| FlashAttention-3 | 2024 | 进一步优化使得标准注意力仍有竞争力 |

---

## 本章小结

| 概念 | 一句话总结 |
|------|-----------|
| **稀疏注意力** | 只计算部分位置对的注意力，用滑动窗口 + 全局 token 组合 |
| **Longformer** | 局部窗口注意力 + 少数全局 token，$O(n \cdot w)$ |
| **BigBird** | Longformer + 随机连接，理论保证图连通性 |
| **线性注意力** | 用核函数近似 softmax，改变矩阵乘法顺序避免 $n^2$ |
| **Performer** | 用随机特征近似 softmax 核，理论优雅但实际效果有限 |
| **SSM / S4** | 线性递归状态方程 + HiPPO 初始化，训练时用卷积/FFT 并行 |
| **Mamba** | 选择性 SSM——让 B, C, Delta 依赖输入，实现动态记忆控制 |
| **Mamba-2** | 揭示 SSM 与注意力的理论统一，性能大幅提升 |
| **RWKV** | 线性递归 + 类注意力交互，"高效版 RNN" |
| **混合架构** | Mamba 层 + Attention 层，兼顾效率与精确检索 |

---

## 时效性说明

本文内容截至 **2026 年 4 月**。以下领域变化最快：

- **SSM vs Transformer 之争**：Mamba-2 缩小了差距，但 Transformer + FlashAttention 仍是主流。混合架构是当前共识方向
- **RWKV 生态**：社区驱动，版本迭代快（已到 RWKV-7），但工业界大规模采用案例仍少
- **稀疏注意力**：Longformer/BigBird 主要用于 BERT 时代的长文档分类，在 LLM 时代已被 FlashAttention + 全注意力取代
- **新架构**：可能出现全新的序列建模范式，目前无法预测

**建议**：关注 Tri Dao（FlashAttention/Mamba 作者）和 Albert Gu（S4/Mamba 作者）的论文更新。RWKV 社区的最新进展见其 GitHub 和 Discord。

**延伸阅读**：[论文与FAQ](论文与FAQ.md)——本模块涉及的关键论文列表和常见问题。
