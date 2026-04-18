# Self-Attention 与 Multi-Head Attention

> **前置知识**：已学完注意力机制基础，理解缩放点积注意力（Scaled Dot-Product Attention）的完整公式 $\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$，清楚 Query / Key / Value 三个矩阵各自的角色。数学符号不熟悉可查阅 [附录：数学基础速览](../附录_数学基础速览.md)。
>
> **本节目标**：理解 Self-Attention 与交叉注意力的区别，掌握 Multi-Head Attention 的完整拆分-并行-拼接流程，理解掩码注意力（Causal Attention）在解码器中的作用，并能从零实现它们。

---

## 1. 直觉与概述

### 1.1 从"查字典"到"读自己"

上一节我们把注意力比作"查字典"——用一个 Query 去和所有 Key 匹配，找到最相关的 Value。那时 Q 和 K/V 可以来自不同的地方（比如解码器去查编码器）。

**Self-Attention（自注意力）** 的特别之处在于：**Q、K、V 全部来自同一个序列**。

类比：你在读一个句子时，每个词都会"回头看"整个句子的所有词，来理解自己在句中的角色。

```
句子: "小猫 坐 在 垫子 上"

当处理 "坐" 这个词时:
  Q("坐") 与 K("小猫") → 高分（谁在坐？小猫）
  Q("坐") 与 K("垫子") → 中分（坐在哪？垫子上）
  Q("坐") 与 K("在")   → 低分（功能词）
  Q("坐") 与 K("坐")   → 中分（自己和自己也有一定相关性）
```

每个词通过自注意力"看到"了序列中与自己最相关的其他词，从而获得了**上下文感知**的表示。

### 1.2 为什么需要 Multi-Head？

一组 Q/K/V 只能学到一种"关注模式"。但语言中的依赖关系是多维度的：

```
句子: "The cat sat on the mat because it was tired"

  角度1 (语法): "it" 指代 → "cat"（指代消解）
  角度2 (语义): "tired" 关联 → "cat"（谁累了？）
  角度3 (位置): "sat" 关联 → "on the mat"（动作的地点）
```

**Multi-Head Attention** 让模型同时拥有多个"关注角度"——每个"头"（head）独立学习一种注意力模式，然后把所有头的结果拼接起来。类比：一个团队，每个成员从不同角度分析同一段文本，最后综合所有人的见解。

### 1.3 掩码注意力：不能偷看未来

在语言生成（如 GPT 逐词预测）中，生成第 $t$ 个词时只应看到前 $t-1$ 个词。**掩码注意力（Masked / Causal Attention）** 通过上三角掩码矩阵，把"未来位置"的注意力分数设为 $-\infty$（softmax 后变成 0）。

```
       位置1  位置2  位置3  位置4
位置1  [  ok    -∞    -∞    -∞  ]   ← 只能看自己
位置2  [  ok    ok    -∞    -∞  ]   ← 能看 1 和 2
位置3  [  ok    ok    ok    -∞  ]   ← 能看 1、2、3
位置4  [  ok    ok    ok    ok  ]   ← 能看所有
```

### 1.4 全景图

```
注意力机制基础（上一节）
  │
  ├── Self-Attention: Q/K/V 来自同一个输入
  │     ├── 编码器中: 无掩码，每个位置看到所有位置
  │     └── 解码器中: 加掩码，只能看到已生成的位置
  │
  ├── Multi-Head Attention: 多个头并行做注意力
  │     └── 拆分 d_model → h 个头 → 各自注意力 → 拼接 → 投影
  │
  └── Cross-Attention: Q 和 K/V 来自不同序列
        └── 典型场景：解码器查看编码器的输出
```

---

## 2. 严谨定义与原理

### 2.1 Self-Attention 的数学定义

给定输入序列 $X \in \mathbb{R}^{n \times d_{\text{model}}}$（$n$ 个 token，每个 $d_{\text{model}}$ 维）：

$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$
$$\text{SelfAttn}(X) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

其中 $W_Q, W_K \in \mathbb{R}^{d_{\text{model}} \times d_k}$，$W_V \in \mathbb{R}^{d_{\text{model}} \times d_v}$。关键：**Q、K、V 都从同一个 $X$ 投影而来**，这就是"Self"的含义。

| 符号 | 含义 | 维度 |
|------|------|------|
| $X$ | 输入序列 | $n \times d_{\text{model}}$ |
| $Q, K$ | Query / Key 矩阵 | $n \times d_k$ |
| $V$ | Value 矩阵 | $n \times d_v$ |
| $QK^T / \sqrt{d_k}$ | 注意力分数矩阵 | $n \times n$ |
| 输出 | 加权聚合后的表示 | $n \times d_v$ |

### 2.2 Self-Attention vs Cross-Attention

| 对比维度 | Self-Attention | Cross-Attention |
|---------|---------------|-----------------|
| Q 来源 | 输入序列 $X$ 本身 | 解码器的表示 $X_{\text{dec}}$ |
| K/V 来源 | 同一个 $X$ | 编码器的输出 $X_{\text{enc}}$ |
| 注意力矩阵 | $n \times n$（序列自身） | $n_{\text{dec}} \times n_{\text{enc}}$（跨序列） |
| 典型用途 | 编码器每层；解码器的掩码自注意力 | 解码器中"查看"编码器信息 |

### 2.3 Multi-Head Attention 的完整定义

**核心思想**：把 $d_{\text{model}}$ 拆成 $h$ 个头，每个头在低维空间独立做注意力，最后拼接。

假设 $d_{\text{model}} = 768$，$h = 12$，则每个头的维度 $d_k = d_v = 768 / 12 = 64$。

**步骤 1**：每个头有自己的投影矩阵

$$Q_i = X W_Q^{(i)}, \quad K_i = X W_K^{(i)}, \quad V_i = X W_V^{(i)} \quad \text{for } i = 1, \dots, h$$

**步骤 2**：每个头独立计算注意力

$$\text{head}_i = \text{softmax}\!\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right) V_i$$

**步骤 3**：拼接所有头，通过输出投影

$$\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) \, W^O$$

其中 $W^O \in \mathbb{R}^{(h \cdot d_v) \times d_{\text{model}}}$。输出维度仍为 $n \times d_{\text{model}}$——**输入和输出形状完全相同**，支持多层堆叠。

**参数量**：$W_Q + W_K + W_V + W^O = 4 \times d_{\text{model}}^2$。以 BERT-base 为例：$4 \times 768^2 \approx 2.36\text{M}$ /层。

> **实现细节**：实际代码中不会创建 $h$ 个独立小矩阵，而是用一个大的 $W_Q \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$ 投影后 reshape 拆头。数学等价，但对 GPU 更友好。

### 2.4 为什么 Multi-Head 比 Single-Head 更好？

**子空间多样性**：每个头在自己的子空间自由学习——局部语法、长距离依赖、位置模式、语义角色等。研究者可视化确认了这些现象（Clark et al. 2019）。

**稳定训练**：类似集成学习，即使某些头暂时学不好，其他头仍能提供有用信息。

**免费的午餐**：Multi-Head 的参数量和计算量与等效的 Single-Head（$d_k = d_{\text{model}}$）几乎完全相同。拆分不增加成本，却增加表达多样性。

> **参数量直觉**：拆头后参数量基本不变。$d_{\text{model}}$ 被拆成 $h$ 个 $d_k$（$d_k = d_{\text{model}} / h$），每个头的投影矩阵 $W^Q_i, W^K_i, W^V_i$ 变小但一共有 $h$ 组。总参数 = $h \times 3 \times d_k \times d_{\text{model}} = 3 \times d_{\text{model}}^2$，加上输出投影 $W^O$ 的 $d_{\text{model}}^2$，合计 $4 \times d_{\text{model}}^2$，与单头完全一致。拆头只改变了计算结构，没有改变参数规模。

### 2.5 掩码注意力（Masked / Causal Attention）

在 softmax 之前，将未来位置的分数设为 $-\infty$：

$$\text{MaskedAttn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}} + M\right) V$$

其中 $M_{ij} = 0$ 若 $j \leq i$，$M_{ij} = -\infty$ 若 $j > i$。

**在 Transformer 中的使用**：

| 组件 | 是否掩码 | 原因 |
|------|---------|------|
| 编码器 Self-Attention | 否 | 处理完整输入，每个位置看所有位置 |
| 解码器 Self-Attention | 是 | 防止偷看未来 token |
| 解码器 Cross-Attention | 否 | 可以查看编码器的所有位置 |
| GPT 等纯解码器模型 | 是 | 自回归生成 |

### 2.6 完整计算流程（图示）

```
输入 X: (n, d_model)
    │
    ├── X @ W_Q → Q ──→ reshape → (n, h, d_k) → transpose → (h, n, d_k)
    ├── X @ W_K → K ──→ reshape → (n, h, d_k) → transpose → (h, n, d_k)
    └── X @ W_V → V ──→ reshape → (n, h, d_v) → transpose → (h, n, d_v)
                                  │
                    ┌─── 对每个头 i (并行) ───┐
                    │  scores = Q_i @ K_i^T    │
                    │           / sqrt(d_k)     │
                    │  [可选] scores += mask    │
                    │  weights = softmax(scores)│
                    │  head_i = weights @ V_i   │
                    └──────────────────────────┘
                                  │
                    拼接: (h, n, d_v) → (n, h*d_v) = (n, d_model)
                                  │
                    输出投影: @ W_O → (n, d_model)  ← 和输入形状一样
```

---

## 3. Python 代码示例

### 3.1 用 numpy 从零实现 Multi-Head Attention

```python
import numpy as np

# ============================================================
# 用 numpy 从零实现 Multi-Head (Self-)Attention
# 目的：看清每个步骤的矩阵变换和维度变化
# ============================================================

def softmax(x, axis=-1):
    """数值稳定的 softmax"""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    """缩放点积注意力: (n, d_k) → (n, d_v)"""
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)          # (n, n)
    if mask is not None:
        scores = scores + mask                 # 被掩码位置变 -inf
    weights = softmax(scores, axis=-1)         # (n, n)
    return weights @ V, weights                # (n, d_v), (n, n)

class MultiHeadAttention:
    def __init__(self, d_model, n_heads, seed=42):
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        np.random.seed(seed)
        scale = np.sqrt(2.0 / d_model)
        self.W_Q = np.random.randn(d_model, d_model) * scale
        self.W_K = np.random.randn(d_model, d_model) * scale
        self.W_V = np.random.randn(d_model, d_model) * scale
        self.W_O = np.random.randn(d_model, d_model) * scale

    def forward(self, X, mask=None):
        n = X.shape[0]
        # 步骤 1: 线性投影
        Q = X @ self.W_Q          # (n, d_model)
        K = X @ self.W_K
        V = X @ self.W_V
        # 步骤 2: 拆分为多个头 — (n, d_model) → (h, n, d_k)
        Q = Q.reshape(n, self.n_heads, self.d_k).transpose(1, 0, 2)
        K = K.reshape(n, self.n_heads, self.d_k).transpose(1, 0, 2)
        V = V.reshape(n, self.n_heads, self.d_k).transpose(1, 0, 2)
        # 步骤 3: 每个头独立做注意力
        heads, all_weights = [], []
        for i in range(self.n_heads):
            out, w = scaled_dot_product_attention(Q[i], K[i], V[i], mask)
            heads.append(out)
            all_weights.append(w)
        # 步骤 4: 拼接 — (h, n, d_k) → (n, d_model)
        concat = np.stack(heads).transpose(1, 0, 2).reshape(n, self.d_model)
        # 步骤 5: 输出投影
        return concat @ self.W_O, np.stack(all_weights)

# --- 运行演示 ---
n, d_model, n_heads = 4, 8, 2
np.random.seed(0)
X = np.random.randn(n, d_model)

mha = MultiHeadAttention(d_model, n_heads)
output, weights = mha.forward(X)

print(f"输入 X 形状:  {X.shape}")         # (4, 8)
print(f"输出形状:      {output.shape}")     # (4, 8) — 和输入一样！
print(f"权重形状:      {weights.shape}")    # (2, 4, 4) — (h, n, n)
print(f"\n头 0 注意力权重:\n{weights[0].round(3)}")
print(f"\n头 1 注意力权重:\n{weights[1].round(3)}")
print("\n观察: 两个头的注意力分布不同——各自关注不同的模式。")
```

### 3.2 掩码注意力实现

```python
import numpy as np

# ============================================================
# 因果掩码（Causal Mask）的构造与对比
# ============================================================

def create_causal_mask(n):
    """下三角=0(允许), 上三角=-inf(禁止未来)"""
    mask = np.triu(np.ones((n, n)), k=1)
    mask[mask == 1] = -np.inf
    return mask

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

n, d_model = 5, 8
np.random.seed(42)
W_Q = np.random.randn(d_model, d_model) * 0.5
W_K = np.random.randn(d_model, d_model) * 0.5

np.random.seed(0)
X = np.random.randn(n, d_model)
Q, K = X @ W_Q, X @ W_K
scores = Q @ K.T / np.sqrt(d_model)

# 无掩码 vs 有掩码
w_no_mask = softmax(scores)
mask = create_causal_mask(n)
w_masked = softmax(scores + mask)

print("因果掩码矩阵 M:")
for i in range(n):
    row = ["  0  " if mask[i,j] == 0 else "-inf " for j in range(n)]
    print(f"  位置{i}: [{' '.join(row)}]")

print(f"\n无掩码 — 注意力权重 (位置 0):\n  {w_no_mask[0].round(3)}")
print(f"有掩码 — 注意力权重 (位置 0):\n  {w_masked[0].round(3)}")
print(f"有掩码 — 注意力权重 (位置 2):\n  {w_masked[2].round(3)}")

print("\n关键观察:")
print("  位置 0 有掩码后只能看自己，权重 = [1, 0, 0, 0, 0]")
print("  位置 2 只能看 0/1/2，上三角全为 0")
print("  每行仍求和为 1（softmax 重新归一化）")
```

### 3.3 PyTorch nn.MultiheadAttention 使用

```python
import torch
import torch.nn as nn

# ============================================================
# PyTorch nn.MultiheadAttention 完整使用指南
# ============================================================

d_model, n_heads, seq_len, batch = 64, 4, 6, 2

mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads,
                             dropout=0.0, batch_first=True)
torch.manual_seed(42)
x = torch.randn(batch, seq_len, d_model)

# --- Self-Attention ---
output, attn_weights = mha(query=x, key=x, value=x,
                           need_weights=True, average_attn_weights=True)
print(f"输入:   {x.shape}")              # (2, 6, 64)
print(f"输出:   {output.shape}")          # (2, 6, 64) — 形状不变
print(f"权重 (头平均): {attn_weights.shape}")  # (2, 6, 6)

# 每个头的权重
_, attn_per_head = mha(query=x, key=x, value=x,
                       need_weights=True, average_attn_weights=False)
print(f"每头权重: {attn_per_head.shape}")  # (2, 4, 6, 6)

# --- 因果掩码 ---
# 注意: PyTorch 中 True = 禁止关注（和手写 -inf 约定相反！）
causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool),
                         diagonal=1)
out_masked, attn_masked = mha(query=x, key=x, value=x,
                               attn_mask=causal_mask)
print(f"\n因果掩码后注意力权重 (batch 0, 取前 3 行):")
print(attn_masked[0, :3].detach().round(decimals=3))
print("上三角全为 0 ✓")

# --- Cross-Attention ---
enc_out = torch.randn(batch, 10, d_model)
dec_state = torch.randn(batch, 3, d_model)
cross_out, cross_attn = mha(query=dec_state, key=enc_out, value=enc_out)
print(f"\nCross-Attention: Q={dec_state.shape}, K/V={enc_out.shape}")
print(f"  输出: {cross_out.shape}, 权重: {cross_attn.shape}")  # (2,3,64), (2,3,10)

# --- 参数量 ---
total = sum(p.numel() for p in mha.parameters())
print(f"\n参数量: {total:,} (理论值 4*{d_model}^2+4*{d_model} = "
      f"{4*d_model**2+4*d_model:,})")
```

### 3.4 可视化多头的不同注意力模式

```python
import numpy as np

# ============================================================
# 用 ASCII 热力图展示不同头学到的不同关注模式
# ============================================================

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

def ascii_heatmap(weights, tokens, head_id):
    chars = " ░▒▓█"
    n = len(tokens)
    pad = max(len(t) for t in tokens)
    header = " " * (pad + 4) + "".join(f"{t[:3]:>4s}" for t in tokens)
    print(f"  {header}")
    for i in range(n):
        row = "".join(f" {chars[min(int(weights[i,j]*5),4)]}  "
                      for j in range(n))
        print(f"  {tokens[i]:>{pad}s}:  {row}")

tokens = ["The", "cat", "sat", "on", "the", "mat"]
n = len(tokens)
np.random.seed(42)

# 头 0: 局部注意力（关注相邻词）
local = np.array([[-abs(i-j)*1.5 + np.random.randn()*0.3
                   for j in range(n)] for i in range(n)])
w0 = softmax(local)

# 头 1: 语法依赖（主谓宾关联）
syn = np.random.randn(n, n) * 0.3
syn[1,2] = syn[2,1] = 3.0  # cat ↔ sat
syn[2,5] = syn[5,2] = 2.5  # sat ↔ mat
w1 = softmax(syn)

# 头 2: 冠词-名词关系
ref = np.random.randn(n, n) * 0.3
ref[0,1] = 3.0  # The → cat
ref[4,5] = 3.0  # the → mat
ref[1,0] = ref[5,4] = 2.5
w2 = softmax(ref)

print(f"句子: {' '.join(tokens)}")
print(f"图例: ' '=低  '░'=较低  '▒'=中  '▓'=较高  '█'=高\n")

ascii_heatmap(w0, tokens, 0)
print("  → 头 0: 局部上下文（关注相邻位置）\n")
ascii_heatmap(w1, tokens, 1)
print("  → 头 1: 语法关系 cat↔sat(主谓), sat↔mat(谓宾)\n")
ascii_heatmap(w2, tokens, 2)
print("  → 头 2: 限定词-名词 The→cat, the→mat\n")

print("结论: 每个头学到不同模式。拼接后模型同时拥有多维度信息。")
```

### 3.5 可训练的 PyTorch Multi-Head Attention 手写实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ============================================================
# 可训练的 Multi-Head Attention — 与 nn.MultiheadAttention 等价
# ============================================================

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model, self.n_heads = d_model, n_heads
        self.d_k = d_model // n_heads
        self.scale = math.sqrt(self.d_k)

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        query: (batch, seq_q, d_model)
        key/value: (batch, seq_k, d_model)
        mask: (seq_q, seq_k), True=禁止关注
        返回: output (batch, seq_q, d_model), weights (batch, h, seq_q, seq_k)
        """
        B, seq_q = query.size(0), query.size(1)
        seq_k = key.size(1)

        # 投影 + 拆头: (B, seq, d_model) → (B, h, seq, d_k)
        Q = self.W_Q(query).view(B, seq_q, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(key).view(B, seq_k, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(value).view(B, seq_k, self.n_heads, self.d_k).transpose(1, 2)

        # 注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, h, sq, sk)
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(mask, float('-inf'))

        weights = self.dropout(F.softmax(scores, dim=-1))
        context = torch.matmul(weights, V)  # (B, h, sq, d_k)

        # 拼接 + 输出投影
        context = context.transpose(1, 2).contiguous().view(B, seq_q, self.d_model)
        return self.W_O(context), weights

# --- 验证 ---
torch.manual_seed(42)
d_model, n_heads, B, seq = 64, 4, 2, 8
my_mha = MultiHeadAttention(d_model, n_heads)
x = torch.randn(B, seq, d_model)

out, w = my_mha(x, x, x)
print(f"Self-Attn — 输入: {x.shape}, 输出: {out.shape}, 权重: {w.shape}")
print(f"权重行和为 1: {w.sum(-1).allclose(torch.ones_like(w.sum(-1)))}")

# 因果掩码
mask = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)
out_m, w_m = my_mha(x, x, x, mask=mask)
print(f"Masked — 上三角全为 0: "
      f"{w_m[0,0].triu(diagonal=1).abs().max().item() < 1e-6}")

# Cross-Attention
enc = torch.randn(B, 12, d_model)
dec = torch.randn(B, 5, d_model)
out_c, w_c = my_mha(dec, enc, enc)
print(f"Cross — Q: {dec.shape}, K/V: {enc.shape}, Out: {out_c.shape}, W: {w_c.shape}")

print(f"\n参数量: {sum(p.numel() for p in my_mha.parameters()):,}")
```

---

## 4. 工程师视角

### 4.1 典型模型配置

| 模型 | $d_{\text{model}}$ | $h$ | $d_k$ | 层数 | MHA 参数/层 |
|------|----------|------|-------|------|------------|
| BERT-base | 768 | 12 | 64 | 12 | ~2.4M |
| BERT-large | 1024 | 16 | 64 | 24 | ~4.2M |
| GPT-3 175B | 12288 | 96 | 128 | 96 | ~604M |
| LLaMA 2 7B | 4096 | 32 | 128 | 32 | ~67M |

**规律**：$d_k$ 通常固定为 64 或 128，通过增减头数和层数来调整模型规模。

### 4.2 注意力的计算复杂度：O(n^2) 问题

$QK^T$ 产生 $n \times n$ 的注意力矩阵，时间和空间复杂度均为 $O(n^2)$：

```
序列长度   注意力矩阵大小    显存 (fp32)
    512         262K            1 MB
  2,048       4.19M           16 MB
  8,192      67.1M           256 MB
131,072      17.2B            64 GB  ← 单层就需要 64GB！
```

这催生了各种高效注意力变体：
- **Flash Attention**（Dao 2022）：不改数学，优化 GPU 内存访问，大幅减少显存
- **稀疏注意力**（Longformer, BigBird）：只计算部分位置对，$O(n)$
- **线性注意力**（Performer）：用核方法近似 softmax，$O(n)$

### 4.3 KV Cache：推理阶段的关键优化

自回归生成中，已生成 token 的 K/V 不会变。**KV Cache** 把它们缓存起来避免重复计算：

```
朴素方法:
  生成第 t 个 token → 重新计算前 t 个 token 的 K/V → 总计算 O(N^2)

KV Cache:
  生成第 t 个 token → 只算第 t 个的 K/V，拼接缓存 → 总计算 O(N)
```

所有 LLM 推理框架（vLLM、TensorRT-LLM、llama.cpp）都使用 KV Cache。代价是额外显存，这也是为什么生成长度影响显存占用。

> GQA、MQA 等 KV Cache 优化变体将在后续模块详讲。核心思想：**已算过的 K/V 缓存起来不重复算**。

### 4.4 常见陷阱

**1. 掩码方向搞反**（最常见 bug）

```python
# PyTorch nn.MultiheadAttention:  True  = 禁止关注
# 手写实现:                        -inf  = 禁止关注
# HuggingFace Transformers:        0     = 禁止关注, 1 = 允许
# 切换框架时务必检查掩码约定！
```

**2. batch_first 默认是 False**

```python
# 默认输入形状: (seq_len, batch, d_model) ← 不直觉！
# 强烈建议: 总是设置 batch_first=True
```

**3. 头数必须整除 d_model**

$d_{\text{model}} = 768$ 时常用 $h = 12$（BERT-base）或 $h = 16$。

**4. 不要忘记 $\sqrt{d_k}$ 缩放**

$d_k$ 较大时 $QK^T$ 的值会很大，导致 softmax 饱和。缩放因子是必须的。

### 4.5 Transformer 层的完整结构（预告）

Multi-Head Attention 在 Transformer 中不是独立存在的：

```
输入 X
  ├──→ Multi-Head Attention → Dropout → (+) → LayerNorm →
  │                                      ↑                 │
  └──────── 残差连接 ────────────────────┘                 │
                                                           │
  ├──→ FFN (两层全连接+激活) → Dropout → (+) → LayerNorm → 输出
  │                                       ↑
  └──────── 残差连接 ─────────────────────┘
```

残差连接、LayerNorm、FFN 的详细实现将在下一节展开。

---

## 5. 本节小结

| 概念 | 一句话总结 |
|------|-----------|
| Self-Attention | Q/K/V 都来自同一个 $X$，让每个位置"看到"所有位置 |
| Cross-Attention | Q 来自一个序列，K/V 来自另一个序列 |
| Multi-Head | 拆成 $h$ 个头并行做注意力后拼接，不增加计算量但增加表达多样性 |
| 因果掩码 | 上三角设 $-\infty$，防止偷看未来，自回归生成的必需品 |
| $d_k$ | 典型值 64 或 128，通过增减头数调整模型大小 |
| $O(n^2)$ | 注意力矩阵随序列长度二次增长，长序列的核心瓶颈 |
| KV Cache | 缓存已生成 token 的 K/V，推理加速从 $O(N^2)$ 到 $O(N)$ |

**从本节到 Transformer 的路线**：

```
缩放点积注意力（上一节）
    ▼
Self-Attention + Multi-Head + 掩码 ← 本节
    │
    ├── + 残差连接 + LayerNorm
    ├── + 前馈网络 (FFN)
    ├── + 位置编码
    ▼
完整的 Transformer 架构 → 下一节
```

**下一节**：Transformer 完整架构 —— 编码器和解码器如何组合？位置编码是怎么回事？为什么 "Attention Is All You Need"？
