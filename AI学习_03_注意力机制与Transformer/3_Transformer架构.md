# Transformer 完整架构

> **前置知识**：[注意力机制](1_注意力机制.md)、[Self-Attention 与 Multi-Head](2_SelfAttention与MultiHead.md)、[编码器-解码器框架](../AI学习_02_经典网络架构/4_编码器解码器框架.md)

---

## 直觉与概述

### 核心问题

你已经学完了 Self-Attention 和 Multi-Head Attention。它们很强大，但一个注意力层还不是一个完整的模型。要真正处理序列到序列的任务（翻译、摘要、对话），你需要回答以下问题：

1. 注意力层的输出怎么传给下一层？直接堆叠会不会导致训练不稳定？
2. 只有注意力够吗？还需要什么其他组件？
3. 编码器看完整个输入后，解码器如何利用编码器的信息来生成输出？
4. 解码器生成第 $t$ 个词时，怎么保证它看不到第 $t+1$ 个词？

Transformer 的回答是：**把 Multi-Head Attention 与残差连接、Layer Norm、前馈网络组装成标准化的"层"，然后把这些层堆叠成编码器和解码器**。整个架构完全不依赖 RNN，训练时所有位置可以并行计算。

### "Attention Is All You Need" 的含义

2017 年 Vaswani 等人的论文标题并不是说只需要注意力就能解决一切，而是说：

- **不需要 RNN**：在此之前，序列建模几乎都依赖 RNN/LSTM 的循环结构
- **不需要 CNN**：之前也有人尝试用 CNN 处理序列（ConvS2S）
- **注意力足以建模序列中任意两个位置之间的依赖关系**，不再需要让信息沿着时间步逐个传递

### 整体架构鸟瞰

```
                        输出概率分布
                            ↑
                     ┌──────────────┐
                     │ Linear+Softmax │
                     └──────┬───────┘
                     ┌──────────────┐
                     │   解码器      │ × N 层
                     │  (Decoder)   │
                     └──────┬───────┘
          编码器输出 ────────┤
                     ┌──────────────┐        ┌──────────────┐
                     │   编码器      │ × N 层 │  输出 Embed   │
                     │  (Encoder)   │        │  + 位置编码   │
                     └──────┬───────┘        └──────┬───────┘
                     ┌──────────────┐        目标序列（右移一位）
                     │  输入 Embed   │
                     │  + 位置编码   │
                     └──────┬───────┘
                        输入序列
```

**数据流总结**（以翻译任务为例）：

1. 输入序列经过 Embedding + 位置编码，进入编码器
2. 编码器输出一个序列表示（每个位置都有一个向量）
3. 目标序列（右移一位）经过 Embedding + 位置编码，进入解码器
4. 解码器同时接收自身的输入和编码器的输出
5. 解码器输出经过线性层 + Softmax 得到每个位置的词概率分布

---

## 严谨定义与原理

### 编码器层（Encoder Layer）

原始 Transformer 使用 N=6 个相同结构的编码器层堆叠。每一层包含两个子层：

```
输入 x
  │
  ├──→ Multi-Head Self-Attention ──→ (+) ──→ Layer Norm ──→ 输出₁
  │                                   ↑
  └───────────────────────────────────┘  (残差连接)
输出₁
  │
  ├──→ Feed-Forward Network ──→ (+) ──→ Layer Norm ──→ 输出₂
  │                              ↑
  └─────────────────────────────┘  (残差连接)
```

用数学表示（Post-LN 形式，即原始论文的写法）：

$$\text{中间} = \text{LayerNorm}(x + \text{MultiHeadAttention}(x, x, x))$$

$$\text{输出} = \text{LayerNorm}(\text{中间} + \text{FFN}(\text{中间}))$$

其中 $\text{MultiHeadAttention}(Q, K, V)$ 的三个参数都是 $x$ 自身，因此是**自注意力**。

#### 残差连接（Residual Connection）

残差连接的公式：$\text{output} = x + F(x)$，即"跳过"子层，直接把输入加到子层的输出上。

**为什么需要残差连接？** 没有残差连接时，梯度从第 $L$ 层传到第 $1$ 层需要经过所有中间层的连乘。如果某些层的梯度小于 1，连乘后梯度会指数级衰减。有了残差连接后：

$$x_{l+1} = x_l + F_l(x_l) \quad \Rightarrow \quad \frac{\partial x_{l+1}}{\partial x_l} = I + \frac{\partial F_l(x_l)}{\partial x_l}$$

即使 $\frac{\partial F_l}{\partial x_l}$ 很小，梯度中仍包含 identity 项 $I$，为梯度提供了直接传播路径，通常显著缓解深层网络的梯度衰减问题（注意这不是绝对的数学保证，而是实践中非常有效的机制）。6 层甚至几十层的 Transformer 能稳定训练，残差连接功不可没。

#### Layer Normalization

Layer Norm 对每个样本在特征维度上做归一化：

$$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

其中 $\mu, \sigma^2$ 在特征维度上计算，$\gamma, \beta$ 是可学习参数。

| 特性 | Batch Norm | Layer Norm |
|------|-----------|-----------|
| 归一化维度 | batch 维度（跨样本） | 特征维度（每个样本独立） |
| 依赖 batch 大小 | 是（小 batch 不稳定） | 否 |
| 序列长度可变 | 不方便 | 天然支持 |

Layer Norm 在 Transformer 中的位置有两种流派：

- **Post-LN**（原始论文）：$\text{LayerNorm}(x + \text{Sublayer}(x))$
- **Pre-LN**（现代模型）：$x + \text{Sublayer}(\text{LayerNorm}(x))$

#### 前馈网络（FFN）

每个编码器层中的 FFN 是一个**位置独立**的两层全连接网络：

$$\text{FFN}(x) = W_2 \cdot \text{Activation}(W_1 x + b_1) + b_2$$

- $W_1 \in \mathbb{R}^{d_{ff} \times d_{model}}$，$W_2 \in \mathbb{R}^{d_{model} \times d_{ff}}$
- $d_{ff}$ 通常是 $d_{model}$ 的 4 倍（原始论文：$d_{model}=512, d_{ff}=2048$）
- 激活函数的演化：ReLU（原始）→ GELU（GPT-2/BERT）→ SwiGLU（LLaMA/PaLM）

**FFN 的作用**：注意力层负责"位置之间的信息交换"，FFN 负责"每个位置内部的非线性变换"。有研究表明，FFN 的参数中存储了大量的事实知识（类似于键值存储）。

### 解码器层（Decoder Layer）

解码器层比编码器层多一个子层，共三个：

```
输入 x（解码器的）
  │
  ├──→ Masked Multi-Head Self-Attention ──→ (+) ──→ LN ──→ 输出₁
  │                                          ↑
  └──────────────────────────────────────────┘
输出₁
  │
  ├──→ Multi-Head Cross-Attention ──→ (+) ──→ LN ──→ 输出₂
  │    (Q=输出₁, K=V=编码器输出)       ↑
  └────────────────────────────────────┘
输出₂
  │
  ├──→ Feed-Forward Network ──→ (+) ──→ LN ──→ 输出₃
  │                              ↑
  └─────────────────────────────┘
```

#### 掩码自注意力（Masked Self-Attention）

生成第 $t$ 个 token 时，**不能看到第 $t+1, t+2, \ldots$ 位置的信息**。实现方式：用上三角掩码把"未来"位置的分数设为 $-\infty$：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + \text{Mask}\right) V$$

```
位置     1     2     3     4
  1   [ 可以  -∞    -∞    -∞  ]   → 位置1只能看自己
  2   [ 可以  可以  -∞    -∞  ]   → 位置2能看1和2
  3   [ 可以  可以  可以  -∞  ]   → 位置3能看1、2、3
  4   [ 可以  可以  可以  可以 ]   → 位置4能看所有
```

训练时目标序列的所有位置同时计算（Teacher Forcing），掩码保证每个位置"假装"看不到后面的位置，效果等价于逐步生成但计算效率高得多。

#### 交叉注意力（Cross-Attention）

交叉注意力是解码器"读取"编码器信息的通道。关键区别在于 Q、K、V 的来源：

| 参数 | 来源 | 含义 |
|------|------|------|
| Q | 解码器当前层的表示 | "我想查询什么" |
| K | 编码器的最终输出 | "输入序列有什么可查的" |
| V | 编码器的最终输出 | "查到后取什么值" |

### 完整数据流

**编码器侧**：

```
输入 token IDs → Embedding (m, 512) → + 位置编码
  → Encoder Layer 1: Self-Attention → Add&Norm → FFN → Add&Norm
  → Encoder Layer 2~6: (相同结构)
  → 编码器输出: (m, 512)
```

**解码器侧**：

```
目标 token IDs（右移一位）→ Embedding (n, 512) → + 位置编码
  → Decoder Layer 1:
      Masked Self-Attention → Add&Norm
      Cross-Attention(Q=自身, K=V=编码器输出) → Add&Norm
      FFN → Add&Norm
  → Decoder Layer 2~6: (相同结构)
  → Linear (n, vocab_size) → Softmax → 概率分布
```

### 三种注意力的对比

| 类型 | Q 来自 | K, V 来自 | 掩码 | 作用 |
|------|--------|-----------|------|------|
| 编码器 Self-Attention | 输入序列 | 输入序列 | 无 | 输入内部建模依赖 |
| 解码器 Masked Self-Attention | 目标序列 | 目标序列 | 因果掩码 | 已生成序列内部建模依赖 |
| Cross-Attention | 解码器表示 | 编码器输出 | 无 | 解码器从编码器获取信息 |

---

## Python 代码示例

### 示例 1：PyTorch 从零实现完整 Transformer

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ============================================================
# 组件 1：Multi-Head Attention
# ============================================================

class MultiHeadAttention(nn.Module):
    """支持 Self-Attention、Masked Self-Attention、Cross-Attention 三种模式。"""

    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 线性变换 + 分头: (batch, seq, d_model) → (batch, n_heads, seq, d_k)
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 缩放点积注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            # 注意：这里 mask 约定为 1=允许/0=屏蔽（与原始论文一致）
            # 文件 2 中的实现使用 True=屏蔽，两种约定均常见，注意区分
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        # 注：为简化教学省略了注意力 Dropout，实际实现应在此处加 Dropout

        # 加权求和 + 合并多头
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(context), attn_weights


# ============================================================
# 组件 2：前馈网络
# ============================================================

class FeedForward(nn.Module):
    """FFN(x) = W2 * GELU(W1*x + b1) + b2"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


# ============================================================
# 组件 3：位置编码（正弦余弦，详见 4_位置编码.md）
# ============================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])


# ============================================================
# 组件 4：编码器层
# ============================================================

class EncoderLayer(nn.Module):
    """Self-Attention + FFN，各带残差连接和 Layer Norm（Post-LN）。"""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        attn_out, attn_w = self.self_attn(x, x, x, mask=src_mask)
        x = self.norm1(x + self.dropout1(attn_out))
        x = self.norm2(x + self.dropout2(self.ffn(x)))
        return x, attn_w


# ============================================================
# 组件 5：解码器层
# ============================================================

class DecoderLayer(nn.Module):
    """Masked Self-Attention + Cross-Attention + FFN。"""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.masked_self_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        attn_out, self_attn_w = self.masked_self_attn(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout1(attn_out))
        attn_out, cross_attn_w = self.cross_attn(x, memory, memory, mask=memory_mask)
        x = self.norm2(x + self.dropout2(attn_out))
        x = self.norm3(x + self.dropout3(self.ffn(x)))
        return x, self_attn_w, cross_attn_w


# ============================================================
# 组件 6：完整 Transformer
# ============================================================

class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=128, n_heads=4,
                 d_ff=512, n_layers=3, dropout=0.1, max_len=200):
        super().__init__()
        self.d_model = d_model
        self.src_embed = nn.Embedding(src_vocab, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.fc_out = nn.Linear(d_model, tgt_vocab)
        self._init_params()

    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src, pad_idx=0):
        return (src != pad_idx).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt, pad_idx=0):
        pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
        tgt_len = tgt.size(1)
        causal = torch.tril(torch.ones(tgt_len, tgt_len, device=tgt.device)).bool()
        return pad_mask & causal.unsqueeze(0).unsqueeze(0)

    def forward(self, src, tgt, pad_idx=0):
        src_mask = self.make_src_mask(src, pad_idx)
        tgt_mask = self.make_tgt_mask(tgt, pad_idx)
        mem_mask = self.make_src_mask(src, pad_idx)
        scale = math.sqrt(self.d_model)

        # 编码
        x = self.pos_enc(self.src_embed(src) * scale)
        enc_attns = []
        for layer in self.encoder_layers:
            x, attn = layer(x, src_mask)
            enc_attns.append(attn)
        memory = x

        # 解码
        x = self.pos_enc(self.tgt_embed(tgt) * scale)
        self_attns, cross_attns = [], []
        for layer in self.decoder_layers:
            x, sa, ca = layer(x, memory, tgt_mask, mem_mask)
            self_attns.append(sa)
            cross_attns.append(ca)

        logits = self.fc_out(x)
        attn_info = {
            'enc_self': enc_attns,
            'dec_self': self_attns,
            'dec_cross': cross_attns,
        }
        return logits, attn_info


# ============================================================
# 验证：序列复制任务 + 打印注意力权重
# ============================================================

VOCAB_SIZE = 12   # 0=PAD, 1=BOS, 2=EOS, 3~11=数字
PAD, BOS, EOS = 0, 1, 2

model = Transformer(VOCAB_SIZE, VOCAB_SIZE, d_model=64, n_heads=4,
                    d_ff=256, n_layers=2, dropout=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98))
criterion = nn.CrossEntropyLoss(ignore_index=PAD)

print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

def make_batch(batch_size=32, seq_len=6):
    seq = torch.randint(3, VOCAB_SIZE, (batch_size, seq_len))
    src = torch.cat([seq, torch.full((batch_size, 1), EOS)], dim=1)
    tgt_in = torch.cat([torch.full((batch_size, 1), BOS), seq,
                         torch.full((batch_size, 1), EOS)], dim=1)
    tgt_label = torch.cat([seq, torch.full((batch_size, 1), EOS),
                            torch.full((batch_size, 1), PAD)], dim=1)
    return src, tgt_in, tgt_label

# 训练
model.train()
for epoch in range(100):
    src, tgt_in, tgt_label = make_batch(64)
    logits, _ = model(src, tgt_in, pad_idx=PAD)
    loss = criterion(logits.reshape(-1, VOCAB_SIZE), tgt_label.reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    if (epoch + 1) % 25 == 0:
        pred = logits.argmax(dim=-1)
        mask = tgt_label != PAD
        acc = (pred[mask] == tgt_label[mask]).float().mean()
        print(f"Epoch {epoch+1:>3d}, Loss: {loss:.4f}, Acc: {acc:.4f}")

# 推理 + 打印注意力权重
model.eval()
with torch.no_grad():
    test_src = torch.tensor([[3, 7, 5, 9, 4, 6, EOS]])
    test_tgt = torch.tensor([[BOS, 3, 7, 5, 9, 4, 6, EOS]])
    logits, attn = model(test_src, test_tgt, pad_idx=PAD)
    print(f"\n输入: {test_src[0].tolist()}")
    print(f"预测: {logits.argmax(dim=-1)[0].tolist()}")

    # 打印第 1 层交叉注意力（解码器看编码器的模式）
    cross_w = attn['dec_cross'][0][0, 0].numpy()  # 第1层, 第1个样本, head 0
    print(f"\n第1层 Cross-Attention (head 0), tgt→src:")
    for i, row in enumerate(cross_w):
        row_str = " ".join(f"{v:.2f}" for v in row)
        print(f"  tgt[{i}]: [{row_str}]")

    # 打印第 1 层解码器掩码自注意力（验证因果掩码）
    self_w = attn['dec_self'][0][0, 0].numpy()
    print(f"\n第1层 Masked Self-Attention (head 0), 上三角应为 0:")
    for i, row in enumerate(self_w):
        row_str = " ".join(f"{v:.2f}" for v in row)
        print(f"  pos[{i}]: [{row_str}]")
```

### 示例 2：对比 PyTorch 内置 nn.Transformer

> **注意**：本示例依赖示例 1 中定义的 `PositionalEncoding` 类，请确保先运行示例 1 的代码。

```python
import torch
import torch.nn as nn
import math

class BuiltinTransformerModel(nn.Module):
    """使用 PyTorch 内置 nn.Transformer 的封装，功能与手写版等价。"""

    def __init__(self, src_vocab, tgt_vocab, d_model=128, nhead=4,
                 num_layers=3, d_ff=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.src_embed = nn.Embedding(src_vocab, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        # 一行搞定整个 Transformer
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_layers, num_decoder_layers=num_layers,
            dim_feedforward=d_ff, dropout=dropout, batch_first=True,
        )
        self.fc_out = nn.Linear(d_model, tgt_vocab)

    def forward(self, src, tgt):
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1))
        tgt_mask = tgt_mask.to(tgt.device)
        scale = math.sqrt(self.d_model)
        src_emb = self.pos_enc(self.src_embed(src) * scale)
        tgt_emb = self.pos_enc(self.tgt_embed(tgt) * scale)
        output = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)
        return self.fc_out(output)

# --- 对比 ---
VOCAB = 12
custom = Transformer(VOCAB, VOCAB, d_model=64, n_heads=4, d_ff=256, n_layers=2)
builtin = BuiltinTransformerModel(VOCAB, VOCAB, d_model=64, nhead=4, num_layers=2, d_ff=256)

print(f"手写版参数量: {sum(p.numel() for p in custom.parameters()):,}")
print(f"内置版参数量: {sum(p.numel() for p in builtin.parameters()):,}")

src = torch.randint(1, VOCAB, (2, 5))
tgt = torch.randint(1, VOCAB, (2, 7))
out1, _ = custom(src, tgt)
out2 = builtin(src, tgt)
print(f"手写版输出: {list(out1.shape)}, 内置版输出: {list(out2.shape)}")
print("生产环境推荐使用 nn.Transformer 或 Hugging Face transformers 库")
```

---

## 工程师视角

### Pre-LN vs Post-LN

原始 Transformer 使用 Post-LN，但现代模型几乎都切换到了 Pre-LN：

```
Post-LN: LN(x + Sublayer(x))       Pre-LN: x + Sublayer(LN(x))
  x                                   x
  │                                   │
  ├→ Sublayer(x) → (+) → LN          ├→ LN(x) → Sublayer → (+)
  │                 ↑                 │                      ↑
  └─────────────────┘                 └──────────────────────┘
```

| 方面 | Post-LN | Pre-LN |
|------|---------|--------|
| 梯度流动 | 残差路径上的梯度经过 LN，可能被缩放 | 残差路径完全"干净"，梯度无阻碍直通 |
| 训练稳定性 | 需要 warmup | 无需 warmup 也能稳定训练 |
| 层数扩展 | 超过 12 层容易不稳定 | 100+ 层仍然稳定 |

实际选择：**用 Pre-LN**。GPT-3、LLaMA、Mistral 等几乎所有 LLM 都使用 Pre-LN。

### Encoder-only vs Decoder-only vs Encoder-Decoder

```
Encoder-only (BERT)        Decoder-only (GPT)       Encoder-Decoder (T5)
┌─────────────┐            ┌─────────────┐          ┌──────┐  ┌──────┐
│ 双向注意力   │            │ 因果注意力   │          │双向  │→│因果+交叉│
│ (全连接)     │            │ (下三角)     │          │      │  │       │
└─────────────┘            └─────────────┘          └──────┘  └──────┘
  理解类任务                 生成类任务                通用 Seq2Seq
  分类、NER                  文本生成、对话             翻译、摘要
```

| 变体 | 代表模型 | 典型任务 |
|------|---------|---------|
| Encoder-only | BERT, RoBERTa | 分类、抽取、NER |
| Decoder-only | GPT, LLaMA, Mistral | 文本生成、对话、推理 |
| Encoder-Decoder | T5, BART | 翻译、摘要 |

**当前趋势**：Decoder-only 成为绝对主流。原因：预训练目标统一（下一个 token 预测）、通过 in-context learning 可处理几乎所有任务、规模化效率高。

### 参数量估算

单个 Transformer 层的参数（$d_{ff} = 4 d_{model}$）：

| 组件 | 参数量 |
|------|--------|
| Self-Attention（Q, K, V, O） | $4 d_{model}^2$ |
| FFN（两层） | $8 d_{model}^2$ |
| **单层总计** | $\approx 12 d_{model}^2$ |

$N$ 层 Decoder-only 模型总参数量：

$$\text{总参数量} \approx 12 N d_{model}^2 + V \cdot d_{model}$$

| 模型 | $d_{model}$ | $N$ 层 | $n_{heads}$ | 参数量 |
|------|------------|--------|-------------|--------|
| Transformer Base | 512 | 6+6 | 8 | 65M |
| BERT-Base | 768 | 12 | 12 | 110M |
| GPT-3 | 12288 | 96 | 96 | 175B |
| LLaMA-2-7B | 4096 | 32 | 32 | 7B |
| LLaMA-2-70B | 8192 | 80 | 64 | 70B |

**$n_{heads}$ 与 $d_{model}$ 的关系**：通常 $d_k = d_{model} / n_{heads} = 64$ 或 $128$。增大模型时同时增加头数和总维度。

### 为什么 Transformer 能并行

**RNN**：每个时间步依赖前一步的隐藏状态，是严格串行。长度 $n$ 的序列需要 $O(n)$ 步串行计算。

**Transformer**：Self-Attention 的 $QK^T$ 是矩阵乘法，所有位置同时计算；FFN 对每个位置独立计算。训练时整个序列完全并行。

**代价**：Self-Attention 计算量 $O(n^2 \cdot d)$，序列很长时 $n^2$ 成瓶颈。这催生了 Flash Attention、Sparse Attention 等高效变体。

### 训练细节：学习率调度

原始 Transformer 使用 warm-up + decay 策略：

$$lr = d_{model}^{-0.5} \cdot \min(\text{step}^{-0.5}, \;\text{step} \cdot \text{warmup\_steps}^{-1.5})$$

```python
import torch

class TransformerLRScheduler:
    """原始 Transformer 论文中的学习率调度。"""
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self.d_model ** (-0.5) * min(
            self.step_num ** (-0.5),
            self.step_num * self.warmup_steps ** (-1.5)
        )
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr

# 演示
dummy = torch.nn.Linear(512, 512)
opt = torch.optim.Adam(dummy.parameters(), lr=0)
sched = TransformerLRScheduler(opt, d_model=512, warmup_steps=4000)

print("学习率变化:")
for s in [1, 100, 1000, 4000, 8000, 20000, 50000]:
    sched.step_num = s - 1
    lr = sched.step()
    print(f"  Step {s:>6d}: lr = {lr:.6f}")
```

前 warmup 步线性增大学习率，之后按平方根衰减。Post-LN 训练严重依赖 warmup；Pre-LN 则不那么敏感。

---

## 本章小结

| 概念 | 一句话总结 |
|------|-----------|
| **Transformer** | 纯注意力的编码器-解码器架构，完全抛弃 RNN，训练完全可并行 |
| **编码器层** | Self-Attention + FFN，各带残差连接和 Layer Norm |
| **解码器层** | Masked Self-Attention + Cross-Attention + FFN |
| **残差连接** | $x + F(x)$，保证梯度能跨层直接流动 |
| **Layer Norm** | 在特征维度上归一化，稳定训练 |
| **FFN** | 位置独立的两层全连接，隐藏维度通常 $4 \times d_{model}$ |
| **因果掩码** | 上三角设为 $-\infty$，防止解码器偷看未来 token |
| **交叉注意力** | Q 来自解码器，K/V 来自编码器，是信息传递的通道 |
| **Pre-LN** | 先 Norm 后子层，现代 LLM 的标准选择 |
| **参数量** | $\approx 12 N d_{model}^2$（不含 Embedding） |
| **并行优势** | Self-Attention 不依赖时间步顺序，所有位置一次算出 |

**下一章**：[位置编码](4_位置编码.md)——Transformer 没有 RNN 的递归结构，如何知道 token 的位置？正弦编码、可学习编码、RoPE 各有什么优劣？
