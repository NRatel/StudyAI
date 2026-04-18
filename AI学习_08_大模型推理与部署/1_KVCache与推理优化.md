# KV Cache 与推理优化

> **前置知识**：[05 GPT系列深度解析](../AI学习_05_GPT系列深度解析/README.md)

---

## 直觉与概述

### 自回归推理为什么慢？

大语言模型（如 GPT）在生成文本时采用**自回归（Autoregressive）**方式：每次只生成一个 token，然后把这个 token 加入输入，再生成下一个。

```
输入: "今天天气"
第1步: "今天天气" → "真"
第2步: "今天天气真" → "好"
第3步: "今天天气真好" → "啊"
...
```

问题在于：**每生成一个 token，模型都要对之前所有 token 做一次完整的注意力计算**。如果你的序列有 1000 个 token，生成第 1001 个 token 时，模型需要重新计算所有 1000 个 token 的 K 和 V 向量——尽管这 1000 个向量和上一步完全相同。

这就像**每次续写一篇论文时，都要从第一个字重新读一遍**。显然，人类不会这样做——我们会记住之前读过的内容。KV Cache 就是让模型也"记住"之前计算过的内容。

### 真正的瓶颈：内存带宽，而非算力

一个反直觉的事实：**大模型推理的瓶颈通常不是 GPU 算力不够，而是显存带宽不够**。

```
NVIDIA A100 (80GB):
  - 计算能力: 312 TFLOPS (FP16)
  - 显存带宽: 2.0 TB/s
  - 算术强度分界线: 312T / 2T = 156 FLOP/byte

自回归生成单个 token:
  - 需要加载: 整个模型权重（如 7B × 2 bytes = 14GB）
  - 计算量: 约 14 GFLOP（每个参数约 2 FLOP）
  - 算术强度: 14G / 14G = ~1 FLOP/byte

1 FLOP/byte << 156 FLOP/byte → 严重的内存带宽受限（Memory-bound）
```

换句话说：GPU 在 decode 阶段 90%+ 的时间在**等待数据从显存搬到计算单元**，而不是在"算"。这解释了为什么量化（减少模型大小）和批处理（提高计算密度）是推理优化的两大核心方向。

### 本章核心优化全景

```
                    自回归推理的慢
                         │
          ┌──────────────┼──────────────┐
          │              │              │
     重复计算 K/V    请求间等待     一次只出一个token
          │              │              │
     KV Cache       Continuous      推测解码
     PagedAttention  Batching     (Speculative
                                   Decoding)
```

---

## 严谨定义与原理

### 1. KV Cache — 缓存不变的 Key 和 Value

#### 注意力机制回顾

Transformer 的自注意力计算：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中 $Q = XW_Q$, $K = XW_K$, $V = XW_V$，$X$ 是输入序列的隐藏状态。

#### 核心观察

在自回归生成时，对于已经生成的 token，它们的 K 和 V 向量在后续步骤中**永远不会变化**（因为 Decoder 使用 Causal Mask，每个位置只能看到自己和之前的位置）。

```
步骤 t:   输入 = [x1, x2, ..., xt]
           K = [k1, k2, ..., kt]     # 需要所有位置的 K
           V = [v1, v2, ..., vt]     # 需要所有位置的 V
           Q = [qt]                   # 只需要新位置的 Q

步骤 t+1: 输入 = [x1, x2, ..., xt, x_{t+1}]
           K = [k1, k2, ..., kt, k_{t+1}]   # k1~kt 和上一步完全相同!
           V = [v1, v2, ..., vt, v_{t+1}]   # v1~vt 和上一步完全相同!
           Q = [q_{t+1}]                      # 只需要新位置的 Q
```

**KV Cache 的做法**：把每一步计算的 K 和 V 向量缓存起来，下一步只需要计算新 token 的 $k_{t+1}$ 和 $v_{t+1}$，然后拼接到缓存中。

#### 计算量对比

| 方式 | 每步计算量 | 总计算量 (生成 T 个 token) |
|------|-----------|---------------------------|
| 无 KV Cache | $O(t \cdot d^2)$（全序列） | $O(T^2 \cdot d^2)$ |
| 有 KV Cache | $O(d^2)$（只算新 token） | $O(T \cdot d^2)$ |

**从 $O(T^2)$ 降到 $O(T)$，这就是 KV Cache 的威力。**

#### KV Cache 的显存占用

对于一个标准 Transformer：

$$\text{KV Cache 大小} = 2 \times n_{\text{layers}} \times n_{\text{heads}} \times d_{\text{head}} \times \text{seq\_len} \times \text{batch\_size} \times \text{dtype\_size}$$

以 LLaMA-2 7B 为例（32 层、32 头、$d_{head}$=128、FP16）：

```
每个 token 的 KV Cache = 2 × 32 × 32 × 128 × 2 bytes = 512 KB
序列长度 2048 = 2048 × 512 KB = 1 GB
序列长度 4096 = 4096 × 512 KB = 2 GB
批大小 32 × 序列 4096 = 64 GB  ← 比模型权重(14GB)还大!
```

**关键结论**：随着序列长度和批大小增长，KV Cache 的显存占用会快速膨胀，甚至超过模型权重本身。这就是 PagedAttention 要解决的问题。

#### Prefill 与 Decode 两个阶段

自回归推理分为两个截然不同的阶段：

| 阶段 | Prefill（预填充） | Decode（解码） |
|------|-------------------|----------------|
| 做什么 | 一次性处理整个 prompt | 逐个生成新 token |
| 计算特征 | 大矩阵乘法，计算密集 | 小向量乘大矩阵，内存密集 |
| 瓶颈 | 通常是算力（Compute-bound） | 几乎总是内存带宽（Memory-bound） |
| KV Cache | 一次性填充所有 prompt 的 KV | 每步追加一个 KV |
| GPU 利用率 | 高（大块计算） | 低（大量时间在搬数据） |

这两个阶段的性能特征完全不同，后面会看到推理框架如何分别优化它们。

### 2. PagedAttention — 像操作系统管理内存一样管理 KV Cache

#### 朴素 KV Cache 的浪费

在朴素实现中，每个请求预分配的 KV Cache 空间 = 最大可能序列长度。但实际输出长度是不可预知的：

```
请求 A: 预分配 2048 token 的 KV Cache 空间
         实际只用了 200 → 浪费 90%+
请求 B: 预分配 2048 token 的 KV Cache 空间
         实际用了 1500 → 浪费 25%
```

研究表明，朴素 KV Cache 管理的**显存浪费率通常超过 60%**（Kwon et al., 2023）。

#### PagedAttention 的灵感

PagedAttention（Kwon et al., 2023，vLLM 的核心）直接借鉴了操作系统的**虚拟内存和分页**思想：

| 操作系统概念 | PagedAttention 对应 |
|-------------|---------------------|
| 虚拟内存页 | KV Block（固定大小的 KV Cache 块，如 16 tokens） |
| 页表 | Block Table（记录逻辑块到物理块的映射） |
| 按需分页 | 只在需要时分配新的 KV Block |
| 内存碎片整理 | 物理块不需要连续，通过映射表寻址 |

```
朴素方式（连续分配）:
  请求A: [████████░░░░░░░░]    请求B: [██████░░░░░░░░░░]
                  ↑ 浪费                       ↑ 浪费

PagedAttention（分页分配）:
  Block Pool: [A1][B1][A2][B2][A3][B3][空][空]
  请求A 的 Block Table: A1 → A2 → A3  (按需增长)
  请求B 的 Block Table: B1 → B2 → B3
  → 几乎没有内部碎片
```

**核心优势**：
1. **几乎零浪费**：只分配实际使用的块，内部碎片仅最后一个块
2. **更大的批大小**：同样的显存可以服务更多并发请求
3. **共享前缀**：多个请求共享相同 prompt 时，可以共享 KV Cache 块（Copy-on-Write）
4. 实测 vLLM 比 HuggingFace 推理吞吐量提升 **2~24 倍**

### 3. Continuous Batching — 动态批处理

#### 传统静态批处理的问题

```
时间线:  ═══════════════════════════════════════════>
请求 A:  [████████████████████████]  (生成 200 tokens)
请求 B:  [████████████]              (生成 100 tokens)
请求 C:  [████████████████]          (生成 150 tokens)

静态批处理: 三个请求同时开始，但 B 在 100 步后就完成了
           → B 完成后，它的 GPU 槽位闲置，直到 A 完成才能接新请求
           → GPU 利用率低，尾部延迟高
```

#### Continuous Batching 的解决方案

```
时间线:  ═══════════════════════════════════════════>
请求 A:  [████████████████████████]
请求 B:  [████████████]
请求 C:  [████████████████]
请求 D:              [████████████████████]    ← B完成后立即加入
请求 E:                  [████████████]        ← C完成后立即加入

→ 每一步（iteration-level）检查哪些请求完成，立即替换为新请求
→ GPU 始终满载，吞吐量大幅提升
```

**Continuous Batching**（也叫 Iteration-level Scheduling）在每个 decode step 检查批中的请求状态：
- 完成的请求立即移出，释放资源
- 新到达的请求立即加入
- 批大小动态变化，始终保持 GPU 高利用率

实测效果：相比静态批处理，吞吐量提升 **2~5 倍**。

### 4. 推测解码（Speculative Decoding）

#### 动机

自回归解码最根本的限制是**串行的**：必须先生成 token $t$，才能生成 token $t+1$。

但很多时候，接下来几个 token 是高度可预测的。比如生成 "The capital of France is"，接下来几乎一定是 " Paris"。能不能让一个小模型先"猜"几步，然后用大模型一次性验证？

#### 核心思路（Leviathan et al., 2023; Chen et al., 2023）

```
传统自回归（大模型 M_p，每步一个 token）:
  Step 1: M_p("The") → "capital"
  Step 2: M_p("The capital") → "of"
  Step 3: M_p("The capital of") → "France"
  Step 4: M_p("The capital of France") → "is"
  Step 5: M_p("The capital of France is") → "Paris"
  → 5 次大模型前向传播

推测解码（大模型 M_p + 小模型 M_q）:
  猜测阶段: M_q 连续猜 γ=4 个候选 token: ["capital", "of", "France", "is"]
            → 4 次小模型前向传播（很快）
  验证阶段: M_p 一次前向传播，并行检查这 4 个 token
            → 1 次大模型前向传播
            → 全部接受 → 相当于 1 步完成 4+ 个 token!
```

#### 数学保证——采样一致性

推测解码最精妙的地方是：**生成结果的分布与单独使用大模型完全相同**，没有任何质量损失。

验证规则——对于每个候选 token $x$：

$$\text{accept\_prob}(x) = \min\left(1, \frac{p(x)}{q(x)}\right)$$

其中 $p(x)$ 是大模型的概率，$q(x)$ 是小模型的概率。

- 如果 $p(x) \geq q(x)$：一定接受（大模型也同意小模型的选择）
- 如果 $p(x) < q(x)$：以 $p(x)/q(x)$ 的概率接受

如果被拒绝，则从修正分布中重新采样：

$$p'(x) = \text{normalize}\left(\max(0, p(x) - q(x))\right)$$

**这个拒绝采样方案在数学上保证了最终输出等价于从 $p(x)$ 中采样。**

#### 加速效果

| 场景 | 典型加速比 | 说明 |
|------|-----------|------|
| 代码生成 | 2~3x | 代码模式性强，小模型猜中率高 |
| 对话 | 1.5~2x | 中等可预测性 |
| 创意写作 | 1.2~1.5x | 不可预测性高，猜中率低 |

关键参数：
- **猜测长度 $\gamma$**：通常 4~8，太短加速不明显，太长浪费小模型计算
- **小模型选择**：同系列的小版本（如 LLaMA-7B 猜、LLaMA-70B 验）效果最好
- **接受率**：猜中率越高加速越大，通常 60~80%

### 5. 其他推理优化技术简述

| 技术 | 核心思路 | 效果 |
|------|---------|------|
| Flash Attention | 利用 GPU 内存层次（SRAM vs HBM），减少数据搬运 | 训练+推理均有效，内存节省 5~20x |
| GQA (Grouped Query Attention) | 多个 Q 头共享一组 K/V 头 | 减少 KV Cache 大小，LLaMA-2 70B 使用 |
| MQA (Multi-Query Attention) | 所有 Q 头共享同 1 组 K/V | 极致压缩 KV Cache，但精度可能下降 |
| Prefix Caching | 缓存公共系统 prompt 的 KV Cache | 相同系统 prompt 的请求直接复用 |
| Chunked Prefill | 将长 prompt 的 prefill 分块执行 | 避免长 prompt 阻塞 decode 请求 |

---

## Python 代码示例

### 示例 1：手动实现 KV Cache 推理 vs 无缓存推理

```python
import torch
import torch.nn as nn
import time

class SimpleAttention(nn.Module):
    """简化的因果自注意力，演示 KV Cache 的作用"""
    def __init__(self, d_model=256, n_heads=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, kv_cache=None):
        """
        x: [batch, seq_len, d_model]
           无缓存时 seq_len=全序列长度, 有缓存时 seq_len=1（只有新token）
        kv_cache: (cached_k, cached_v) 或 None
        返回: output, (updated_k, updated_v)
        """
        B, S, D = x.shape
        q = self.W_q(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        k = self.W_k(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        v = self.W_v(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        # 如果有缓存，拼接历史 K/V
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2)  # 在序列维度拼接
            v = torch.cat([cached_v, v], dim=2)

        # 标准注意力计算（Q 只有新位置，K/V 包含全部历史）
        scale = self.d_head ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.W_o(out), (k, v)

# ========== 对比实验 ==========
d_model = 256
seq_len = 512  # 模拟生成 512 个 token
attn = SimpleAttention(d_model).eval()

# 方法 1: 无 KV Cache（每步重新计算所有历史 token 的 K/V）
torch.manual_seed(42)
all_hidden = [torch.randn(1, 1, d_model)]
t0 = time.time()
for step in range(1, seq_len):
    full_input = torch.cat(all_hidden, dim=1)   # 拼接所有历史
    with torch.no_grad():
        out, _ = attn(full_input)                # 对整个序列做注意力
    all_hidden.append(torch.randn(1, 1, d_model))
t_no_cache = time.time() - t0

# 方法 2: 有 KV Cache（每步只计算新 token 的 K/V）
torch.manual_seed(42)
hidden = torch.randn(1, 1, d_model)
kv = None
t0 = time.time()
for step in range(seq_len):
    with torch.no_grad():
        out, kv = attn(hidden, kv_cache=kv)     # 只输入新 token
    hidden = torch.randn(1, 1, d_model)
t_with_cache = time.time() - t0

print(f"无 KV Cache: {t_no_cache:.3f}s")
print(f"有 KV Cache: {t_with_cache:.3f}s")
print(f"加速比: {t_no_cache / t_with_cache:.1f}x")
print(f"\n理论分析: 无缓存总注意力计算 ~ O(n^2), 有缓存 ~ O(n)")
print(f"KV Cache 最终大小: {kv[0].shape}")
# 典型输出: 无缓存约 2~3s, 有缓存约 0.3s, 加速约 6~10x
```

### 示例 2：KV Cache 显存占用计算器

```python
def kv_cache_memory(
    model_name: str,
    n_layers: int,
    n_kv_heads: int,
    d_head: int,
    seq_len: int,
    batch_size: int = 1,
    dtype_bytes: int = 2,  # FP16 = 2, INT8 = 1
) -> dict:
    """
    计算 KV Cache 的显存占用。

    公式: 2(K和V) x n_layers x n_kv_heads x d_head x seq_len x batch x dtype
    """
    per_token = 2 * n_layers * n_kv_heads * d_head * dtype_bytes
    per_request = per_token * seq_len
    total = per_request * batch_size
    return {
        "model": model_name,
        "per_token_KB": per_token / 1024,
        "per_request_MB": per_request / (1024 ** 2),
        "total_GB": total / (1024 ** 3),
    }

# ========== 各模型 KV Cache 对比 ==========
models = [
    ("LLaMA-2 7B (MHA)",   32, 32,  128),  # 32层, 32个KV头, d_head=128
    ("LLaMA-2 13B (MHA)",  40, 40,  128),
    ("LLaMA-2 70B (GQA)",  80,  8,  128),  # GQA: 只有 8 个 KV 头
    ("Mistral 7B (GQA)",   32,  8,  128),  # GQA: 只有 8 个 KV 头
    ("GPT-3 175B (MHA)",   96, 96,  128),
    ("LLaMA-3 8B (GQA)",   32,  8,  128),
]

seq_len = 4096
batch_size = 32

print(f"条件: seq_len={seq_len}, batch_size={batch_size}, FP16")
print(f"{'模型':<24} {'每token':>10} {'每请求':>10} {'总KV Cache':>12}")
print("-" * 60)

for name, layers, kv_heads, d_head in models:
    r = kv_cache_memory(name, layers, kv_heads, d_head, seq_len, batch_size)
    print(f"{r['model']:<24} {r['per_token_KB']:>8.0f} KB"
          f" {r['per_request_MB']:>8.0f} MB {r['total_GB']:>10.1f} GB")

# 关键发现:
# LLaMA-2 7B (MHA):   512 KB/token,  2048 MB/req,  64.0 GB total
# LLaMA-2 70B (GQA):  160 KB/token,   640 MB/req,  20.0 GB total  ← GQA 节省 3.2x
# Mistral 7B (GQA):   128 KB/token,   512 MB/req,  16.0 GB total  ← 比 7B MHA 少 4x
print("\n结论: GQA 将 KV Cache 减少到 MHA 的 1/4 ~ 1/8，是长序列/高并发的关键优化")
```

### 示例 3：推测解码核心算法实现

```python
import torch
import torch.nn.functional as F

def speculative_decode(
    target_logits_fn,   # 大模型: tokens -> logits [1, seq_len, vocab]
    draft_logits_fn,    # 小模型: tokens -> logits [1, seq_len, vocab]
    prefix: list,       # 已有的 token 序列
    gamma: int = 4,     # 猜测步数
    temperature: float = 1.0,
) -> list:
    """
    推测解码的核心算法（Leviathan et al., 2023）。

    流程:
    1. 小模型连续猜 gamma 个 token
    2. 大模型一次前向传播验证
    3. 拒绝采样保证输出分布 = 大模型分布

    返回: 本轮新接受的 token 列表
    """
    # === 第 1 步: 小模型连续猜测 γ 个 token ===
    draft_tokens = []
    draft_probs = []
    current = list(prefix)

    for _ in range(gamma):
        logits = draft_logits_fn(torch.tensor([current]))
        q = F.softmax(logits[0, -1] / temperature, dim=-1)
        token = torch.multinomial(q, 1).item()
        draft_tokens.append(token)
        draft_probs.append(q)
        current.append(token)

    # === 第 2 步: 大模型一次前向传播验证所有候选 ===
    all_tokens = prefix + draft_tokens
    target_logits = target_logits_fn(torch.tensor([all_tokens]))
    start = len(prefix) - 1
    target_probs = [
        F.softmax(target_logits[0, start + i] / temperature, dim=-1)
        for i in range(gamma + 1)
    ]

    # === 第 3 步: 逐个验证（拒绝采样） ===
    accepted = []
    for i in range(gamma):
        token = draft_tokens[i]
        p_i = target_probs[i][token].item()   # 大模型概率
        q_i = draft_probs[i][token].item()    # 小模型概率

        # 接受概率 = min(1, p/q)
        accept_prob = min(1.0, p_i / (q_i + 1e-10))

        if torch.rand(1).item() < accept_prob:
            accepted.append(token)  # 接受
        else:
            # 拒绝: 从修正分布 max(0, p - q) 中重新采样
            residual = torch.clamp(target_probs[i] - draft_probs[i], min=0)
            residual = residual / (residual.sum() + 1e-10)
            new_token = torch.multinomial(residual, 1).item()
            accepted.append(new_token)
            return accepted  # 拒绝后停止本轮

    # 所有猜测都被接受! 额外采样一个 bonus token
    bonus = torch.multinomial(target_probs[gamma], 1).item()
    accepted.append(bonus)
    return accepted

# ========== 模拟演示 ==========
vocab_size = 100
torch.manual_seed(42)

# 模拟大模型（分布更集中/更确定）和小模型（分布更分散/更不确定）
target_dist = F.softmax(torch.randn(vocab_size) * 3.0, dim=-1)  # 温度低 → 更集中
draft_dist  = F.softmax(torch.randn(vocab_size) * 1.5, dim=-1)  # 温度高 → 更分散

def mock_target(tokens):
    B, S = tokens.shape
    return target_dist.log().unsqueeze(0).unsqueeze(0).expand(B, S, -1)

def mock_draft(tokens):
    B, S = tokens.shape
    return draft_dist.log().unsqueeze(0).unsqueeze(0).expand(B, S, -1)

# 运行多轮推测解码
gamma = 4
total_tokens = 0
total_rounds = 20

for r in range(total_rounds):
    result = speculative_decode(mock_target, mock_draft, [0], gamma=gamma)
    total_tokens += len(result)
    status = "全部接受+bonus" if len(result) > gamma else f"第{len(result)}个被拒"
    print(f"轮次 {r+1:2d}: 生成 {len(result)} tokens ({status})")

avg = total_tokens / total_rounds
print(f"\n平均每轮: {avg:.1f} tokens (最大可能: {gamma + 1})")
print(f"等效加速比 (忽略小模型开销): ~{avg:.1f}x")
```

### 示例 4：Continuous Batching 模拟对比

```python
import random
import copy

class Request:
    """模拟一个推理请求"""
    def __init__(self, id, arrival_time, total_tokens):
        self.id = id
        self.arrival = arrival_time
        self.total_tokens = total_tokens
        self.generated = 0
        self.start_time = None
        self.end_time = None

def simulate_batching(requests, max_batch_size, mode="static"):
    """
    模拟静态 vs 连续批处理的性能差异。

    参数:
        requests: 请求列表
        max_batch_size: GPU 最大并发请求数
        mode: "static"（一批做完才接新批）或 "continuous"（随到随入）
    返回: 统计指标字典
    """
    clock = 0
    queue = sorted(requests, key=lambda r: r.arrival)
    q_idx = 0
    active = []
    pending = []
    completed = []
    total_steps = 0
    total_idle_slots = 0

    while q_idx < len(queue) or active or pending:
        # 将已到达的请求加入 pending
        while q_idx < len(queue) and queue[q_idx].arrival <= clock:
            pending.append(queue[q_idx])
            q_idx += 1

        if mode == "continuous":
            # 连续批处理：有空位就立即加入
            while pending and len(active) < max_batch_size:
                r = pending.pop(0)
                r.start_time = clock
                active.append(r)
        elif mode == "static":
            # 静态批处理：当前批为空才组新批
            if not active and pending:
                batch = pending[:max_batch_size]
                pending = pending[max_batch_size:]
                for r in batch:
                    r.start_time = clock
                active = batch

        if not active:
            # 没有活跃请求，快进到下一个请求到达时间
            if q_idx < len(queue):
                clock = queue[q_idx].arrival
            elif pending:
                pass  # static 模式下 pending 非空但 active 空的不会出现
            else:
                break
            continue

        # 执行一步 decode
        total_steps += 1
        total_idle_slots += max_batch_size - len(active)

        still_active = []
        for r in active:
            r.generated += 1
            if r.generated >= r.total_tokens:
                r.end_time = clock + 1
                completed.append(r)
            else:
                still_active.append(r)
        active = still_active
        clock += 1

    # 统计
    latencies = [r.end_time - r.arrival for r in completed]
    utilization = 1 - total_idle_slots / (total_steps * max_batch_size) if total_steps else 0

    return {
        "total_steps": total_steps,
        "gpu_utilization": utilization,
        "avg_latency": sum(latencies) / len(latencies) if latencies else 0,
        "max_latency": max(latencies) if latencies else 0,
        "throughput": len(completed) / total_steps if total_steps else 0,
    }

# ========== 模拟 ==========
random.seed(42)
N = 30
max_batch = 8

requests_template = [
    Request(i, random.randint(0, 60), random.randint(10, 120))
    for i in range(N)
]

for mode in ["static", "continuous"]:
    reqs = copy.deepcopy(requests_template)
    stats = simulate_batching(reqs, max_batch, mode)
    print(f"=== {mode.upper()} Batching ===")
    print(f"  总 decode 步数:  {stats['total_steps']}")
    print(f"  GPU 利用率:      {stats['gpu_utilization']:.1%}")
    print(f"  平均延迟:        {stats['avg_latency']:.1f} steps")
    print(f"  最大延迟:        {stats['max_latency']} steps")
    print(f"  吞吐量:          {stats['throughput']:.2f} req/step")
    print()

# 典型结果:
# STATIC:     GPU利用率 ~45%, 平均延迟 ~90 steps
# CONTINUOUS: GPU利用率 ~80%, 平均延迟 ~55 steps
```

---

## 工程师视角

### 推理优化的优先级

在实际部署中，优化的投入产出比（从高到低）：

| 优先级 | 优化 | 效果 | 实施难度 |
|--------|------|------|----------|
| 1 | KV Cache（基础） | 必须有，否则无法使用 | 所有框架已内置 |
| 2 | 量化（INT8/INT4） | 减少 2~4x 内存，提升 2~3x 吞吐 | 通常一行配置 |
| 3 | 换用高效框架（vLLM） | 2~24x 吞吐提升 | 更换推理引擎 |
| 4 | Flash Attention | 内存节省，速度提升 20~40% | 框架通常已集成 |
| 5 | 推测解码 | 1.5~3x 延迟降低 | 需要选择/训练 draft 模型 |

### Prefill vs Decode 的不同优化策略

这是工程部署中最容易忽略的点：

```
场景 1: 长 prompt + 短回复 (如 RAG 检索增强生成)
  → Prefill 耗时占主导
  → 优化方向: Flash Attention, Chunked Prefill, Prefix Caching

场景 2: 短 prompt + 长回复 (如 创意写作、代码生成)
  → Decode 耗时占主导
  → 优化方向: 量化、推测解码、更大 batch size

场景 3: 高并发在线服务
  → 吞吐量是关键
  → 优化方向: Continuous Batching + PagedAttention + 量化
```

### KV Cache 是显存的隐形杀手

初学者常见误解："模型权重 14GB，那 80GB 的 A100 应该绑绑有余。"

实际情况：

```
LLaMA-2 7B, FP16:
  模型权重:                         14 GB
  KV Cache (batch=1, seq=4096):      2 GB   ← 看起来不多
  KV Cache (batch=32, seq=4096):    64 GB   ← 超过模型权重 4.5 倍!

→ 实际能服务的最大并发数，往往由 KV Cache 大小决定
→ 这就是 GQA（减少 KV 头数）和 PagedAttention（减少浪费）如此重要的原因
```

### 推测解码的适用场景判断

推测解码不是万能的：

```
适合推测解码:
  ✓ 代码生成（模式性强，猜中率高）
  ✓ 翻译（源文本提供强约束）
  ✓ 格式化输出（JSON、XML）
  ✓ batch_size=1 的低延迟场景

不适合推测解码:
  ✗ 高并发高吞吐场景（GPU 已通过大 batch 满载）
  ✗ 高度创意性任务（猜中率低）
  ✗ 小模型和大模型差异太大（接受率 < 50% 就不划算）
```

### 选型快速决策

```
你需要部署大模型推理？
  │
  ├─ 高吞吐在线服务 → vLLM（PagedAttention + Continuous Batching）
  │
  ├─ 极致低延迟 → TensorRT-LLM（编译优化） + 推测解码
  │
  ├─ 消费级 GPU / CPU → llama.cpp（GGUF 量化，资源友好）
  │
  └─ 快速实验 → HuggingFace Transformers（最灵活，但最慢）
```

---

## 本章小结

| 概念 | 一句话总结 |
|------|-----------|
| 自回归瓶颈 | 每步只生成一个 token，且 decode 阶段瓶颈在内存带宽而非算力 |
| KV Cache | 缓存已计算的 K/V 向量，避免重复计算，从 O(T^2) 降到 O(T) |
| Prefill vs Decode | Prefill 计算密集、Decode 内存密集，需要针对性优化 |
| PagedAttention | 用分页管理 KV Cache 显存，消除碎片浪费，提升并发量 2~24x |
| Continuous Batching | 动态替换完成的请求保持 GPU 满载，吞吐提升 2~5x |
| 推测解码 | 小模型猜大模型验，拒绝采样保证无损，延迟降低 1.5~3x |
| GQA/MQA | 减少 KV 头数直接减少 KV Cache 大小，长序列高并发的关键优化 |
