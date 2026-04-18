# RNN 循环神经网络

## 直觉与概述

### 核心问题

你已经学了全连接网络和 CNN。全连接网络擅长处理固定长度的向量输入，CNN 擅长从网格状数据（如图像）中提取空间特征。但当你面对**序列数据**时，它们就束手无策了：

- **变长输入**：一句话可能 5 个词，也可能 50 个词。全连接网络需要固定大小的输入向量，把短句子补零、长句子截断？太粗暴了。
- **顺序关系**："狗咬人" 和 "人咬狗" 包含完全相同的词，但意思截然不同。全连接网络看到的只是一个词袋（bag of words），完全丢失了顺序信息。
- **参数爆炸**：如果你想用全连接网络处理一个 100 步的序列，每步有 256 维特征，那输入层就有 $100 \times 256 = 25600$ 维。而且这种方式完全无法处理第 101 步——模型结构已经写死了。
- **无法共享时间步之间的知识**：第 3 个词出现 "not" 学到的否定含义，不会自动泛化到第 7 个词出现 "not" 的情况——因为全连接网络中这两个位置使用的是完全不同的权重。

### 阅读的类比

想象你在读一句话：

> "The cat, which had been sleeping on the warm windowsill all afternoon, suddenly **jumped**."

当你读到 "jumped" 时，你能理解这个动作的主语是 "cat"，尽管中间隔了十几个词。你是怎么做到的？因为你的大脑在逐词阅读时，一直在维护一个**隐含的理解状态**——你知道主语是谁、当前的语境是什么、哪些信息需要记住。

RNN 模仿的正是这个过程：

```
                      隐藏状态（"记忆"）
                      ┌──────────┐
                      │          │
                      v          │
输入 x₁ ──→ [ RNN 单元 ] ──→ 输出 y₁
              │
              │ 隐藏状态传递
              v
输入 x₂ ──→ [ RNN 单元 ] ──→ 输出 y₂
              │
              │ 隐藏状态传递
              v
输入 x₃ ──→ [ RNN 单元 ] ──→ 输出 y₃
              │
              ...
```

**关键点**：每个时间步使用的是**同一个** RNN 单元（参数共享），但隐藏状态 $h_t$ 在不断更新——它就是网络的"记忆"。

### 一句话总结

> **RNN = 参数共享 + 隐藏状态递推。** 同一组权重在每个时间步复用，隐藏状态 $h_t$ 将历史信息压缩编码并逐步传递，使网络能处理任意长度的序列输入。

---

## 严谨定义与原理

### 基本 RNN 结构

#### 核心公式

一个最简单（vanilla）的 RNN，在每个时间步 $t$ 做两件事：

**1. 更新隐藏状态**：

$$h_t = \tanh(W_{hh} \cdot h_{t-1} + W_{xh} \cdot x_t + b_h)$$

**2. 计算输出**：

$$y_t = W_{hy} \cdot h_t + b_y$$

其中：
- $x_t \in \mathbb{R}^d$：时间步 $t$ 的输入向量（$d$ 是输入维度）
- $h_t \in \mathbb{R}^n$：时间步 $t$ 的隐藏状态（$n$ 是隐藏层大小）
- $h_{t-1} \in \mathbb{R}^n$：上一个时间步的隐藏状态
- $y_t$：时间步 $t$ 的输出
- $W_{xh} \in \mathbb{R}^{n \times d}$：输入到隐藏的权重矩阵
- $W_{hh} \in \mathbb{R}^{n \times n}$：隐藏到隐藏的权重矩阵（这是 RNN 的"灵魂"）
- $W_{hy}$：隐藏到输出的权重矩阵
- $b_h, b_y$：偏置
- $h_0$：初始隐藏状态，通常初始化为零向量

**参数共享**：$W_{xh}$、$W_{hh}$、$W_{hy}$、$b_h$、$b_y$ 在所有时间步中都是同一组参数。这意味着：
- 无论序列多长，参数量恒定
- 在第 1 步学到的模式，自动适用于第 100 步

#### 为什么用 tanh 而不是 Sigmoid？

回忆 Sigmoid 的输出范围是 $(0, 1)$，而 tanh 的输出范围是 $(-1, 1)$。在 RNN 中 tanh 更好用，有两个原因：

1. **零中心**：tanh 的输出均值接近 0，而 Sigmoid 的输出均值是 0.5。非零中心的激活值会让梯度更新时只朝一个方向走（所有参数要么全增大，要么全减小），训练效率低。
2. **更大的梯度**：tanh 的导数最大值是 1（在 $z=0$ 处），而 Sigmoid 的导数最大值只有 0.25。在 RNN 这种需要跨多个时间步传播梯度的结构中，更大的梯度意味着更慢的衰减。

不过要注意，tanh 也只是缓解了梯度消失，并没有从根本上解决——后面会详细分析。

#### 展开图（Unrolled Diagram）

RNN 的结构看起来像一个"自环"。为了理解它的计算过程，把它沿时间轴展开：

```
折叠视图（Folded）:

     ┌──────┐
     │      ├──── y_t
x_t ─┤ RNN  │
     │      ├─┐
     └──────┘ │
       ▲      │
       └──────┘
      h_{t-1} → h_t


展开视图（Unrolled）:

h₀    h₁          h₂          h₃          h₄
 │     │           │           │           │
 ▼     ▼           ▼           ▼           ▼
[RNN] ──→ [RNN] ──→ [RNN] ──→ [RNN] ──→ [RNN]
 ▲          ▲          ▲          ▲          ▲
 │          │          │          │          │
 x₁         x₂         x₃         x₄         x₅
             │          │          │          │
             ▼          ▼          ▼          ▼
             y₁         y₂         y₃         y₄   ← 可选输出

注意：所有 [RNN] 共享同一组参数！
展开后看起来像一个很深的前馈网络，只不过每层权重相同。
```

展开图给你两个重要直觉：

1. **RNN 就是一个"深度"网络**——展开后有多少时间步就有"多少层"。这直接预示了为什么长序列会遇到梯度消失/爆炸问题。
2. **参数共享**——所有时间步的权重完全相同。这就像 CNN 中卷积核在空间维度上滑动一样，RNN 的权重在时间维度上"滑动"。

#### 隐藏状态 $h_t$ 的含义

$h_t$ 是对"到目前为止所有输入 $x_1, x_2, \dots, x_t$"的**压缩表示**。

用信息论的语言来说，$h_t$ 是序列 $(x_1, \dots, x_t)$ 的一个有损编码。之所以是"有损"的，是因为你把任意长度的序列压缩到了一个固定大小的向量 $\mathbb{R}^n$ 里——必然会丢失信息。这个压缩有多大的损失，很大程度上取决于：
- 隐藏层大小 $n$（越大，容量越大）
- 序列的实际长度（越长，越难压缩）
- 信息的时间跨度（需要"记住"多远之前的信息）

后面我们会看到，vanilla RNN 在序列超过 20~30 步时，$h_t$ 中几乎不包含早期输入的信息了——这就是长距离依赖问题的根源。

### 具体数值示例：手动追踪 RNN 的计算过程

为了把公式变成肌肉记忆，我们手算一个最简单的 RNN。

**设置**：输入维度 $d = 2$，隐藏层大小 $n = 2$，序列长度 $T = 3$。

```
参数：
W_xh = [[0.3, 0.5],    W_hh = [[0.2, 0.1],    b_h = [0.0, 0.0]
         [0.4, 0.6]]            [0.3, 0.4]]

初始隐藏状态：h₀ = [0.0, 0.0]

输入序列：
x₁ = [1.0, 0.5]
x₂ = [0.2, 0.8]
x₃ = [0.7, 0.3]
```

**时间步 1**：$h_1 = \tanh(W_{hh} \cdot h_0 + W_{xh} \cdot x_1 + b_h)$

$$W_{hh} \cdot h_0 = \begin{bmatrix} 0.2 & 0.1 \\ 0.3 & 0.4 \end{bmatrix} \begin{bmatrix} 0 \\ 0 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$

$$W_{xh} \cdot x_1 = \begin{bmatrix} 0.3 & 0.5 \\ 0.4 & 0.6 \end{bmatrix} \begin{bmatrix} 1.0 \\ 0.5 \end{bmatrix} = \begin{bmatrix} 0.3 \times 1.0 + 0.5 \times 0.5 \\ 0.4 \times 1.0 + 0.6 \times 0.5 \end{bmatrix} = \begin{bmatrix} 0.55 \\ 0.70 \end{bmatrix}$$

$$h_1 = \tanh\begin{pmatrix} 0.55 \\ 0.70 \end{pmatrix} = \begin{pmatrix} 0.4995 \\ 0.6044 \end{pmatrix}$$

> 第一步 $h_0 = 0$，所以 $W_{hh}$ 项消失了。$h_1$ 完全由 $x_1$ 决定。

**时间步 2**：$h_2 = \tanh(W_{hh} \cdot h_1 + W_{xh} \cdot x_2 + b_h)$

$$W_{hh} \cdot h_1 = \begin{bmatrix} 0.2 & 0.1 \\ 0.3 & 0.4 \end{bmatrix} \begin{bmatrix} 0.4995 \\ 0.6044 \end{bmatrix} = \begin{bmatrix} 0.2 \times 0.4995 + 0.1 \times 0.6044 \\ 0.3 \times 0.4995 + 0.4 \times 0.6044 \end{bmatrix} = \begin{bmatrix} 0.1603 \\ 0.3916 \end{bmatrix}$$

$$W_{xh} \cdot x_2 = \begin{bmatrix} 0.3 & 0.5 \\ 0.4 & 0.6 \end{bmatrix} \begin{bmatrix} 0.2 \\ 0.8 \end{bmatrix} = \begin{bmatrix} 0.46 \\ 0.56 \end{bmatrix}$$

$$h_2 = \tanh\begin{pmatrix} 0.1603 + 0.46 \\ 0.3916 + 0.56 \end{pmatrix} = \tanh\begin{pmatrix} 0.6203 \\ 0.9516 \end{pmatrix} = \begin{pmatrix} 0.5511 \\ 0.7404 \end{pmatrix}$$

> 关键观察：$h_2$ 既包含 $x_2$ 的信息（通过 $W_{xh} \cdot x_2$），也包含 $x_1$ 的信息（通过 $W_{hh} \cdot h_1$，而 $h_1$ 编码了 $x_1$）。

**时间步 3**：$h_3 = \tanh(W_{hh} \cdot h_2 + W_{xh} \cdot x_3 + b_h)$

$$W_{hh} \cdot h_2 = \begin{bmatrix} 0.2 \times 0.5511 + 0.1 \times 0.7404 \\ 0.3 \times 0.5511 + 0.4 \times 0.7404 \end{bmatrix} = \begin{bmatrix} 0.1843 \\ 0.4615 \end{bmatrix}$$

$$W_{xh} \cdot x_3 = \begin{bmatrix} 0.3 \times 0.7 + 0.5 \times 0.3 \\ 0.4 \times 0.7 + 0.6 \times 0.3 \end{bmatrix} = \begin{bmatrix} 0.36 \\ 0.46 \end{bmatrix}$$

$$h_3 = \tanh\begin{pmatrix} 0.5443 \\ 0.9215 \end{pmatrix} = \begin{pmatrix} 0.4966 \\ 0.7267 \end{pmatrix}$$

> $h_3$ 是整个序列 $(x_1, x_2, x_3)$ 的压缩表示。但 $x_1$ 的影响已经被两次 $W_{hh}$ 乘法和 tanh 压缩衰减了。

### RNN 的应用模式

RNN 的灵活性在于：你可以根据任务需要，选择在哪些时间步"读入"输入、在哪些时间步"吐出"输出。

```
┌─────────────────────────────────────────────────────────────────┐
│                        RNN 的四种应用模式                          │
├──────────────┬──────────────────────────────────────────────────┤
│              │                                                  │
│  一对多       │   输入: 一个向量（如图像特征）                        │
│  (1 → N)     │   输出: 一个序列（如图像描述的每个词）                 │
│              │                                                  │
│              │   [图像特征] → RNN → "a" → "cat" → "on" → "mat"  │
│              │                                                  │
├──────────────┼──────────────────────────────────────────────────┤
│              │                                                  │
│  多对一       │   输入: 一个序列（如电影评论的每个词）                 │
│  (N → 1)     │   输出: 一个向量（如情感分类：正面/负面）              │
│              │                                                  │
│              │   "this" → "movie" → "is" → "great" → [正面 0.9] │
│              │                                                  │
├──────────────┼──────────────────────────────────────────────────┤
│              │                                                  │
│  多对多       │   输入: 一个序列                                    │
│  (同步)      │   输出: 等长序列（每个输入位置都有输出）               │
│  (N → N)     │                                                  │
│              │   "I" → "love" → "cats"                          │
│              │    ↓       ↓       ↓                              │
│              │  PRP     VBP     NNS     ← 词性标注                │
│              │                                                  │
├──────────────┼──────────────────────────────────────────────────┤
│              │                                                  │
│  多对多       │   输入: 一个序列                                    │
│  (异步)      │   输出: 不等长的另一个序列                            │
│  (N → M)     │                                                  │
│              │   编码器: "I" → "love" → "cats" → [编码向量]        │
│              │   解码器: [编码向量] → "我" → "爱" → "猫"            │
│              │                                                  │
│              │   ← 机器翻译（Seq2Seq 架构，下一章详讲）             │
│              │                                                  │
└──────────────┴──────────────────────────────────────────────────┘
```

**多对一**（情感分析）是最常见的入门用法：序列逐步输入 RNN，最后一步的隐藏状态 $h_T$ 作为整个序列的表示，接一个全连接层做分类。

**同步多对多**（序列标注）也很常见：每个时间步都输出一个标签，用于命名实体识别（NER）、词性标注（POS tagging）等。

### 反向传播 Through Time（BPTT）

#### 展开后的反向传播

RNN 训练的核心算法叫 **BPTT（Backpropagation Through Time）**。本质上就是把 RNN 沿时间展开成一个"深层前馈网络"，然后用标准的反向传播。

假设我们在每个时间步都有一个损失 $L_t$（比如语言模型中，每一步都在预测下一个词），总损失为：

$$L = \sum_{t=1}^{T} L_t$$

对参数 $W_{hh}$ 的梯度：

$$\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^{T} \frac{\partial L_t}{\partial W_{hh}}$$

关键在于，$L_t$ 通过 $h_t$ 依赖于 $W_{hh}$，而 $h_t$ 又通过 $h_{t-1}$ 递归地依赖于 $W_{hh}$。展开链式法则：

$$\frac{\partial L_t}{\partial W_{hh}} = \sum_{k=1}^{t} \frac{\partial L_t}{\partial h_t} \cdot \frac{\partial h_t}{\partial h_k} \cdot \frac{\partial h_k}{\partial W_{hh}}$$

其中 $\frac{\partial h_t}{\partial h_k}$ 是从时间步 $k$ 到时间步 $t$ 的梯度传播，它是一系列雅可比矩阵的连乘：

$$\frac{\partial h_t}{\partial h_k} = \prod_{i=k+1}^{t} \frac{\partial h_i}{\partial h_{i-1}}$$

每个雅可比矩阵为：

$$\frac{\partial h_i}{\partial h_{i-1}} = \text{diag}(\tanh'(W_{hh} h_{i-1} + W_{xh} x_i + b_h)) \cdot W_{hh}$$

其中 $\text{diag}(\tanh'(\cdot))$ 是 tanh 导数构成的对角矩阵，$\tanh'(z) = 1 - \tanh^2(z)$。

#### 梯度消失问题：为什么 RNN 记不住长期依赖

看这个连乘：

$$\frac{\partial h_t}{\partial h_k} = \prod_{i=k+1}^{t} \text{diag}(\tanh'(\cdot)) \cdot W_{hh}$$

$\tanh'(z) = 1 - \tanh^2(z)$，其值域是 $(0, 1]$。最大值 1 仅在 $z = 0$ 时取到，实际中大多数值远小于 1。

假设简化情况：$\tanh'$ 的值约为 $\alpha$（$0 < \alpha \leq 1$），$W_{hh}$ 的最大特征值的绝对值为 $\lambda$。那么连乘项的量级大约是：

$$\left\| \frac{\partial h_t}{\partial h_k} \right\| \approx (\alpha \cdot \lambda)^{t-k}$$

- 如果 $\alpha \cdot \lambda < 1$（常见情况），梯度随 $t - k$ **指数级衰减**
- 如果 $\alpha \cdot \lambda > 1$，梯度随 $t - k$ **指数级增长**

**用具体数值说明**：

假设 $\alpha = 0.7$，$\lambda = 0.8$，那么 $\alpha \cdot \lambda = 0.56$。

| 时间步距离 $t - k$ | 梯度衰减因子 $0.56^{t-k}$ |
|:---:|:---:|
| 1 | 0.56 |
| 5 | 0.055 |
| 10 | 0.003 |
| 20 | $9.5 \times 10^{-6}$ |
| 30 | $3.2 \times 10^{-8}$ |
| 50 | $3.6 \times 10^{-13}$ |

**到 20 步以外，梯度就只剩百万分之一了。** 这意味着：
- $W_{hh}$ 几乎不会因为 20 步之前的输入而更新
- 网络事实上"看不到"20 步之前的信息
- 这不是网络"选择"忘记——而是数学上根本无法传播梯度

#### 梯度爆炸问题：更容易检测，更容易修复

如果 $\alpha \cdot \lambda > 1$，梯度会指数增长。例如 $\alpha \cdot \lambda = 1.5$：

| 时间步距离 $t - k$ | 梯度增长因子 $1.5^{t-k}$ |
|:---:|:---:|
| 10 | 57.7 |
| 20 | 3325.3 |
| 50 | $6.4 \times 10^{8}$ |

梯度爆炸会导致参数更新极大，模型训练时 loss 突然飙升到 NaN。

**好消息**：梯度爆炸比梯度消失容易处理得多——直接用**梯度裁剪**（gradient clipping）就行：

```python
# 梯度裁剪一行搞定
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
```

**坏消息**：梯度消失没有这么简单的解法。你不能"放大"梯度——因为你不知道那些消失的梯度"本来应该"是什么方向。这就是为什么需要 LSTM/GRU 这种结构性的解决方案（下一章）。

### BPTT 的时间截断（Truncated BPTT）

实际训练中，序列可能有几千甚至几万步（比如一篇长文章）。完整的 BPTT 需要：
1. 把整个序列都展开（内存爆炸）
2. 从最后一步一直回传到第一步（计算爆炸）

**解决方案：截断 BPTT**——只回传固定的 $k$ 步。

```
完整 BPTT（内存和计算量与序列长度成正比）：

h₀ → h₁ → h₂ → h₃ → h₄ → h₅ → h₆ → h₇ → h₈
←────────────────────────────────────────────────  梯度传播

截断 BPTT（k=3，每3步截断一次）：

第1段：h₀ → h₁ → h₂ → h₃       ← 前向传播后回传梯度（回传3步）
第2段：h₃ → h₄ → h₅ → h₆       ← 前向传播后回传梯度（回传3步）
第3段：h₆ → h₇ → h₈             ← 前向传播后回传梯度（回传2步）

注意：前向传播中 h₃ 的值传递给了第2段，但反向传播时梯度不会从第2段回传到第1段。
```

截断 BPTT 是一个工程妥协：牺牲长距离依赖的学习能力，换取可控的内存和计算量。在 PyTorch 中，你可以通过 `h = h.detach()` 来截断梯度图。

---

## Python 代码示例

### 示例 1：用 numpy 手写单层 RNN 前向传播

从零实现 RNN 的前向传播，逐时间步循环，对照公式理解每一步。

```python
import numpy as np

# ============================================================
# 用 numpy 手写 RNN 前向传播（逐时间步循环）
# 对应公式：h_t = tanh(W_hh @ h_{t-1} + W_xh @ x_t + b_h)
#           y_t = W_hy @ h_t + b_y
# ============================================================

np.random.seed(42)

# --- 超参数 ---
input_size = 3     # 输入维度 d
hidden_size = 4    # 隐藏层大小 n
output_size = 2    # 输出维度
seq_len = 5        # 序列长度 T

# --- 初始化参数 ---
# Xavier 初始化
W_xh = np.random.randn(hidden_size, input_size) * np.sqrt(2.0 / (input_size + hidden_size))
W_hh = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / (hidden_size + hidden_size))
b_h = np.zeros(hidden_size)

W_hy = np.random.randn(output_size, hidden_size) * np.sqrt(2.0 / (hidden_size + output_size))
b_y = np.zeros(output_size)

# --- 输入序列 ---
# shape: (seq_len, input_size) — 5 个时间步，每步 3 维输入
X = np.random.randn(seq_len, input_size)

# --- RNN 前向传播 ---
h = np.zeros(hidden_size)  # 初始隐藏状态 h₀ = 0

print("=" * 70)
print("RNN 前向传播（逐时间步）")
print("=" * 70)
print(f"输入序列 shape: ({seq_len}, {input_size})")
print(f"隐藏层大小: {hidden_size}")
print(f"输出维度: {output_size}\n")

hidden_states = [h.copy()]  # 保存所有隐藏状态
outputs = []

for t in range(seq_len):
    x_t = X[t]

    # 核心公式：h_t = tanh(W_hh @ h_{t-1} + W_xh @ x_t + b_h)
    z_t = W_hh @ h + W_xh @ x_t + b_h     # 线性组合
    h = np.tanh(z_t)                         # 非线性激活

    # 输出：y_t = W_hy @ h_t + b_y
    y_t = W_hy @ h + b_y

    hidden_states.append(h.copy())
    outputs.append(y_t.copy())

    print(f"时间步 t={t+1}:")
    print(f"  输入 x_t      = [{', '.join(f'{v:.4f}' for v in x_t)}]")
    print(f"  线性组合 z_t  = [{', '.join(f'{v:.4f}' for v in z_t)}]")
    print(f"  隐藏状态 h_t  = [{', '.join(f'{v:.4f}' for v in h)}]")
    print(f"  输出 y_t      = [{', '.join(f'{v:.4f}' for v in y_t)}]")
    print()

hidden_states = np.array(hidden_states)  # shape: (seq_len+1, hidden_size)
outputs = np.array(outputs)              # shape: (seq_len, output_size)

print("=" * 70)
print(f"所有隐藏状态 shape: {hidden_states.shape}")
print(f"所有输出 shape: {outputs.shape}")
print(f"\n观察：隐藏状态的范围始终在 [-1, 1] 内（tanh 的值域）")
print(f"  h 的最小值: {hidden_states[1:].min():.4f}")
print(f"  h 的最大值: {hidden_states[1:].max():.4f}")
```

### 示例 2：用 numpy 实现字符级语言模型

这是理解 RNN "记忆"能力的经典示例：输入一个字符序列，RNN 学会预测下一个字符。经过训练，RNN 能生成看起来"像模像样"的文本——因为它学到了字符之间的统计规律。

```python
import numpy as np

# ============================================================
# 字符级 RNN 语言模型（numpy 从零实现）
# 输入: 字符序列   输出: 下一个字符的概率分布
# 这个例子展示 RNN 如何"记住"上下文
# ============================================================

class CharRNN:
    """
    字符级 RNN 语言模型。
    - 输入：one-hot 编码的字符
    - 输出：下一个字符的概率分布（softmax）
    - 损失：交叉熵
    """

    def __init__(self, vocab_size, hidden_size, lr=0.01):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.lr = lr

        # Xavier 初始化
        scale_xh = np.sqrt(2.0 / (vocab_size + hidden_size))
        scale_hh = np.sqrt(2.0 / (hidden_size + hidden_size))
        scale_hy = np.sqrt(2.0 / (hidden_size + vocab_size))

        self.W_xh = np.random.randn(hidden_size, vocab_size) * scale_xh
        self.W_hh = np.random.randn(hidden_size, hidden_size) * scale_hh
        self.b_h = np.zeros(hidden_size)

        self.W_hy = np.random.randn(vocab_size, hidden_size) * scale_hy
        self.b_y = np.zeros(vocab_size)

        # Adagrad 的累积梯度平方和（用于自适应学习率）
        self.mW_xh = np.zeros_like(self.W_xh)
        self.mW_hh = np.zeros_like(self.W_hh)
        self.mb_h = np.zeros_like(self.b_h)
        self.mW_hy = np.zeros_like(self.W_hy)
        self.mb_y = np.zeros_like(self.b_y)

    def forward(self, inputs, h_prev):
        """
        前向传播。
        inputs: 字符索引列表（不含最后一个，因为最后一个是标签）
        h_prev: 上一个时间步的隐藏状态
        返回: xs, hs, ys, ps（用于反向传播）
        """
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = h_prev.copy()

        for t in range(len(inputs)):
            # one-hot 编码
            xs[t] = np.zeros(self.vocab_size)
            xs[t][inputs[t]] = 1.0

            # h_t = tanh(W_hh @ h_{t-1} + W_xh @ x_t + b_h)
            hs[t] = np.tanh(
                self.W_hh @ hs[t - 1] + self.W_xh @ xs[t] + self.b_h
            )

            # y_t = W_hy @ h_t + b_y （logits）
            ys[t] = self.W_hy @ hs[t] + self.b_y

            # softmax: 把 logits 转成概率分布
            exp_ys = np.exp(ys[t] - np.max(ys[t]))  # 减最大值防溢出
            ps[t] = exp_ys / np.sum(exp_ys)

        return xs, hs, ys, ps

    def loss_and_grads(self, inputs, targets, h_prev):
        """
        前向 + 反向传播，返回损失和梯度。
        inputs:  输入字符索引列表
        targets: 目标字符索引列表（inputs 右移一位）
        """
        xs, hs, ys, ps = self.forward(inputs, h_prev)
        T = len(inputs)

        # --- 计算损失（交叉熵）---
        loss = 0.0
        for t in range(T):
            loss -= np.log(ps[t][targets[t]] + 1e-8)

        # --- 反向传播 ---
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        db_h = np.zeros_like(self.b_h)
        dW_hy = np.zeros_like(self.W_hy)
        db_y = np.zeros_like(self.b_y)

        dh_next = np.zeros(self.hidden_size)

        for t in reversed(range(T)):
            # softmax + 交叉熵的梯度：dy = p - one_hot(target)
            dy = ps[t].copy()
            dy[targets[t]] -= 1.0  # 这就是 softmax 交叉熵的导数

            # 输出层梯度
            dW_hy += np.outer(dy, hs[t])
            db_y += dy

            # 隐藏层梯度
            dh = self.W_hy.T @ dy + dh_next  # 来自输出 + 来自下一个时间步
            dh_raw = (1 - hs[t] ** 2) * dh    # 穿过 tanh: tanh'(z) = 1 - tanh²(z)

            dW_xh += np.outer(dh_raw, xs[t])
            dW_hh += np.outer(dh_raw, hs[t - 1])
            db_h += dh_raw

            # 传给前一个时间步
            dh_next = self.W_hh.T @ dh_raw

        # 梯度裁剪（防止爆炸）
        for grad in [dW_xh, dW_hh, db_h, dW_hy, db_y]:
            np.clip(grad, -5, 5, out=grad)

        return loss, dW_xh, dW_hh, db_h, dW_hy, db_y, hs[T - 1]

    def update(self, dW_xh, dW_hh, db_h, dW_hy, db_y):
        """Adagrad 参数更新。"""
        for param, grad, mem in [
            (self.W_xh, dW_xh, 'mW_xh'), (self.W_hh, dW_hh, 'mW_hh'),
            (self.b_h, db_h, 'mb_h'), (self.W_hy, dW_hy, 'mW_hy'),
            (self.b_y, db_y, 'mb_y')
        ]:
            m = getattr(self, mem)
            m += grad * grad
            param -= self.lr * grad / (np.sqrt(m) + 1e-8)

    def sample(self, h, seed_idx, length):
        """
        从模型中采样生成文本。
        h: 起始隐藏状态
        seed_idx: 起始字符的索引
        length: 生成长度
        """
        x = np.zeros(self.vocab_size)
        x[seed_idx] = 1.0
        indices = []

        for _ in range(length):
            h = np.tanh(self.W_hh @ h + self.W_xh @ x + self.b_h)
            y = self.W_hy @ h + self.b_y
            exp_y = np.exp(y - np.max(y))
            p = exp_y / np.sum(exp_y)

            # 按概率分布采样（不是 argmax，这样更有多样性）
            idx = np.random.choice(self.vocab_size, p=p)
            indices.append(idx)

            x = np.zeros(self.vocab_size)
            x[idx] = 1.0

        return indices


# ============================================================
# 训练：在一段文本上学习字符级语言模型
# ============================================================

# 训练数据
text = """To be or not to be that is the question.
Whether tis nobler in the mind to suffer
the slings and arrows of outrageous fortune
or to take arms against a sea of troubles."""

# 构建字符到索引的映射
chars = sorted(set(text))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for ch, i in char_to_idx.items()}
vocab_size = len(chars)

print("=" * 70)
print("字符级 RNN 语言模型")
print("=" * 70)
print(f"文本长度: {len(text)} 字符")
print(f"词汇表大小: {vocab_size}")
print(f"词汇表: {''.join(chars)}\n")

# 超参数
hidden_size = 64
seq_len = 25  # 每次训练用 25 个字符的片段
lr = 0.1

model = CharRNN(vocab_size, hidden_size, lr)

# 编码文本
data = [char_to_idx[ch] for ch in text]

# 训练循环
n_iters = 5000
print_every = 500
smooth_loss = -np.log(1.0 / vocab_size) * seq_len  # 初始 loss 估计

np.random.seed(42)
pointer = 0
h_prev = np.zeros(hidden_size)

print("开始训练...\n")

for iteration in range(1, n_iters + 1):
    # 如果到了文本末尾，重置
    if pointer + seq_len + 1 >= len(data):
        pointer = 0
        h_prev = np.zeros(hidden_size)

    inputs = data[pointer:pointer + seq_len]
    targets = data[pointer + 1:pointer + seq_len + 1]

    # 前向 + 反向 + 更新
    loss, dW_xh, dW_hh, db_h, dW_hy, db_y, h_prev = \
        model.loss_and_grads(inputs, targets, h_prev)
    model.update(dW_xh, dW_hh, db_h, dW_hy, db_y)

    smooth_loss = 0.999 * smooth_loss + 0.001 * loss
    pointer += seq_len

    if iteration % print_every == 0 or iteration == 1:
        # 采样生成一段文本
        sample_indices = model.sample(h_prev, inputs[0], 60)
        sample_text = ''.join(idx_to_char[i] for i in sample_indices)

        print(f"迭代 {iteration:>5d} | 平滑损失: {smooth_loss:.2f}")
        print(f"  生成样本: \"{sample_text}\"")
        print()

print("=" * 70)
print("观察：")
print("- 早期生成的文本是随机字符")
print("- 随着训练进行，模型学会了单词拼写和空格的位置")
print("- RNN 通过隐藏状态'记住'了之前看过的字符，从而预测下一个字符")
print("- 这就是'记忆'的力量：上下文信息通过 h_t 在时间步之间传递")
```

### 示例 3：PyTorch nn.RNN 使用示例

对比手写版本，看 PyTorch 如何封装 RNN，以及参数的对应关系。

```python
import torch
import torch.nn as nn

# ============================================================
# PyTorch nn.RNN 使用详解（对比手写版本）
# ============================================================

# --- 基本用法 ---
input_size = 3
hidden_size = 4
seq_len = 5
batch_size = 2

# 创建 RNN 层
rnn = nn.RNN(
    input_size=input_size,     # 每个时间步输入的特征维度
    hidden_size=hidden_size,   # 隐藏层大小
    num_layers=1,              # RNN 层数（堆叠）
    batch_first=True,          # 输入 shape: (batch, seq_len, input_size)
    nonlinearity='tanh',       # 激活函数: 'tanh' 或 'relu'
)

# 输入: (batch_size, seq_len, input_size)
x = torch.randn(batch_size, seq_len, input_size)

# 初始隐藏状态: (num_layers, batch_size, hidden_size)
h0 = torch.zeros(1, batch_size, hidden_size)

# 前向传播
output, h_n = rnn(x, h0)

print("=" * 70)
print("nn.RNN 基本用法")
print("=" * 70)
print(f"输入 x shape:       {list(x.shape)}")
print(f"初始 h0 shape:      {list(h0.shape)}")
print(f"输出 output shape:  {list(output.shape)}")
print(f"最终 h_n shape:     {list(h_n.shape)}")
print()
print("output 包含每个时间步的隐藏状态 h₁, h₂, ..., h_T")
print("h_n 只包含最后一个时间步的隐藏状态 h_T")
print(f"验证: output[:, -1, :] == h_n[0] ? "
      f"{torch.allclose(output[:, -1, :], h_n[0])}")

# --- 查看 nn.RNN 的参数 ---
print("\n" + "=" * 70)
print("nn.RNN 内部参数")
print("=" * 70)
for name, param in rnn.named_parameters():
    print(f"  {name:>20s}: shape = {list(param.shape)}")

print(f"""
参数对应关系:
  weight_ih_l0 = W_xh (输入到隐藏的权重), shape = ({hidden_size}, {input_size})
  weight_hh_l0 = W_hh (隐藏到隐藏的权重), shape = ({hidden_size}, {hidden_size})
  bias_ih_l0   = b_ih (输入到隐藏的偏置), shape = ({hidden_size},)
  bias_hh_l0   = b_hh (隐藏到隐藏的偏置), shape = ({hidden_size},)

注意：PyTorch 把偏置拆成了两个（b_ih 和 b_hh），实际等价于一个偏置。
即 h_t = tanh(W_xh @ x_t + b_ih + W_hh @ h_{t-1} + b_hh)
""")

# --- 手写 RNN vs PyTorch RNN 数值对比 ---
print("=" * 70)
print("手写 RNN vs PyTorch RNN 数值验证")
print("=" * 70)

# 用 PyTorch RNN 的参数做手动计算
W_xh = rnn.weight_ih_l0.detach().numpy()  # (hidden_size, input_size)
W_hh = rnn.weight_hh_l0.detach().numpy()  # (hidden_size, hidden_size)
b_ih = rnn.bias_ih_l0.detach().numpy()    # (hidden_size,)
b_hh = rnn.bias_hh_l0.detach().numpy()    # (hidden_size,)

x_np = x[0].detach().numpy()  # 取第一个 batch 的输入

# 手动逐步计算
import numpy as np
h_manual = np.zeros(hidden_size)
for t in range(seq_len):
    h_manual = np.tanh(W_xh @ x_np[t] + b_ih + W_hh @ h_manual + b_hh)

h_pytorch = output[0, -1].detach().numpy()

print(f"手动计算的 h_T: [{', '.join(f'{v:.6f}' for v in h_manual)}]")
print(f"PyTorch 的 h_T:  [{', '.join(f'{v:.6f}' for v in h_pytorch)}]")
print(f"最大差异: {np.max(np.abs(h_manual - h_pytorch)):.2e}")
print(f"结果一致: {np.allclose(h_manual, h_pytorch, atol=1e-6)}")


# --- 多层 RNN ---
print("\n" + "=" * 70)
print("多层（堆叠）RNN")
print("=" * 70)

rnn_stacked = nn.RNN(
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=3,          # 3 层 RNN 堆叠
    batch_first=True,
    dropout=0.1,           # 层间 dropout（仅在 num_layers > 1 时有效）
)

# 初始隐藏状态: (num_layers, batch_size, hidden_size)
h0_stacked = torch.zeros(3, batch_size, hidden_size)
output_stacked, h_n_stacked = rnn_stacked(x, h0_stacked)

print(f"3 层 RNN:")
print(f"  output shape: {list(output_stacked.shape)}  "
      f"(仍然是最顶层每个时间步的输出)")
print(f"  h_n shape:    {list(h_n_stacked.shape)}  "
      f"(每一层最后时间步的隐藏状态)")

print("\n参数数量对比:")
single_params = sum(p.numel() for p in rnn.parameters())
stacked_params = sum(p.numel() for p in rnn_stacked.parameters())
print(f"  1 层 RNN: {single_params:>6d} 个参数")
print(f"  3 层 RNN: {stacked_params:>6d} 个参数")

print("""
多层 RNN 的结构：

  第3层: h₁⁽³⁾ ──→ h₂⁽³⁾ ──→ h₃⁽³⁾ ──→ ... ──→ h_T⁽³⁾ ──→ output
           ↑          ↑          ↑                  ↑
  第2层: h₁⁽²⁾ ──→ h₂⁽²⁾ ──→ h₃⁽²⁾ ──→ ... ──→ h_T⁽²⁾
           ↑          ↑          ↑                  ↑
  第1层: h₁⁽¹⁾ ──→ h₂⁽¹⁾ ──→ h₃⁽¹⁾ ──→ ... ──→ h_T⁽¹⁾
           ↑          ↑          ↑                  ↑
          x₁         x₂         x₃       ...      x_T

  每一层都是独立的 RNN，下一层的输入是上一层的输出序列。
""")

# --- 完整的情感分类模型（多对一） ---
print("=" * 70)
print("实战：用 RNN 做情感分类（多对一模式）")
print("=" * 70)


class SentimentRNN(nn.Module):
    """
    基于 RNN 的情感分类模型。
    输入: 词索引序列 → Embedding → RNN → 取最后一步 → 全连接 → 分类
    """

    def __init__(self, vocab_size, embed_dim, hidden_size, output_size,
                 num_layers=1, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 词嵌入层：把词索引映射到稠密向量
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # RNN 层
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # 分类层
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        x: (batch_size, seq_len) 词索引
        """
        # (batch_size, seq_len) → (batch_size, seq_len, embed_dim)
        embedded = self.embedding(x)

        # RNN 前向传播
        # output: (batch_size, seq_len, hidden_size) — 每个时间步的隐藏状态
        # h_n: (num_layers, batch_size, hidden_size) — 最后一步的隐藏状态
        output, h_n = self.rnn(embedded)

        # 多对一：只取最后一个时间步的输出
        # h_n[-1] 是最后一层、最后一个时间步的隐藏状态
        last_hidden = h_n[-1]  # (batch_size, hidden_size)

        # 分类
        out = self.dropout(last_hidden)
        logits = self.fc(out)  # (batch_size, output_size)
        return logits


# 模型示例
model = SentimentRNN(
    vocab_size=10000,    # 词汇表大小
    embed_dim=128,       # 词嵌入维度
    hidden_size=256,     # RNN 隐藏层大小
    output_size=2,       # 二分类（正面/负面）
    num_layers=2,
    dropout=0.3,
)

# 假数据测试
batch_x = torch.randint(0, 10000, (4, 50))  # 4 条评论，每条 50 个词
logits = model(batch_x)
print(f"输入 shape: {list(batch_x.shape)}")
print(f"输出 logits shape: {list(logits.shape)}")

total_params = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total_params:,}")

# 打印模型结构
print(f"\n模型结构:")
print(model)
```

### 示例 4：可视化隐藏状态随时间步的变化

这个示例展示 RNN 的隐藏状态如何随着输入序列逐步演变，帮助你建立对"记忆"的直觉。

```python
import numpy as np

# ============================================================
# 可视化 RNN 隐藏状态随时间步的变化（纯文本可视化）
# 展示隐藏状态如何编码序列信息
# ============================================================

np.random.seed(42)

# --- 创建一个简单的 RNN ---
input_size = 5
hidden_size = 8

# 随机初始化参数
W_xh = np.random.randn(hidden_size, input_size) * 0.5
W_hh = np.random.randn(hidden_size, hidden_size) * 0.3
b_h = np.zeros(hidden_size)

def rnn_forward(sequence, W_xh, W_hh, b_h):
    """RNN 前向传播，返回所有时间步的隐藏状态。"""
    h = np.zeros(hidden_size)
    hidden_states = [h.copy()]
    for x_t in sequence:
        h = np.tanh(W_hh @ h + W_xh @ x_t + b_h)
        hidden_states.append(h.copy())
    return np.array(hidden_states)

# --- 实验 1：相同前缀 + 不同后缀 → 隐藏状态分叉 ---
print("=" * 70)
print("实验 1：相同前缀的两个序列，隐藏状态何时分叉？")
print("=" * 70)

prefix = [np.random.randn(input_size) for _ in range(5)]
suffix_a = [np.random.randn(input_size) for _ in range(5)]
suffix_b = [np.random.randn(input_size) for _ in range(5)]

seq_a = prefix + suffix_a  # 10 步
seq_b = prefix + suffix_b  # 10 步

h_a = rnn_forward(seq_a, W_xh, W_hh, b_h)
h_b = rnn_forward(seq_b, W_xh, W_hh, b_h)

print(f"\n序列 A 和 B 的前 5 步输入完全相同，后 5 步不同")
print(f"各时间步隐藏状态的距离 ||h_A - h_B||:")
print()

for t in range(11):
    dist = np.linalg.norm(h_a[t] - h_b[t])
    bar_len = int(dist * 30)
    bar = '#' * bar_len
    marker = " <-- 分叉开始" if t == 6 else ""
    print(f"  t={t:>2d}: {dist:.6f}  {bar}{marker}")

print("\n观察: 前 5 步距离为 0（输入相同），第 6 步开始分叉（输入不同）")

# --- 实验 2：隐藏状态各维度的演变 ---
print("\n" + "=" * 70)
print("实验 2：隐藏状态各维度随时间步的演变（热力图风格）")
print("=" * 70)

seq = [np.random.randn(input_size) for _ in range(15)]
h_states = rnn_forward(seq, W_xh, W_hh, b_h)

# 用 ASCII 字符表示值的大小
def val_to_char(v):
    """将 [-1, 1] 范围的值映射到字符。"""
    chars = " ░▒▓█"
    idx = int((v + 1) / 2 * (len(chars) - 1))
    idx = max(0, min(len(chars) - 1, idx))
    return chars[idx]

print(f"\n时间步 →  ", end="")
for t in range(16):
    print(f"{t:>3d}", end="")
print()
print("  " + "-" * 52)

for dim in range(hidden_size):
    print(f"h[{dim}]       ", end="")
    for t in range(16):
        v = h_states[t][dim]
        char = val_to_char(v)
        print(f"  {char}", end="")
    print(f"   (末值: {h_states[-1][dim]:>+.3f})")

print("""
图例: ' ' = -1.0 (强负),  '░' = -0.5,  '▒' = 0.0,  '▓' = +0.5,  '█' = +1.0

观察:
- h₀ = 0 (全部从中间'▒'开始)
- 不同维度对输入的响应模式不同
- 有些维度快速饱和到 +1 或 -1（tanh 的压缩效应）
- 这些维度组合在一起，形成了对序列的"编码"
""")

# --- 实验 3：远距离输入的影响力衰减 ---
print("=" * 70)
print("实验 3：早期输入对隐藏状态的影响如何衰减？")
print("=" * 70)

# 方法：在不同位置插入一个脉冲信号，观察最终隐藏状态的差异
base_seq = [np.zeros(input_size) for _ in range(30)]  # 30 步的零输入
pulse = np.ones(input_size) * 2.0  # 一个大脉冲信号

h_base = rnn_forward(base_seq, W_xh, W_hh, b_h)

print(f"\n基准序列: 30 步全零输入")
print(f"在不同时间步插入一个脉冲信号，观察对最终 h_30 的影响:\n")

for pulse_t in [1, 5, 10, 15, 20, 25, 28, 30]:
    test_seq = [np.zeros(input_size) for _ in range(30)]
    if pulse_t <= 30:
        test_seq[pulse_t - 1] = pulse.copy()

    h_test = rnn_forward(test_seq, W_xh, W_hh, b_h)
    diff = np.linalg.norm(h_test[-1] - h_base[-1])

    steps_ago = 30 - pulse_t
    bar = '#' * int(diff * 40)
    print(f"  脉冲在 t={pulse_t:>2d} ({steps_ago:>2d} 步之前): "
          f"影响力 = {diff:.6f}  {bar}")

print("""
关键观察:
- 越早的输入，对最终隐藏状态的影响越小
- 大约 15~20 步之前的输入，影响力就接近于零了
- 这就是 vanilla RNN "短期记忆"的直观体现
- LSTM/GRU 通过门控机制解决这个问题（下一章）
""")
```

### 示例 5：完整可运行的 RNN 训练示例（PyTorch）

一个端到端的训练流程：用 RNN 做简单的序列分类任务。

```python
import torch
import torch.nn as nn
import numpy as np

# ============================================================
# 完整示例：用 PyTorch RNN 做序列分类（合成数据）
# 任务：判断一个数值序列的均值是正还是负
# ============================================================

torch.manual_seed(42)
np.random.seed(42)

# --- 生成合成数据 ---
def generate_data(n_samples, seq_len, input_size):
    """
    生成二分类序列数据。
    规则：如果序列元素均值 > 0，标签为 1；否则标签为 0。
    RNN 需要"记住"整个序列的统计信息才能正确分类。
    """
    X = np.random.randn(n_samples, seq_len, input_size).astype(np.float32)
    # 给一半样本的序列加正偏移，另一半加负偏移
    for i in range(n_samples):
        if i < n_samples // 2:
            X[i] += 0.5   # 正偏移 → 均值 > 0 → 标签 1
        else:
            X[i] -= 0.5   # 负偏移 → 均值 < 0 → 标签 0

    y = (X.mean(axis=(1, 2)) > 0).astype(np.int64)
    return torch.from_numpy(X), torch.from_numpy(y)

seq_len = 20
input_size = 5
hidden_size = 32

X_train, y_train = generate_data(800, seq_len, input_size)
X_test, y_test = generate_data(200, seq_len, input_size)

print("=" * 70)
print("RNN 序列分类训练")
print("=" * 70)
print(f"训练集: {X_train.shape[0]} 样本")
print(f"测试集: {X_test.shape[0]} 样本")
print(f"序列长度: {seq_len}, 输入维度: {input_size}")
print(f"正样本占比: {y_train.float().mean():.2f}")

# --- 定义模型 ---
class SimpleRNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # output: (batch, seq_len, hidden_size)
        # h_n: (1, batch, hidden_size)
        output, h_n = self.rnn(x)
        # 取最后一个时间步
        last_hidden = h_n.squeeze(0)  # (batch, hidden_size)
        logits = self.fc(last_hidden)
        return logits

model = SimpleRNNClassifier(input_size, hidden_size, num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --- 训练循环 ---
batch_size = 32
n_epochs = 30

print(f"\n{'Epoch':>6s} | {'Train Loss':>10s} | {'Train Acc':>9s} | {'Test Acc':>8s}")
print("-" * 45)

for epoch in range(1, n_epochs + 1):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    # 简单的 mini-batch
    indices = torch.randperm(len(X_train))
    for start in range(0, len(X_train), batch_size):
        batch_idx = indices[start:start + batch_size]
        batch_x = X_train[batch_idx]
        batch_y = y_train[batch_idx]

        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪（RNN 训练必备）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * len(batch_idx)
        correct += (logits.argmax(dim=1) == batch_y).sum().item()
        total += len(batch_idx)

    # 评估
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test)
        test_acc = (test_logits.argmax(dim=1) == y_test).float().mean().item()

    if epoch % 5 == 0 or epoch == 1:
        train_loss = total_loss / total
        train_acc = correct / total
        print(f"{epoch:>6d} | {train_loss:>10.4f} | {train_acc:>8.1%} | {test_acc:>7.1%}")

# --- 最终评估 ---
model.eval()
with torch.no_grad():
    test_logits = model(X_test)
    test_pred = test_logits.argmax(dim=1)
    test_acc = (test_pred == y_test).float().mean().item()

print(f"\n最终测试准确率: {test_acc:.1%}")
print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

# --- 查看隐藏状态的梯度范数（验证梯度消失） ---
print("\n" + "=" * 70)
print("检查梯度传播：各时间步的梯度范数")
print("=" * 70)

model.train()
x_sample = X_test[:1]  # 取一个样本

# 记录每个时间步隐藏状态的梯度
h = torch.zeros(1, 1, hidden_size)
hidden_states = []

for t in range(seq_len):
    x_t = x_sample[:, t:t+1, :]
    _, h = model.rnn(x_t, h)
    hidden_states.append(h.squeeze())

# 对最后一步做预测并反向传播
logits = model.fc(hidden_states[-1])
loss = criterion(logits, y_test[:1])
loss.backward()

print(f"\n各时间步隐藏状态的梯度范数:")
for t, h_t in enumerate(hidden_states):
    if h_t.grad is not None:
        grad_norm = h_t.grad.norm().item()
        bar = '#' * int(grad_norm * 100)
        print(f"  t={t+1:>2d}: {grad_norm:.6f}  {bar}")
    else:
        print(f"  t={t+1:>2d}: (无梯度 — PyTorch 自动截断了非叶子节点)")

print("""
注意: 上面部分时间步可能显示"无梯度"，这是因为 PyTorch 默认只保存叶子节点的梯度。
要查看完整的梯度传播情况，需要对中间结果调用 retain_grad()。
这只是验证性的代码——在实际训练中，RNN 内部的梯度传播由框架自动处理。
""")
```

---

## 工程师视角

### RNN 的致命缺陷

#### 缺陷 1：无法并行化

这是 RNN 最严重的工程问题。看核心公式：

$$h_t = \tanh(W_{hh} \cdot h_{t-1} + W_{xh} \cdot x_t + b_h)$$

$h_t$ 依赖 $h_{t-1}$，$h_{t-1}$ 依赖 $h_{t-2}$，... 必须**严格串行计算**。

对比 CNN 和 Transformer：
| 架构 | 能否并行处理序列？ | 原因 |
|------|:---:|------|
| **全连接** | 是 | 所有输入一次性处理 |
| **CNN** | 是 | 卷积核在不同位置的计算互相独立 |
| **RNN** | **否** | $h_t$ 必须等 $h_{t-1}$ 算完 |
| **Transformer** | 是 | 自注意力一次性看到所有位置 |

**实际影响**：在 GPU 上训练时，一个长度 100 的序列，RNN 需要 100 次串行计算步骤，而 Transformer 只需 1 步（并行计算所有位置的注意力）。序列越长，差距越大。

这就是为什么 Transformer 在 2017 年论文标题里就写着 "Attention is All You Need"——**抛弃 RNN 的串行依赖，是获得训练速度突破的关键。**

#### 缺陷 2：长距离依赖问题

前面已经用数学和实验证明了：vanilla RNN 实际上只能记住大约 20~30 步以内的依赖关系。

**这意味着什么？**
- 一个英语句子平均 15~20 个词——刚好在 RNN 的记忆极限边缘
- 一篇文章可能有几百个句子——RNN 完全无法捕捉段落间的关系
- 在机器翻译中，长句子的翻译质量急剧下降（因为编码器 RNN 的最终隐藏状态丢失了早期输入的信息）

LSTM 和 GRU 通过**门控机制**缓解（注意：是缓解，不是彻底解决）了这个问题。Transformer 的自注意力机制则从根本上解决了它——任意两个位置之间的距离都是 1（直接连接）。

#### 两个缺陷催生的进化路线

```
RNN (1986)
 │
 │  问题：梯度消失 → 长距离依赖差
 │
 ├──→ LSTM (1997) ─── 门控机制，缓解梯度消失
 │
 ├──→ GRU (2014) ─── LSTM 的简化版，参数更少
 │
 │  问题仍在：串行计算 → 无法并行 → 训练慢
 │
 └──→ Transformer (2017) ─── 自注意力机制
      • 完全并行化（抛弃递归结构）
      • O(1) 路径长度（任意两个位置直接连接）
      • 代价：O(n²) 内存（注意力矩阵），但 GPU 友好
```

### PyTorch nn.RNN 参数详解

```python
torch.nn.RNN(
    input_size,      # 每个时间步输入的特征维度
    hidden_size,     # 隐藏状态的维度（越大容量越大，但计算越慢）
    num_layers=1,    # 堆叠的 RNN 层数
    nonlinearity='tanh',  # 'tanh' 或 'relu'
    bias=True,       # 是否使用偏置
    batch_first=False,    # True: 输入为 (batch, seq, feature)
                          # False: 输入为 (seq, batch, feature)
    dropout=0,       # 层间 dropout（仅 num_layers > 1 时生效）
    bidirectional=False,  # 是否双向 RNN
)
```

**关键参数选择指南**：

| 参数 | 典型值 | 建议 |
|------|--------|------|
| `hidden_size` | 64 ~ 512 | 越大越能编码复杂模式，但过拟合风险增大 |
| `num_layers` | 1 ~ 3 | 超过 3 层收益递减，且加剧梯度消失 |
| `batch_first` | `True` | **强烈推荐**设为 True，与数据加载的习惯一致 |
| `dropout` | 0.1 ~ 0.5 | 层间正则化，防过拟合 |
| `bidirectional` | 视任务 | 分类/标注任务用 True，生成任务用 False |
| `nonlinearity` | `'tanh'` | 默认 tanh，几乎不用改 |

**双向 RNN**：

普通 RNN 只能看到"过去"。双向 RNN 用两个 RNN，一个从左往右读，一个从右往左读，然后拼接两个方向的隐藏状态：

```
正向: x₁ → x₂ → x₃ → x₄ → x₅
      h₁→  h₂→  h₃→  h₄→  h₅→

反向: x₁ ← x₂ ← x₃ ← x₄ ← x₅
     ←h₁  ←h₂  ←h₃  ←h₄  ←h₅

输出: [h₁→;←h₁]  [h₂→;←h₂]  [h₃→;←h₃]  ...
      (拼接后维度变为 2 × hidden_size)
```

双向 RNN 在序列标注和文本分类中效果更好（因为每个位置都能同时看到前后文），但**不能用于生成任务**（生成时你还没看到后面的词）。

### 实际工程中还用 RNN 吗？

**短回答**：绝大多数场景已经被 Transformer 取代了。

**长回答**：RNN/LSTM 在以下场景仍然有价值：

1. **资源受限的边缘设备**：RNN 的参数量远小于 Transformer，推理时内存占用恒定（只需保存当前隐藏状态 $h_t$），适合嵌入式设备上的实时语音处理。
2. **在线/流式处理**：RNN 天然支持逐步输入、逐步输出。Transformer 需要看到完整序列（或使用 causal mask），在真正的流式场景中不如 RNN 自然。
3. **极长序列 + 内存有限**：Transformer 的注意力矩阵占用 $O(n^2)$ 内存。对于超长序列（如 DNA 序列），RNN 的 $O(1)$ 内存优势明显。
4. **理解 Transformer 的前置知识**：你必须先理解 RNN 的问题（串行瓶颈、梯度消失），才能真正理解 Transformer 为什么要用自注意力机制。

**结论**：学 RNN 不是为了用它，而是为了理解它的问题——这些问题正是 LSTM/GRU 和 Transformer 要解决的。

---

## 本章小结

| 概念 | 一句话总结 |
|------|-----------|
| **RNN 核心思想** | 参数共享 + 隐藏状态递推，使网络能处理任意长度的序列 |
| **核心公式** | $h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$ |
| **隐藏状态 $h_t$** | 对"到目前为止所有输入"的有损压缩表示 |
| **参数共享** | 所有时间步用同一组 $W_{xh}, W_{hh}$，类似 CNN 卷积核在空间上滑动 |
| **BPTT** | 展开 RNN 后做标准反向传播，梯度沿时间轴回传 |
| **梯度消失** | $W_{hh}$ 特征值 < 1 时，梯度指数衰减，远距离依赖学不到 |
| **梯度爆炸** | $W_{hh}$ 特征值 > 1 时，梯度指数增长，用梯度裁剪解决 |
| **串行瓶颈** | $h_t$ 依赖 $h_{t-1}$，无法并行，训练极慢 |
| **长距离依赖** | 实践中 vanilla RNN 只能记住 ~20 步内的信息 |
| **应用模式** | 一对多（图像描述）、多对一（分类）、多对多（翻译/标注） |

**下一章**：[LSTM 与 GRU](3_LSTM与GRU.md)——门控机制如何解决梯度消失？遗忘门、输入门、输出门各自在做什么？GRU 为什么更简洁？
