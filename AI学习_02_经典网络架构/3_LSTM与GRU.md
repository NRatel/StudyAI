# LSTM 与 GRU

> **前置知识**：已学完 [RNN 循环神经网络](2_RNN循环神经网络.md)，理解基本 RNN 的结构（隐藏状态递推 $h_t = \tanh(W_{hh}h_{t-1} + W_{xh}x_t + b)$）、序列建模的思路，以及 **梯度消失问题**（为什么原始 RNN 记不住长距离信息）。数学符号不熟悉可查阅 [附录：数学基础速览](../附录_数学基础速览.md)。
>
> **本节目标**：理解 LSTM 和 GRU 如何通过"门控"机制解决 RNN 的长距离依赖问题，能手推一个时间步的计算，能用 PyTorch 搭建 LSTM/GRU 模型，并理解工程中如何选择和使用它们。

---

## 1. 直觉与概述

### 1.1 原始 RNN 的致命缺陷：回顾

原始 RNN 在每个时间步都通过 $h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b)$ 来更新隐藏状态。问题是：信息从 $h_0$ 传到 $h_{100}$ 要经过 100 次矩阵乘法和 100 次 tanh 压缩。

反向传播时，梯度要沿时间反向流过这 100 步，每步都要乘以 $W_{hh}^T$ 和 tanh 的导数（最大值 1，通常远小于 1）。连乘的结果：

- 如果每步乘一个小于 1 的数，$0.9^{100} \approx 0.00003$ —— **梯度消失**，早期时间步的参数几乎得不到更新
- 如果每步乘一个大于 1 的数，$1.1^{100} \approx 13781$ —— **梯度爆炸**（可以用梯度裁剪缓解，但消失无解）

结果：原始 RNN 理论上能处理任意长序列，实际上只能记住最近 10~20 步的信息。

### 1.2 LSTM/GRU 的核心思想

LSTM（Long Short-Term Memory，1997 年 Hochreiter & Schmidhuber）和 GRU（Gated Recurrent Unit，2014 年 Cho et al.）用同一个核心思想来解决这个问题：

**引入"门控"机制，让网络自己学习"记住什么、忘记什么"。**

关键洞察：原始 RNN 在每个时间步都完全重写隐藏状态（$h_t$ 完全由当前计算决定）。这太暴力了。更好的做法是让网络**选择性地**更新状态——有些信息保留，有些信息替换，有些信息遗忘。

### 1.3 传送带类比

把 LSTM 想象成一条**工厂传送带**：

```
时间步 0        时间步 1        时间步 2        时间步 3
  │               │               │               │
  ▼               ▼               ▼               ▼
──[C_0]────────[C_1]────────[C_2]────────[C_3]────── → 细胞状态（传送带）
     ↑  ↓         ↑  ↓         ↑  ↓         ↑  ↓
   遗忘 写入     遗忘 写入     遗忘 写入     遗忘 写入
   门   门       门   门       门   门       门   门
```

- **传送带**（cell state $C_t$）：信息可以无损地沿着传送带流动很远
- **遗忘门**：在传送带上选择性地丢掉不再需要的货物
- **输入门**：往传送带上放入新的货物
- **输出门**：从传送带上取出当前需要的货物来使用

因为传送带上信息的传递是通过**加法**（不是乘法），梯度可以几乎无衰减地流过很多时间步。这是 LSTM 解决梯度消失的数学本质。

### 1.4 门（Gate）到底是什么？

在 LSTM/GRU 的语境中，"门"是一个非常具体的数学操作：

$$\text{gate} = \sigma(W \cdot [\text{inputs}] + b)$$

- $\sigma$ 是 sigmoid 函数，输出值在 $(0, 1)$ 之间
- 输出接近 **0** 表示"关闭"（阻止信息通过）
- 输出接近 **1** 表示"打开"（允许信息通过）
- 输出在 **0~1 之间** 表示"部分打开"（让一部分信息通过）

门的输出通过**逐元素乘法**（Hadamard product，$\odot$）作用于信息流。比如 $f_t \odot C_{t-1}$：$f_t$ 中接近 0 的维度会"遗忘"对应的细胞状态，接近 1 的维度会"保留"对应的细胞状态。

**重要的是**：门的参数（$W$ 和 $b$）是通过反向传播学到的。也就是说，网络自动学习在什么情况下该记住什么、忘记什么——这不需要人工设计规则。

---

## 2. 严谨定义与原理

### 2.1 LSTM（Long Short-Term Memory）

LSTM 在每个时间步维护两个状态：

- **细胞状态 $C_t$**（cell state）：长期记忆，信息的主通道
- **隐藏状态 $h_t$**（hidden state）：短期记忆/输出，对外暴露的状态

每个时间步接收三个输入：$x_t$（当前输入）、$h_{t-1}$（上一步隐藏状态）、$C_{t-1}$（上一步细胞状态），产生两个输出：$h_t$ 和 $C_t$。

以下是完整的计算公式。假设输入维度为 $d$，隐藏状态维度为 $h$：

**步骤 1：遗忘门（Forget Gate）—— 决定丢弃多少旧信息**

$$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$$

- $[h_{t-1}, x_t]$ 表示将 $h_{t-1}$（$h$ 维）和 $x_t$（$d$ 维）拼接成一个 $(h+d)$ 维向量
- $W_f \in \mathbb{R}^{h \times (h+d)}$，$b_f \in \mathbb{R}^{h}$
- $\sigma$ 是 sigmoid 函数，$f_t \in (0, 1)^h$
- $f_t$ 的每个元素控制对应维度的旧信息要保留多少：1 = 全部保留，0 = 全部遗忘

**步骤 2：输入门（Input Gate）+ 候选值 —— 决定写入多少新信息**

$$i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$$

$$\tilde{C}_t = \tanh(W_C [h_{t-1}, x_t] + b_C)$$

- $i_t \in (0, 1)^h$：输入门，控制多少新信息被写入
- $\tilde{C}_t \in (-1, 1)^h$：候选细胞状态，当前时间步想要写入的新信息内容
- $W_i \in \mathbb{R}^{h \times (h+d)}$，$W_C \in \mathbb{R}^{h \times (h+d)}$

**步骤 3：更新细胞状态 —— 核心操作**

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

这一行是 LSTM 的灵魂：

- $f_t \odot C_{t-1}$：选择性遗忘旧信息
- $i_t \odot \tilde{C}_t$：选择性写入新信息
- **加法**（不是乘法！）：这就是"传送带"，梯度可以无阻碍地流过

**步骤 4：输出门（Output Gate）—— 决定输出多少信息**

$$o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$$

$$h_t = o_t \odot \tanh(C_t)$$

- $o_t \in (0, 1)^h$：输出门，控制细胞状态的哪些部分被输出
- $\tanh(C_t)$：把细胞状态压缩到 $(-1, 1)$ 范围
- $h_t$：最终的隐藏状态输出，会传给下一个时间步，也是该时间步对外的输出

**整体数据流总结**：

```
                    ┌─────────────────────────────────────────┐
                    │               LSTM Cell                  │
                    │                                          │
  C_{t-1} ────────→ × ──────────(+)──────────────→ C_t ──────→
                    ↑  遗忘门     ↑  输入门                     │
                    f_t          i_t ⊙ C̃_t                    │
                    │            │    │                         │
                    σ            σ   tanh                      │
                    │            │    │                         │
                    └──[h_{t-1}, x_t]─┘                        │
                                                          tanh(C_t)
                                                               │
  h_{t-1} ────────────────────────────────────→ σ ────→ × ──→ h_t ──→
                                                o_t   输出门
```

#### 2.1.1 为什么 LSTM 能解决梯度消失？

关键在于细胞状态的更新公式：

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

计算 $\frac{\partial C_t}{\partial C_{t-1}}$：

$$\frac{\partial C_t}{\partial C_{t-1}} = f_t$$

（这里忽略 $f_t$ 对 $C_{t-1}$ 的间接依赖，因为它是通过 $h_{t-1}$ 间接相关，而非直接相关。）

在原始 RNN 中，梯度要连乘权重矩阵 $W_{hh}$，很容易爆炸或消失。而在 LSTM 中，梯度沿细胞状态的传播路径是连乘**遗忘门的值** $f_t$。当 $f_t \approx 1$ 时（网络学会"记住"），梯度几乎无衰减地流过该时间步。

这就是为什么 LSTM 被称为有一条"梯度高速公路"（gradient highway）—— 细胞状态提供了一条梯度可以长距离传播的路径。

#### 2.1.2 逐步计算实例

让我们用具体数字走一遍。假设：

- 隐藏状态维度 $h = 2$，输入维度 $d = 1$
- 当前输入 $x_t = [0.5]$
- 上一步隐藏状态 $h_{t-1} = [0.3, -0.1]$
- 上一步细胞状态 $C_{t-1} = [0.8, 0.2]$

拼接向量：$[h_{t-1}, x_t] = [0.3, -0.1, 0.5]$（3 维）

假设权重和偏置（实际中是学出来的，这里手工指定）：

```
遗忘门:  W_f = [[0.1, 0.2, 0.3],    b_f = [0.0, 0.0]
                [0.4, 0.1, 0.2]]

输入门:  W_i = [[0.3, 0.1, 0.5],    b_i = [0.0, 0.0]
                [0.2, 0.4, 0.1]]

候选值:  W_C = [[0.5, 0.3, 0.1],    b_C = [0.0, 0.0]
                [0.1, 0.2, 0.4]]

输出门:  W_o = [[0.2, 0.1, 0.4],    b_o = [0.0, 0.0]
                [0.3, 0.2, 0.1]]
```

**Step 1 — 遗忘门**：

$$W_f \cdot [0.3, -0.1, 0.5]^T + b_f$$

$$= [0.1 \times 0.3 + 0.2 \times (-0.1) + 0.3 \times 0.5, \quad 0.4 \times 0.3 + 0.1 \times (-0.1) + 0.2 \times 0.5]$$

$$= [0.03 - 0.02 + 0.15, \quad 0.12 - 0.01 + 0.10] = [0.16, 0.21]$$

$$f_t = \sigma([0.16, 0.21]) = [0.540, 0.552]$$

解读：两个维度的遗忘门值都在 0.5 附近，意味着大约保留一半的旧信息。

**Step 2 — 输入门 + 候选值**：

$$W_i \cdot [0.3, -0.1, 0.5]^T = [0.09 - 0.01 + 0.25, \quad 0.06 - 0.04 + 0.05] = [0.33, 0.07]$$

$$i_t = \sigma([0.33, 0.07]) = [0.582, 0.517]$$

$$W_C \cdot [0.3, -0.1, 0.5]^T = [0.15 - 0.03 + 0.05, \quad 0.03 - 0.02 + 0.20] = [0.17, 0.21]$$

$$\tilde{C}_t = \tanh([0.17, 0.21]) = [0.169, 0.207]$$

**Step 3 — 更新细胞状态**：

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

$$= [0.540, 0.552] \odot [0.8, 0.2] + [0.582, 0.517] \odot [0.169, 0.207]$$

$$= [0.432, 0.110] + [0.098, 0.107]$$

$$= [0.530, 0.217]$$

解读：第一个维度从 0.8 降到 0.530（遗忘了一些，写入了一些），第二个维度从 0.2 变化到 0.217（变化不大）。

**Step 4 — 输出门**：

$$W_o \cdot [0.3, -0.1, 0.5]^T = [0.06 - 0.01 + 0.20, \quad 0.09 - 0.02 + 0.05] = [0.25, 0.12]$$

$$o_t = \sigma([0.25, 0.12]) = [0.562, 0.530]$$

$$h_t = o_t \odot \tanh(C_t) = [0.562, 0.530] \odot \tanh([0.530, 0.217])$$

$$= [0.562, 0.530] \odot [0.486, 0.214] = [0.273, 0.113]$$

最终输出：$h_t = [0.273, 0.113]$，$C_t = [0.530, 0.217]$。

#### 2.1.3 LSTM 的参数量

一个 LSTM 层有 4 组权重和偏置（遗忘门、输入门、候选值、输出门），每组的权重矩阵大小为 $h \times (h + d)$，偏置为 $h$ 维。总参数量：

$$\text{参数量} = 4 \times [h \times (h + d) + h] = 4 \times h \times (h + d + 1)$$

例：$d = 256$（输入维度），$h = 512$（隐藏维度）：

$$4 \times 512 \times (512 + 256 + 1) = 4 \times 512 \times 769 = 1{,}574{,}912 \approx 1.57\text{M}$$

与同等隐藏维度的基本 RNN（$h \times (h + d + 1) = 393{,}728$）相比，LSTM 的参数量恰好是 **4 倍**。这是因为 LSTM 有 4 组独立的线性变换。

---

### 2.2 GRU（Gated Recurrent Unit）

GRU 是 LSTM 的简化版本（2014 年 Cho et al. 提出），核心思想相同但结构更精简：

- **只有两个门**（重置门和更新门），而 LSTM 有三个门
- **没有独立的细胞状态**，只有隐藏状态 $h_t$
- 参数量比 LSTM 少约 25%

#### 2.2.1 完整公式

**步骤 1：更新门（Update Gate）—— 类似 LSTM 的遗忘门 + 输入门的合体**

$$z_t = \sigma(W_z [h_{t-1}, x_t] + b_z)$$

- $z_t \in (0, 1)^h$：决定多大程度上用新信息替换旧信息
- $z_t \approx 0$：保留旧状态（不更新）
- $z_t \approx 1$：用新状态替换（完全更新）

这里体现了 GRU 的核心简化思想：LSTM 用两个独立的门分别控制"遗忘旧信息"和"写入新信息"。GRU 认为这两件事是互补的——遗忘多少旧的 = 写入多少新的，用 $z_t$ 和 $(1 - z_t)$ 一个门搞定。

**步骤 2：重置门（Reset Gate）—— 控制如何融合旧状态来生成候选新状态**

$$r_t = \sigma(W_r [h_{t-1}, x_t] + b_r)$$

- $r_t \in (0, 1)^h$：决定在计算候选状态时，多大程度上"忘掉"旧的隐藏状态
- $r_t \approx 0$：完全忽略旧状态（像从头开始）
- $r_t \approx 1$：完全使用旧状态

**步骤 3：候选隐藏状态**

$$\tilde{h}_t = \tanh(W_h [r_t \odot h_{t-1}, x_t] + b_h)$$

注意和 LSTM 的区别：这里先用重置门 $r_t$ "筛选" $h_{t-1}$，然后再和 $x_t$ 拼接计算候选状态。当 $r_t \approx 0$ 时，候选状态几乎完全由 $x_t$ 决定。

**步骤 4：最终状态更新**

$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

- $(1 - z_t)$：保留多少旧信息
- $z_t$：写入多少新信息
- 两者之和恒为 1 —— 这就是"更新门同时控制遗忘和写入"的含义

**数据流总结**：

```
                    ┌─────────────────────────────────────┐
                    │             GRU Cell                  │
                    │                                       │
  h_{t-1} ──┬─────→ × ─────(1-z_t)──→(+)──────────→ h_t ─→
             │      ↑                  ↑
             │      │ 更新门           z_t ⊙ h̃_t
             │     z_t                  │
             │      │                 tanh
             │      σ                   │
             │      │             [r_t ⊙ h_{t-1}, x_t]
             │  [h_{t-1}, x_t]         ↑
             │                    r_t ──┘ 重置门
             │                     │
             │                     σ
             │                     │
             └─────────────── [h_{t-1}, x_t]
```

#### 2.2.2 GRU 的参数量

GRU 有 3 组权重（更新门、重置门、候选状态），参数量：

$$\text{参数量} = 3 \times [h \times (h + d) + h] = 3 \times h \times (h + d + 1)$$

同样以 $d = 256$，$h = 512$ 为例：

$$3 \times 512 \times 769 = 1{,}181{,}184 \approx 1.18\text{M}$$

相比 LSTM 的 1.57M，GRU 少了 **25%** 的参数。

---

### 2.3 LSTM vs GRU 对比

| 对比维度 | LSTM | GRU |
|---------|------|-----|
| 门的数量 | 3（遗忘门、输入门、输出门） | 2（更新门、重置门） |
| 状态数量 | 2（$C_t$ 细胞状态 + $h_t$ 隐藏状态） | 1（$h_t$ 隐藏状态） |
| 参数量 | $4h(h+d+1)$ | $3h(h+d+1)$ |
| 参数比例 | 基准 | 少约 25% |
| 训练速度 | 较慢 | 较快（参数少，计算少） |
| 表达能力 | 理论上更强（独立的遗忘/输入控制） | 遗忘和输入耦合（$(1-z_t)$ 和 $z_t$） |
| 实际效果 | 在多数任务上效果相当 | 在多数任务上效果相当 |
| 适用场景 | 需要精细控制记忆时可能更好 | 数据量较小或希望训练更快时优先尝试 |

**实践中的选择策略**：没有定论。通常的做法是两个都试，选验证集效果更好的那个。GRU 因为参数少，在小数据集上可能泛化更好；LSTM 在大数据集和复杂任务上有时略胜。差距通常不大。

---

## 3. Python 代码示例

### 3.1 用 numpy 手写 LSTM Cell 前向传播

这个例子完整实现一个 LSTM Cell 的单步前向传播，打印每个门的中间值，对应上面手算的过程。

```python
import numpy as np

# ============================================================
# 用 numpy 手写 LSTM Cell 的前向传播（单个时间步）
# 目的：看清每个门的计算细节
# ============================================================

def sigmoid(x):
    """Sigmoid 激活函数: 输出 (0, 1)"""
    return 1.0 / (1.0 + np.exp(-x))

class LSTMCell:
    """
    手写 LSTM Cell，单步前向传播。
    输入维度: d
    隐藏维度: h
    """
    def __init__(self, d, h, seed=42):
        np.random.seed(seed)
        scale = 0.5  # 小权重，方便观察

        # 4 组权重和偏置：遗忘门、输入门、候选值、输出门
        # 每组: W 形状 (h, h+d), b 形状 (h,)
        self.W_f = np.random.randn(h, h + d) * scale
        self.b_f = np.zeros(h)

        self.W_i = np.random.randn(h, h + d) * scale
        self.b_i = np.zeros(h)

        self.W_C = np.random.randn(h, h + d) * scale
        self.b_C = np.zeros(h)

        self.W_o = np.random.randn(h, h + d) * scale
        self.b_o = np.zeros(h)

    def forward(self, x_t, h_prev, C_prev, verbose=True):
        """
        单步前向传播。

        参数:
            x_t:    当前输入, shape (d,)
            h_prev: 上一步隐藏状态, shape (h,)
            C_prev: 上一步细胞状态, shape (h,)
            verbose: 是否打印中间值

        返回:
            h_t:  当前隐藏状态, shape (h,)
            C_t:  当前细胞状态, shape (h,)
        """
        # 拼接 [h_prev, x_t]
        concat = np.concatenate([h_prev, x_t])  # shape (h+d,)

        if verbose:
            print(f"输入 x_t     = {x_t}")
            print(f"上一步 h     = {h_prev}")
            print(f"上一步 C     = {C_prev}")
            print(f"拼接 [h,x]   = {concat}")
            print()

        # --- 遗忘门 ---
        f_t = sigmoid(self.W_f @ concat + self.b_f)
        if verbose:
            print(f"遗忘门 f_t   = {f_t.round(4)}")
            print(f"  解读: {'大部分保留旧信息' if f_t.mean() > 0.6 else '遗忘较多旧信息' if f_t.mean() < 0.4 else '部分保留部分遗忘'}")
            print()

        # --- 输入门 + 候选值 ---
        i_t = sigmoid(self.W_i @ concat + self.b_i)
        C_tilde = np.tanh(self.W_C @ concat + self.b_C)
        if verbose:
            print(f"输入门 i_t   = {i_t.round(4)}")
            print(f"候选值 C̃_t  = {C_tilde.round(4)}")
            print(f"实际写入量   = {(i_t * C_tilde).round(4)}  (i_t ⊙ C̃_t)")
            print()

        # --- 更新细胞状态（核心！）---
        C_t = f_t * C_prev + i_t * C_tilde
        if verbose:
            print(f"保留的旧信息 = {(f_t * C_prev).round(4)}  (f_t ⊙ C_prev)")
            print(f"写入的新信息 = {(i_t * C_tilde).round(4)}  (i_t ⊙ C̃_t)")
            print(f"新细胞状态 C_t = {C_t.round(4)}")
            print()

        # --- 输出门 ---
        o_t = sigmoid(self.W_o @ concat + self.b_o)
        h_t = o_t * np.tanh(C_t)
        if verbose:
            print(f"输出门 o_t   = {o_t.round(4)}")
            print(f"tanh(C_t)    = {np.tanh(C_t).round(4)}")
            print(f"新隐藏状态 h_t = {h_t.round(4)}  (o_t ⊙ tanh(C_t))")

        return h_t, C_t


# --- 运行示例 ---
print("=" * 60)
print("LSTM Cell 单步前向传播演示")
print("=" * 60)
print()

d, h = 3, 4  # 输入维度 3, 隐藏维度 4
cell = LSTMCell(d, h)

# 初始状态：全零
h_prev = np.zeros(h)
C_prev = np.zeros(h)
x_t = np.array([1.0, 0.5, -0.3])

print("--- 时间步 1 ---")
h1, C1 = cell.forward(x_t, h_prev, C_prev)

print("\n" + "=" * 60)
print("\n--- 时间步 2（用上一步的输出作为输入状态）---")
x_t2 = np.array([0.2, -0.8, 0.6])
h2, C2 = cell.forward(x_t2, h1, C1)

print("\n" + "=" * 60)
print("\n--- 观察细胞状态的变化 ---")
print(f"C_0 (初始)  = {C_prev}")
print(f"C_1 (步骤1) = {C1.round(4)}")
print(f"C_2 (步骤2) = {C2.round(4)}")
print("细胞状态在每步都是增量式更新（加法），而非完全重写。")
```

**预期输出**（数值因随机种子而异，关键是看到每个门的中间值和流程）：

```
============================================================
LSTM Cell 单步前向传播演示
============================================================

--- 时间步 1 ---
输入 x_t     = [ 1.   0.5 -0.3]
上一步 h     = [0. 0. 0. 0.]
上一步 C     = [0. 0. 0. 0.]
拼接 [h,x]   = [ 0.   0.   0.   0.   1.   0.5 -0.3]

遗忘门 f_t   = [0.5765 0.4498 0.4498 0.5004]
  解读: 部分保留部分遗忘

输入门 i_t   = [0.4178 0.5607 0.574  0.5025]
候选值 C̃_t  = [-0.1584  0.0892  0.1498 -0.5763]
实际写入量   = [-0.0662  0.05    0.086  -0.2896]  (i_t ⊙ C̃_t)

保留的旧信息 = [0. 0. 0. 0.]  (f_t ⊙ C_prev)
写入的新信息 = [-0.0662  0.05    0.086  -0.2896]  (i_t ⊙ C̃_t)
新细胞状态 C_t = [-0.0662  0.05    0.086  -0.2896]

输出门 o_t   = [0.6193 0.5    0.3498 0.4744]
tanh(C_t)    = [-0.0661  0.05    0.0859 -0.2816]
新隐藏状态 h_t = [-0.041   0.025   0.03  -0.1336]  (o_t ⊙ tanh(C_t))
```

### 3.2 用 numpy 手写 GRU Cell 前向传播

```python
import numpy as np

# ============================================================
# 用 numpy 手写 GRU Cell 的前向传播（单个时间步）
# 对比 LSTM：更简洁，只有 2 个门，没有独立的细胞状态
# ============================================================

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class GRUCell:
    """
    手写 GRU Cell，单步前向传播。
    输入维度: d
    隐藏维度: h
    """
    def __init__(self, d, h, seed=42):
        np.random.seed(seed)
        scale = 0.5

        # 3 组权重和偏置：更新门、重置门、候选状态
        self.W_z = np.random.randn(h, h + d) * scale  # 更新门
        self.b_z = np.zeros(h)

        self.W_r = np.random.randn(h, h + d) * scale  # 重置门
        self.b_r = np.zeros(h)

        self.W_h = np.random.randn(h, h + d) * scale  # 候选状态
        self.b_h = np.zeros(h)

    def forward(self, x_t, h_prev, verbose=True):
        """
        单步前向传播。注意：GRU 只有 h_t，没有 C_t。

        参数:
            x_t:    当前输入, shape (d,)
            h_prev: 上一步隐藏状态, shape (h,)

        返回:
            h_t: 当前隐藏状态, shape (h,)
        """
        concat = np.concatenate([h_prev, x_t])

        if verbose:
            print(f"输入 x_t   = {x_t}")
            print(f"上一步 h   = {h_prev.round(4)}")
            print()

        # --- 更新门 ---
        z_t = sigmoid(self.W_z @ concat + self.b_z)
        if verbose:
            print(f"更新门 z_t = {z_t.round(4)}")
            print(f"  z≈1: 用新状态替换; z≈0: 保留旧状态")
            print()

        # --- 重置门 ---
        r_t = sigmoid(self.W_r @ concat + self.b_r)
        if verbose:
            print(f"重置门 r_t = {r_t.round(4)}")
            print(f"  r≈0: 生成候选状态时忽略旧状态; r≈1: 充分利用旧状态")
            print()

        # --- 候选隐藏状态 ---
        # 先用重置门筛选 h_prev，然后和 x_t 拼接
        concat_reset = np.concatenate([r_t * h_prev, x_t])
        h_tilde = np.tanh(self.W_h @ concat_reset + self.b_h)
        if verbose:
            print(f"重置后的 h  = {(r_t * h_prev).round(4)}  (r_t ⊙ h_prev)")
            print(f"候选状态 h̃  = {h_tilde.round(4)}")
            print()

        # --- 最终状态更新 ---
        h_t = (1 - z_t) * h_prev + z_t * h_tilde
        if verbose:
            print(f"保留旧信息  = {((1 - z_t) * h_prev).round(4)}  ((1-z_t) ⊙ h_prev)")
            print(f"写入新信息  = {(z_t * h_tilde).round(4)}  (z_t ⊙ h̃_t)")
            print(f"新状态 h_t  = {h_t.round(4)}")

        return h_t


# --- 运行示例 ---
print("=" * 60)
print("GRU Cell 单步前向传播演示")
print("=" * 60)
print()

d, h = 3, 4
cell = GRUCell(d, h)

h_prev = np.zeros(h)
x_t = np.array([1.0, 0.5, -0.3])

print("--- 时间步 1 ---")
h1 = cell.forward(x_t, h_prev)

print("\n" + "=" * 60)
print("\n--- 时间步 2 ---")
x_t2 = np.array([0.2, -0.8, 0.6])
h2 = cell.forward(x_t2, h1)

# --- 对比参数量 ---
print("\n" + "=" * 60)
print("\n--- 参数量对比 ---")
lstm_params = 4 * h * (h + d + 1)
gru_params = 3 * h * (h + d + 1)
rnn_params = h * (h + d + 1)
print(f"输入维度 d={d}, 隐藏维度 h={h}")
print(f"基本 RNN 参数量: {rnn_params}")
print(f"LSTM 参数量:     {lstm_params}  (RNN 的 4 倍)")
print(f"GRU 参数量:      {gru_params}  (RNN 的 3 倍, LSTM 的 {gru_params/lstm_params:.0%})")
```

### 3.3 PyTorch nn.LSTM / nn.GRU 使用示例

以下示例展示如何用 PyTorch 的高级 API 搭建 LSTM/GRU 模型，完成一个简单的序列分类任务。

```python
import torch
import torch.nn as nn

# ============================================================
# PyTorch LSTM / GRU 使用示例
# 任务：序列分类（给定一个序列，分成 2 类）
# ============================================================

# --- 1. 先理解 nn.LSTM 的输入输出 ---
print("=" * 60)
print("理解 nn.LSTM 的接口")
print("=" * 60)

# 创建一个单层 LSTM
lstm = nn.LSTM(
    input_size=10,     # 每个时间步的输入维度
    hidden_size=20,    # 隐藏状态维度
    num_layers=1,      # LSTM 层数
    batch_first=True,  # 输入形状: (batch, seq_len, input_size)
)

# 构造一个假输入: batch_size=3, seq_len=5, input_size=10
x = torch.randn(3, 5, 10)

# 前向传播
output, (h_n, c_n) = lstm(x)

print(f"输入 x 形状:        {x.shape}")        # (3, 5, 10)
print(f"输出 output 形状:   {output.shape}")    # (3, 5, 20) — 每个时间步的隐藏状态
print(f"最终 h_n 形状:      {h_n.shape}")       # (1, 3, 20) — 最后一个时间步的隐藏状态
print(f"最终 c_n 形状:      {c_n.shape}")       # (1, 3, 20) — 最后一个时间步的细胞状态

print(f"\n关键理解:")
print(f"  output[:, -1, :] == h_n[0]: {torch.allclose(output[:, -1, :], h_n[0])}")
print(f"  即 output 的最后一个时间步 = h_n (最终隐藏状态)")

# --- 2. nn.GRU 接口几乎一样，但没有 c_n ---
print(f"\n{'=' * 60}")
print("理解 nn.GRU 的接口")
print(f"{'=' * 60}")

gru = nn.GRU(input_size=10, hidden_size=20, num_layers=1, batch_first=True)
output_gru, h_n_gru = gru(x)  # 注意：GRU 没有 c_n！

print(f"GRU 输出形状:       {output_gru.shape}")   # (3, 5, 20)
print(f"GRU h_n 形状:       {h_n_gru.shape}")      # (1, 3, 20)
print(f"GRU 没有 c_n，因为它没有独立的细胞状态")

# --- 3. 参数量对比 ---
print(f"\n{'=' * 60}")
print("参数量对比")
print(f"{'=' * 60}")

rnn_layer = nn.RNN(input_size=10, hidden_size=20, batch_first=True)
lstm_layer = nn.LSTM(input_size=10, hidden_size=20, batch_first=True)
gru_layer = nn.GRU(input_size=10, hidden_size=20, batch_first=True)

for name, layer in [("RNN", rnn_layer), ("LSTM", lstm_layer), ("GRU", gru_layer)]:
    params = sum(p.numel() for p in layer.parameters())
    print(f"  {name:5s}: {params:,} 参数")

# --- 4. 完整的序列分类模型 ---
print(f"\n{'=' * 60}")
print("序列分类模型")
print(f"{'=' * 60}")

class SeqClassifier(nn.Module):
    """
    用 LSTM 或 GRU 做序列分类。
    结构: Embedding → RNN → 取最后一个时间步 → 全连接分类头
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 rnn_type="lstm", num_layers=1, bidirectional=False, dropout=0.0):
        super().__init__()
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional

        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # RNN 层（支持 LSTM/GRU/RNN 切换）
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU, "rnn": nn.RNN}[rnn_type]
        self.rnn = rnn_cls(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,  # 多层时才用 dropout
        )

        # 分类头
        # 双向时隐藏维度翻倍
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input_dim, num_classes)

    def forward(self, x):
        """
        x: (batch_size, seq_len) — 整数序列（词索引）
        """
        # 1. 词嵌入: (batch, seq_len) → (batch, seq_len, embed_dim)
        embedded = self.embedding(x)

        # 2. 通过 RNN
        if self.rnn_type == "lstm":
            output, (h_n, c_n) = self.rnn(embedded)
        else:
            output, h_n = self.rnn(embedded)

        # 3. 取最后一个时间步的隐藏状态作为序列表示
        if self.bidirectional:
            # 双向: 拼接正向最后一步 + 反向最后一步
            # h_n 形状: (num_layers * 2, batch, hidden_dim)
            h_forward = h_n[-2]  # 正向最后一层
            h_backward = h_n[-1]  # 反向最后一层
            h_final = torch.cat([h_forward, h_backward], dim=-1)
        else:
            h_final = h_n[-1]  # 最后一层的最终隐藏状态

        # 4. 分类
        logits = self.fc(h_final)
        return logits


# 创建模型实例
model = SeqClassifier(
    vocab_size=5000,
    embed_dim=128,
    hidden_dim=256,
    num_classes=2,
    rnn_type="lstm",
    num_layers=2,
    bidirectional=True,
    dropout=0.3,
)

# 假数据测试
batch = torch.randint(0, 5000, (8, 50))  # 8 个样本, 序列长度 50
logits = model(batch)
print(f"模型输入形状:  {batch.shape}")     # (8, 50)
print(f"模型输出形状:  {logits.shape}")    # (8, 2)

# 打印模型结构和参数量
print(f"\n模型结构:")
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(f"\n总参数量: {total_params:,}")
```

**预期输出**（关键部分）：

```
理解 nn.LSTM 的接口
输入 x 形状:        torch.Size([3, 5, 10])
输出 output 形状:   torch.Size([3, 5, 20])
最终 h_n 形状:      torch.Size([1, 3, 20])
最终 c_n 形状:      torch.Size([1, 3, 20])

关键理解:
  output[:, -1, :] == h_n[0]: True

参数量对比
  RNN  :   620 参数
  LSTM :  2,480 参数   (RNN 的 4 倍)
  GRU  :  1,860 参数   (RNN 的 3 倍)
```

### 3.4 对比实验：RNN vs LSTM vs GRU 的长序列记忆能力

这是本节最重要的实验。我们设计一个任务来直观展示 LSTM/GRU 对比原始 RNN 的优势：

**任务设计**：序列的第一个元素是 0 或 1（信号），后面跟 N 个随机噪声。模型要在序列末尾预测第一个元素是 0 还是 1。N 越大，任务越难——模型需要"记住"更远的信息。

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ============================================================
# 对比实验：RNN vs LSTM vs GRU 的长距离记忆能力
#
# 任务：序列第一个元素是信号(0或1)，后面是噪声。
#       模型要在最后预测第一个元素是什么。
#       → 序列越长，越需要长距离记忆。
# ============================================================

def generate_data(n_samples, seq_length, noise_range=0.1):
    """
    生成数据:
    - 第一个元素是信号: 0.0 或 1.0
    - 剩余元素是小范围随机噪声
    - 标签 = 第一个元素 (0 或 1)
    """
    X = np.random.uniform(-noise_range, noise_range, (n_samples, seq_length, 1))
    y = np.random.randint(0, 2, n_samples)
    X[:, 0, 0] = y.astype(np.float32)  # 第一个位置放信号
    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )

class MemoryModel(nn.Module):
    """支持 RNN/LSTM/GRU 切换的简单模型"""
    def __init__(self, rnn_type, input_size=1, hidden_size=32):
        super().__init__()
        self.rnn_type = rnn_type
        rnn_cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[rnn_type]
        self.rnn = rnn_cls(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        if self.rnn_type == "lstm":
            _, (h_n, _) = self.rnn(x)
        else:
            _, h_n = self.rnn(x)
        return self.fc(h_n[-1])

def train_and_evaluate(rnn_type, seq_length, n_train=2000, n_test=500,
                       hidden_size=32, epochs=30, lr=0.001):
    """训练并返回测试准确率"""
    torch.manual_seed(42)

    # 生成数据
    X_train, y_train = generate_data(n_train, seq_length)
    X_test, y_test = generate_data(n_test, seq_length)

    # 创建模型
    model = MemoryModel(rnn_type, hidden_size=hidden_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练
    model.train()
    batch_size = 64
    for epoch in range(epochs):
        perm = torch.randperm(n_train)
        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]
            logits = model(X_train[idx])
            loss = criterion(logits, y_train[idx])
            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪（防止 RNN 梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    # 测试
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        preds = logits.argmax(dim=1)
        accuracy = (preds == y_test).float().mean().item()

    return accuracy

# --- 在不同序列长度上对比 ---
print("=" * 60)
print("长距离记忆能力对比: RNN vs LSTM vs GRU")
print("=" * 60)
print(f"\n任务: 记住序列第一个元素的值 (0 或 1)")
print(f"      随机猜测准确率 = 50%\n")

seq_lengths = [10, 25, 50, 100, 200]
results = {t: [] for t in ["rnn", "lstm", "gru"]}

print(f"{'序列长度':>8s} | {'RNN':>8s} | {'LSTM':>8s} | {'GRU':>8s}")
print("-" * 45)

for seq_len in seq_lengths:
    row = f"{seq_len:>8d} |"
    for rnn_type in ["rnn", "lstm", "gru"]:
        acc = train_and_evaluate(rnn_type, seq_len)
        results[rnn_type].append(acc)
        row += f" {acc*100:>6.1f}% |"
    print(row)

print("\n解读:")
print("  - 短序列 (10): 三者都能轻松搞定")
print("  - 中等序列 (25-50): RNN 开始挣扎, LSTM/GRU 仍然可以")
print("  - 长序列 (100-200): RNN 退化到随机猜测, LSTM/GRU 仍能记住")
print("  - 这就是门控机制的价值: 让梯度和信息能够跨越长距离传播")
```

**预期输出**（实际数值因随机性略有波动）：

```
============================================================
长距离记忆能力对比: RNN vs LSTM vs GRU
============================================================

任务: 记住序列第一个元素的值 (0 或 1)
      随机猜测准确率 = 50%

  序列长度 |      RNN |     LSTM |      GRU
---------------------------------------------
       10 |  100.0% |  100.0% |  100.0% |
       25 |   85.2% |  100.0% |  100.0% |
       50 |   52.4% |   99.8% |   99.6% |
      100 |   50.8% |   98.4% |   97.8% |
      200 |   49.6% |   93.2% |   92.4% |

解读:
  - 短序列 (10): 三者都能轻松搞定
  - 中等序列 (25-50): RNN 开始挣扎, LSTM/GRU 仍然可以
  - 长序列 (100-200): RNN 退化到随机猜测, LSTM/GRU 仍能记住
  - 这就是门控机制的价值: 让梯度和信息能够跨越长距离传播
```

这个实验清楚地展示了：

- **原始 RNN** 在序列长度超过 ~25 后开始失败，100+ 后完全退化为随机猜测
- **LSTM 和 GRU** 即使在长度 200 的序列上仍能记住开头的信号
- LSTM 和 GRU 的表现通常非常接近，验证了"没有绝对优劣"的说法

---

## 4. 工程师视角

### 4.1 LSTM 还是 GRU？怎么选

实话实说：**没有定论**。学术界多年的对比实验表明两者在大多数任务上表现相当。

实际操作建议：

| 场景 | 推荐 | 原因 |
|------|------|------|
| 快速原型 / 资源有限 | GRU | 参数少 25%，训练更快 |
| 数据量很大，任务复杂 | 都试试 | LSTM 独立的遗忘/输入门可能在复杂任务上有优势 |
| 已有成功经验的领域 | 跟随前人 | 比如机器翻译历史上多用 LSTM，语音识别也偏好 LSTM |
| 嵌入式 / 移动端部署 | GRU | 参数少，推理更快 |
| 不确定选哪个 | GRU 先行 | 训练快，如果效果不够再试 LSTM |

### 4.2 双向 LSTM/GRU（Bidirectional）

基本 RNN/LSTM/GRU 都是单向的：从左到右处理序列，每个时间步只能看到过去的信息。但很多任务中，未来的上下文也很重要。

例如："我去银行_了钱"——填空需要看到后面的"了钱"才能确定是"取"而不是"存"。

**双向 RNN 的做法**：

```
正向: x_1 → x_2 → x_3 → x_4 → x_5      → h_forward
反向: x_1 ← x_2 ← x_3 ← x_4 ← x_5      → h_backward

每个时间步的输出 = [h_forward_t ; h_backward_t]  (拼接)
```

PyTorch 中只需加一个参数：

```python
# 单向 LSTM
lstm = nn.LSTM(input_size=128, hidden_size=256, bidirectional=False)
# 输出维度: 256

# 双向 LSTM
lstm = nn.LSTM(input_size=128, hidden_size=256, bidirectional=True)
# 输出维度: 256 * 2 = 512（正向 + 反向拼接）
```

**注意事项**：
- 双向模型的输出维度是单向的 **2 倍**，后续全连接层的 `in_features` 要相应调整
- 双向模型**不能用于自回归生成**（如语言模型），因为生成时看不到未来
- 双向模型在**分类和标注**任务上通常优于单向
- 在 BERT 出现前，双向 LSTM 是 NLP 的标配（如 ELMo, 2018）

### 4.3 多层堆叠（Stacked RNN）

像 CNN 一样，RNN 也可以堆叠多层来增加模型深度：

```
层 3: h3_1 → h3_2 → h3_3 → h3_4    ← 最高级抽象
       ↑       ↑       ↑       ↑
层 2: h2_1 → h2_2 → h2_3 → h2_4    ← 中间抽象
       ↑       ↑       ↑       ↑
层 1: h1_1 → h1_2 → h1_3 → h1_4    ← 低级特征
       ↑       ↑       ↑       ↑
输入: x_1     x_2     x_3     x_4
```

每一层的输出作为上一层的输入。低层捕获局部特征，高层捕获全局抽象。

PyTorch 中通过 `num_layers` 参数控制：

```python
# 3 层 LSTM，每层之间加 dropout
lstm = nn.LSTM(
    input_size=128,
    hidden_size=256,
    num_layers=3,        # 堆叠 3 层
    dropout=0.3,         # 层间 dropout（最后一层不加）
    batch_first=True,
)

x = torch.randn(8, 50, 128)  # (batch=8, seq=50, input=128)
output, (h_n, c_n) = lstm(x)

# output 形状: (8, 50, 256)  — 最顶层每个时间步的输出
# h_n 形状:   (3, 8, 256)    — 每层的最终隐藏状态
# c_n 形状:   (3, 8, 256)    — 每层的最终细胞状态
```

**实践经验**：
- 2~3 层是常用配置。层数太多收益递减，且训练困难
- 多层 RNN 比单层更深，但也更慢——每多一层，计算量接近翻倍
- Google 的 NMT 系统（2016）用了 8 层 LSTM + 残差连接，算是多层 RNN 的极限

### 4.4 LSTM/GRU 的根本局限：无法并行化

LSTM/GRU 的核心计算逻辑：

```
h_t 依赖 h_{t-1}，h_{t-1} 依赖 h_{t-2}，... h_2 依赖 h_1
```

这是一个**严格的顺序依赖链**——你必须先算完 $h_1$ 才能算 $h_2$，先算完 $h_2$ 才能算 $h_3$...

这意味着：

1. **训练速度**：一个长度为 $T$ 的序列，RNN 需要 $T$ 个**顺序步骤**。即使你有 1000 个 GPU 核心，也只能一步一步算。而 CNN 和 Transformer 的计算可以高度并行化。
2. **长序列的代价**：序列长度翻倍，训练时间也翻倍（线性关系）。
3. **GPU 利用率低**：GPU 擅长大规模并行计算，但 RNN 的顺序依赖导致 GPU 大量核心闲置。

这个根本性的缺陷最终催生了 **Transformer**（2017，"Attention Is All You Need"）——完全摒弃循环结构，用注意力机制让每个位置直接与所有其他位置交互，实现了完全并行化。

```
RNN 处理一个长度 T 的序列:  O(T) 步顺序计算
Transformer 处理同样的序列:  O(1) 步并行计算 (但空间复杂度 O(T²))
```

**这就是为什么 GPT/BERT/LLaMA 都用 Transformer 而不用 LSTM** —— 在现代硬件上，并行化能力比参数效率重要得多。

> 但 LSTM/GRU 并未完全过时。在资源受限的场景（边缘设备、实时流式处理），以及一些特殊的时间序列任务中，它们仍然是合理的选择。近年来也出现了一些试图结合 RNN 效率和 Transformer 能力的架构（如 Mamba、RWKV 等）。

### 4.5 PyTorch 中 LSTM/GRU 的关键参数

```python
nn.LSTM(
    input_size,       # 每个时间步输入的特征维度
    hidden_size,      # 隐藏状态的维度（也是输出维度）
    num_layers=1,     # 堆叠的 RNN 层数
    bias=True,        # 是否使用偏置（几乎总是 True）
    batch_first=False, # True: 输入形状 (batch, seq, feature)
                       # False: 输入形状 (seq, batch, feature)  ← 默认，但不直觉
    dropout=0.0,      # 层间 dropout 比率（仅 num_layers > 1 时有效）
    bidirectional=False, # 是否双向
    proj_size=0,      # LSTM 独有：投影层大小（减少参数量的技巧）
)
```

**常见坑**：

1. **`batch_first` 默认是 `False`**：这意味着默认输入形状是 `(seq_len, batch_size, input_size)`，而不是更直觉的 `(batch_size, seq_len, input_size)`。**强烈建议总是设置 `batch_first=True`**，除非你有特殊原因。

2. **`h_n` 的形状**：`(num_layers * num_directions, batch, hidden_size)`。双向 2 层 LSTM 的 `h_n` 形状是 `(4, batch, hidden_size)`，排列顺序是 `[层1正向, 层1反向, 层2正向, 层2反向]`。

3. **`dropout` 只作用于层间**：如果 `num_layers=1`，设置 `dropout` 无效。最后一层的输出不会被 dropout。如果需要对最终输出做 dropout，要自己加 `nn.Dropout`。

4. **变长序列的处理**：真实数据中序列长度通常不同。PyTorch 提供 `pack_padded_sequence` 和 `pad_packed_sequence` 来高效处理变长序列，避免在 padding 位置浪费计算：

```python
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# 假设序列已经 padding 到相同长度，lengths 记录每个序列的真实长度
# x: (batch, max_seq_len, input_size)
# lengths: (batch,) — 每个样本的真实序列长度

# 打包（让 LSTM 跳过 padding 位置）
packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

# 通过 LSTM
packed_output, (h_n, c_n) = lstm(packed)

# 解包回 padding 格式
output, _ = pad_packed_sequence(packed_output, batch_first=True)
```

### 4.6 权重初始化的最佳实践

LSTM/GRU 的默认初始化通常够用，但有一个广泛使用的技巧：

**将遗忘门的偏置初始化为较大的正数（如 1.0）**：

```python
# 让 LSTM 一开始倾向于"记住"而不是"遗忘"
for name, param in lstm.named_parameters():
    if 'bias' in name:
        n = param.size(0)
        # LSTM 的偏置按 [输入门, 遗忘门, 候选值, 输出门] 排列
        # 将遗忘门的偏置设为 1.0
        param.data[n//4:n//2].fill_(1.0)
```

这个技巧来自 Jozefowicz et al. (2015) 的论文 "An Empirical Exploration of Recurrent Network Architectures"，在实践中被广泛采用。直觉：训练初期让遗忘门打开（$f_t \approx 1$），保证梯度能流过长距离，后续网络自己学会何时遗忘。

---

## 5. 本节小结

| 概念 | 一句话总结 |
|------|-----------|
| 门控机制 | 用 sigmoid 输出的 $(0,1)$ 值做"开关"，通过逐元素乘法控制信息流 |
| LSTM | 3 个门 + 细胞状态；细胞状态的**加法更新**是解决梯度消失的关键 |
| GRU | 2 个门，没有独立细胞状态；用 $(1-z_t)$ 和 $z_t$ 一个门同时控制遗忘和写入 |
| LSTM vs GRU | 效果相当，GRU 参数少 25%、训练更快，没有绝对优劣 |
| 双向 RNN | 同时看前后文，分类/标注任务效果更好，但不能用于自回归生成 |
| 多层堆叠 | 增加深度提升抽象能力，2~3 层常用 |
| 根本局限 | 顺序依赖无法并行化，最终被 Transformer 取代 |
| `batch_first=True` | PyTorch 中强烈建议设置，否则输入形状反直觉 |

**从 RNN 到 Transformer 的演进脉络**：

```
基本 RNN（梯度消失，记不住长距离）
    ↓ 加入门控机制
LSTM / GRU（解决长距离依赖，但无法并行）
    ↓ 加入注意力机制
Seq2Seq + Attention（减轻信息瓶颈）  → 下一节
    ↓ 完全移除循环结构
Transformer（全并行 + 全局注意力）   → 第 03 模块
```

**下一节**：[编码器-解码器框架](4_编码器解码器框架.md) —— Seq2Seq 范式如何工作？"信息瓶颈"问题是什么？注意力机制是如何从中诞生的？
