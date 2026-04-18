# Scaling Laws 与涌现能力

> **前置知识**：[GPT 系列架构演进](1_GPT系列架构演进.md)（GPT-1/2/3/4 的架构与训练细节）、[微调与迁移学习](../AI学习_04_预训练语言模型演进/4_微调与迁移学习.md)

---

## 直觉与概述

### "大力出奇迹"的科学依据

GPT-2（1.5B）能零样本续写新闻，GPT-3（175B）能 Few-shot 翻译、算术、编程——这些能力似乎是"堆参数堆出来的"。但 OpenAI 在 2020 年发现，这背后有精确的数学规律：**模型性能可以用参数量、数据量、计算量来预测，且遵循简洁的幂律关系。**

```
  log(Loss)
      ▲
      │ ╲
      │   ╲  斜率 = -α（幂律指数）
      │     ╲
      │       ╲        ← 双对数坐标下是直线
      └────────╲──→ log(参数量 N)
```

这意味着：在训练之前，仅凭资源预算就能预测最终模型表现。训练大模型从"炼丹"变成了**工程优化问题**。

### 四个核心概念

| 概念 | 年份 | 核心结论 |
|------|------|---------|
| **Kaplan Scaling Laws** | 2020 | Loss 与 N/D/C 存在幂律关系；偏向增大模型 |
| **Chinchilla Scaling Laws** | 2022 | 参数量和数据量应等比例增长（D/N $\approx$ 20） |
| **In-Context Learning** | 2020 | 推理时通过 prompt 示例学习新任务，不更新参数 |
| **涌现能力** | 2022 | 某些能力在小模型中不存在，大模型中突然出现 |

---

## 严谨定义与原理

### Kaplan Scaling Laws (2020)

Kaplan et al. 在 2020 年 1 月发表 *"Scaling Laws for Neural Language Models"*，系统研究了 Transformer 语言模型的性能如何随三个变量变化：

- **N**：模型参数量（不含 Embedding 层参数）
- **D**：训练数据量（token 数）
- **C**：训练计算量（FLOPs，浮点运算次数）

核心发现是交叉熵损失 L 与 N、D、C 各自服从独立的幂律关系：

$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad \alpha_N \approx 0.076$$

$$L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad \alpha_D \approx 0.095$$

$$L(C) = \left(\frac{C_c}{C}\right)^{\alpha_C}, \quad \alpha_C \approx 0.050$$

其中 $N_c$、$D_c$、$C_c$ 是与不可约损失（irreducible loss $L_\infty \approx 1.69$ nats）相关的常数。不可约损失是数据本身的固有熵——即使有无限模型和无限数据也无法降低的部分。

```
双对数坐标系下（log-log plot）：

  log L                               log L
  ▲                                   ▲
  │ ╲                                 │ ╲
  │   ╲  斜率 = -α_N                  │   ╲  斜率 = -α_D
  │     ╲                             │     ╲
  │       ╲                           │       ╲
  └────────╲──→ log N                 └────────╲──→ log D

  注意：α_D > α_N，同等倍数下增加数据的收益 > 增加参数
```

#### 关键推论

**1. 模型越大，数据效率越高**——大模型每个 token 学到的信息更多。训练到相同 Loss，大模型需要的 token 数反而更少（当然总计算量更大）。

**2. 计算效率最优分配偏向大模型**——Kaplan 的最优配比：$N_{\text{opt}} \propto C^{0.73}$，$D_{\text{opt}} \propto C^{0.27}$。计算预算翻 10 倍时，模型应放大 5.4 倍，数据仅增 1.8 倍。

**3. 与架构细节无关**——在合理范围内，同样参数总量 N 的不同层数/宽度组合，Loss 几乎相同。参数总量是决定性因素，而非如何分配。

```
同样 1B 参数的不同配置:
  - 24 层 × 宽度 2048  → Loss ≈ X
  - 48 层 × 宽度 1448  → Loss ≈ X    几乎一样
  - 12 层 × 宽度 2896  → Loss ≈ X
结论: 参数总量决定性能，架构细节影响有限
```

---

### Chinchilla Scaling Laws (2022, Hoffmann et al.)

#### Kaplan 的问题

Kaplan 的"优先增大模型"直接影响了 GPT-3（175B, 300B tokens）和 Gopher（280B, 300B tokens）。但 DeepMind 发现 **Kaplan 严重低估了数据的重要性**——因为实验中未充分调整大模型的学习率调度，高估了大模型的训练效率。

#### 核心发现

Hoffmann 重新拟合得出联合 Scaling Law：

$$L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}, \quad \alpha \approx 0.34,\ \beta \approx 0.28$$

**最优配比**：参数量和数据量应**等比例增长**：$N_{\text{opt}} \propto C^{0.50}$，$D_{\text{opt}} \propto C^{0.50}$。

经验法则：**最优训练 token 数 $\approx$ 20 $\times$ 参数量**。

#### 实验验证

| 模型 | 参数量 | 训练 tokens | D/N 比 | 训练 FLOPs |
|------|--------|------------|--------|-----------|
| Gopher | 280B | 300B | 1.1 | ~5.0 × 10²³ |
| Chinchilla | 70B | 1.4T | 20 | ~5.0 × 10²³ |

**相同计算预算，Chinchilla 全面超越 Gopher**——小 4 倍的模型，训练数据多 4.7 倍就赢了。

#### 对后续模型的影响

| 模型 | 参数量 | 训练 tokens | D/N 比 | 诊断 |
|------|--------|------------|--------|------|
| GPT-3 (2020) | 175B | 300B | 1.7 | 严重欠训练 |
| Chinchilla (2022) | 70B | 1.4T | 20 | 最优配比 |
| LLaMA (2023) | 65B | 1.4T | 21.5 | 严格遵循 Chinchilla |
| LLaMA 2 (2023) | 70B | 2T | 28.6 | 有意过度训练 |
| Mistral 7B (2023) | 7B | 估计 > 700B | > 100 | 大幅过度训练 |

**2023 年后的趋势**：工程界开始**有意过度训练**小模型（D/N >> 20），因为推理成本远高于训练成本。多花训练时间，换取每次推理的成本节省。

---

### In-Context Learning (ICL)

**定义**：模型在推理时通过 prompt 中的少量示例"学习"新任务，**不更新任何参数**。

```
┌──────────────────────────────────────┐
│  示例 1: "hello → 你好"    ← Few-shot │
│  示例 2: "thank you → 谢谢"          │
│  查询:   "goodbye →"       ← 待推理  │
└──────────────────┬───────────────────┘
            模型前向传播（不更新参数）→ 输出: "再见"
```

| 模式 | 示例数 | 说明 |
|------|-------|------|
| **Zero-shot** | 0 | 仅给任务描述 |
| **One-shot** | 1 | 给 1 个示例 |
| **Few-shot** | 2~32 | 给多个示例 |

#### 为什么 ICL 有效？三种假说

**假说 1：隐式梯度下降**（Dai et al., 2023）——Transformer 注意力层在数学上等价于对 prompt 示例做一步梯度下降。模型没有显式更新参数，但注意力的计算过程等效于参数更新。

```
传统微调:  θ' = θ - η∇L(θ)            ← 显式更新参数
ICL:       attention(Q, K, V)          ← 等价于隐式梯度下降
           其中 K, V 来自 prompt 中的示例
```

**假说 2：任务识别**——模型预训练时已见过大量任务模式（翻译、分类、问答）。ICL 示例帮助模型**识别**当前属于哪类任务，调用已有的内部"子程序"。

**假说 3：贝叶斯推理**（Xie et al., 2022）——ICL 可以理解为隐式贝叶斯推断：根据示例推断"生成这些示例的潜在概念"，据此预测：

$$P(y \mid x, \text{examples}) = \int P(y \mid x, \theta) P(\theta \mid \text{examples}) d\theta$$

```
三种假说的直觉:
  隐式梯度下降:  ICL = "模型在偷偷学习"
  任务识别:      ICL = "模型在回忆相关经验"
  贝叶斯推理:    ICL = "模型在推断规则"
→ 三种视角并不互斥，可能同时成立。
```

#### 工程实践要点

| 因素 | 建议 |
|------|------|
| 示例数量 | 4~8 个通常性价比最高 |
| 示例质量 | 高质量远比数量重要 |
| 示例顺序 | 对结果影响显著，可多排列取平均 |
| 标签空间 | "positive/negative" 比 "1/0" 效果好 |

---

### 涌现能力 (Emergent Abilities)

**定义**（Wei et al., 2022）：在小模型中**几乎不存在**（接近随机水平），但当模型规模超过某个阈值后**突然出现**的能力。

```
准确率
  ▲
  │                          ·────── 大模型：突然会了
  │                         ╱
  │                        ╱  ← 尖锐的相变
  │ · · · · · · · · · · ·╱
  │ ← 小模型：完全不会 (随机水平)
  └──────────────────────────────→ 模型参数量（对数尺度）
```

这与 Scaling Laws 描述的"平滑改进"形成鲜明对比：

| 行为 | Scaling Laws 预测 | 涌现能力观测 |
|------|-------------------|-------------|
| Loss 变化 | 平滑幂律下降 | 平滑幂律下降 |
| 任务准确率变化 | 应该也平滑？ | 实际上是阶跃突变！ |

Wei et al. 在 BIG-Bench 上识别出的经典涌现案例：

| 能力 | 涌现阈值（约） | 描述 |
|------|---------------|------|
| 多步算术 | ~10B 参数 | 三位数加法、两位数乘法 |
| 词义消歧 | ~10B 参数 | 根据上下文选择正确词义 |
| 代码生成 | ~60B 参数 | 根据描述写出正确代码 |
| 逻辑推理 / CoT | ~100B 参数 | "Let's think step by step" 有效 |

#### 争议：Schaeffer et al. (2023)

**核心质疑**：涌现可能是**评估指标的人为产物**。以三位数加法为例——

```
Exact Match: 回答 "579" → 得 1 分; "578" → 得 0 分（差一位也是 0）

每位准确率可能平滑提升: 60% → 80% → 99%
三位全对 = 三个概率相乘:  12% → 51% → 94%  ← 看起来像"突然"涌现

换用连续指标（Brier Score / token 准确率） → "涌现"变成平滑提升
```

**当前共识**：

| 观点 | 支持者 | 要点 |
|------|--------|------|
| 涌现是真实的 | Wei et al. | 某些能力确实只在大模型中出现 |
| 涌现是指标产物 | Schaeffer et al. | 换连续指标后涌现消失 |
| 综合观点 | 多数研究者 | Loss 平滑改进，但某些任务存在有效相变 |

关键洞察：**语言模型的 Loss 始终平滑下降**（Scaling Laws），但需要组合多种子能力的任务（如多步推理），可能需要每个子能力都超过某个阈值才能成功，导致整体任务表现呈现阶跃——**两者不矛盾**。

---

## Python 代码示例

### 示例 1：Scaling Laws 幂律曲线可视化

```python
import numpy as np
import matplotlib.pyplot as plt

# Kaplan 参数 (2020)
L_inf = 1.69  # 不可约损失
configs = {
    "N (Parameters)": (np.logspace(6, 11, 200), 8.8e13, 0.076),
    "D (Tokens)":     (np.logspace(8, 13, 200), 5.4e13, 0.095),
    "C (FLOPs)":      (np.logspace(16, 24, 200), 3.1e8,  0.050),
}

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (label, (x, xc, alpha)) in zip(axes, configs.items()):
    L = (xc / x) ** alpha  # 可约损失部分
    ax.loglog(x, L, linewidth=2, label=f"$\\alpha$ = {alpha}")
    ax.set_xlabel(label);  ax.set_ylabel("$L - L_\\infty$")
    ax.set_title(f"Kaplan: {label} Scaling");  ax.legend();  ax.grid(True, alpha=0.3)
plt.suptitle("Kaplan Scaling Laws (log-log 均为直线)", fontsize=14, y=1.02)
plt.tight_layout();  plt.savefig("kaplan_scaling_laws.png", dpi=150, bbox_inches="tight")
plt.show()

# Kaplan vs Chinchilla 最优分配对比
C = np.logspace(18, 25, 100)
fig, ax = plt.subplots(figsize=(8, 5))
ax.loglog(C, 1.3e-2 * C**0.73, "--b", label="Kaplan N (∝ C^0.73)")
ax.loglog(C, 0.6 * C**0.50,    "-b",  label="Chinchilla N (∝ C^0.50)")
ax.loglog(C, 1.5e1 * C**0.27,  "--r", label="Kaplan D (∝ C^0.27)")
ax.loglog(C, 0.6 * C**0.50,    "-r",  label="Chinchilla D (∝ C^0.50)")
ax.set_xlabel("Compute C (FLOPs)"); ax.set_ylabel("Optimal N or D")
ax.set_title("Kaplan 偏向大模型, Chinchilla 等比例增长"); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig("kaplan_vs_chinchilla.png", dpi=150, bbox_inches="tight")
plt.show()
```

### 示例 2：Chinchilla 最优配比计算器

```python
import math

def chinchilla_optimal(C_flops: float, tokens_per_param: float = 20.0) -> dict:
    """给定计算预算 C (FLOPs)，算出 Chinchilla 最优 N 和 D。
    核心: C ≈ 6*N*D, D = tokens_per_param * N → N = sqrt(C / (6 * tpp))"""
    N = math.sqrt(C_flops / (6.0 * tokens_per_param))
    D = tokens_per_param * N
    return {"N": N, "D": D, "N_B": N/1e9, "D_T": D/1e12}

def fmt(n):
    return f"{n/1e12:.1f}T" if n >= 1e12 else f"{n/1e9:.1f}B" if n >= 1e9 else f"{n/1e6:.0f}M"

# 不同计算预算的场景
print(f"{'场景':>10s} | {'计算预算':>12s} | {'最优 N':>8s} | {'最优 D':>8s} | {'A100 GPU-hrs':>12s}")
print("-" * 65)
for name, C in [("小实验", 6e18), ("7B 级", 6e21), ("70B 级", 6e23), ("175B 级", 3.5e24)]:
    r = chinchilla_optimal(C)
    gpu_hrs = C / (312e12 * 3600)  # A100 BF16 = 312 TFLOPS
    print(f"{name:>10s} | {fmt(C)+' FLOPs':>12s} | {fmt(r['N']):>8s} | {fmt(r['D']):>8s} | {gpu_hrs:>12,.0f}")

# 诊断实际模型
print(f"\n{'模型':>14s} | {'N':>6s} | {'D':>6s} | {'D/N':>6s} | {'诊断':>16s}")
print("-" * 60)
for name, N, D in [("GPT-3", 175e9, 300e9), ("Chinchilla", 70e9, 1.4e12),
                    ("LLaMA-65B", 65e9, 1.4e12), ("Mistral-7B", 7e9, 8e12)]:
    r = D / N
    diag = "欠训练" if r < 15 else "接近最优" if r <= 25 else "过度训练(推理优化)"
    print(f"{name:>14s} | {fmt(N):>6s} | {fmt(D):>6s} | {r:>6.1f} | {diag:>16s}")
```

### 示例 3：涌现能力模拟——Exact Match vs 连续指标

```python
import numpy as np
import matplotlib.pyplot as plt

def simulate_emergence(sizes, n_sub=3, center=9.5, steep=2.0):
    """模拟: 每个子步骤准确率 sigmoid 提升, Exact Match = 全对概率 = 各步之积。"""
    log_s = np.log10(sizes)
    sub_accs = np.array([1/(1+np.exp(-steep*(log_s - center + (i-n_sub/2)*0.3)))
                         for i in range(n_sub)])
    return {"exact_match": np.prod(sub_accs, axis=0),
            "avg_acc": np.mean(sub_accs, axis=0), "sub_accs": sub_accs}

sizes = np.logspace(7, 12, 200)
r3 = simulate_emergence(sizes, n_sub=3, center=9.5)   # 三位数加法
r5 = simulate_emergence(sizes, n_sub=5, center=10.5)   # 复杂推理

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
# 左: Exact Match 呈现阶跃
for i, acc in enumerate(r3["sub_accs"]):
    ax1.semilogx(sizes, acc, "--", alpha=0.4, label=f"Digit {i+1}")
ax1.semilogx(sizes, r3["exact_match"], "r-", lw=2.5, label="Exact Match (全对)")
ax1.set_title("Exact Match → 涌现 (阶跃)"); ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
ax1.set_xlabel("Parameters"); ax1.set_ylabel("Accuracy")

# 右: 连续指标 → 平滑
ax2.semilogx(sizes, r3["avg_acc"], "g-", lw=2.5, label="Avg Subtask Acc (连续)")
ax2.semilogx(sizes, 1-r3["avg_acc"], "purple", lw=2.5, label="Brier Score")
ax2.set_title("连续指标 → 无涌现 (平滑)"); ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
ax2.set_xlabel("Parameters"); ax2.set_ylabel("Score")
plt.tight_layout(); plt.savefig("emergence_simulation.png", dpi=150, bbox_inches="tight")
plt.show()
print("同一任务，换评估指标，'涌现'就消失了——这是 Schaeffer (2023) 的核心论点。")
```

### 示例 4：用 GPT-2 演示 In-Context Learning

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()
print(f"GPT-2 参数量: {sum(p.numel() for p in model.parameters()):,}")

def generate(prompt, max_tokens=10):
    ids = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(ids).logits[:, -1, :]
            ids = torch.cat([ids, logits.argmax(-1, keepdim=True)], dim=1)
            if ids[0, -1].item() in (tokenizer.encode("\n")[0], tokenizer.eos_token_id):
                break
    return tokenizer.decode(ids[0], skip_special_tokens=True)[len(prompt):]

def icl_test(task, examples, queries, sep=" -> "):
    print(f"\n{'='*50}\n任务: {task}\n{'='*50}")
    for q in queries:
        zero = generate(f"{q}{sep}", 10).strip()
        few_prompt = "".join(f"{i}{sep}{o}\n" for i, o in examples) + f"{q}{sep}"
        few = generate(few_prompt, 10).strip()
        print(f"  \"{q}\"  Zero-shot: \"{zero}\"  |  Few-shot: \"{few}\"")

# 情感分类
icl_test("情感分类",
         [("I love this movie", "positive"), ("This is terrible", "negative"),
          ("What a great day", "positive"), ("Worst food ever", "negative")],
         ["The service was excellent", "I'm really disappointed"])

# 简单算术 (不同 shot 数对比)
print(f"\n{'='*50}\n不同 shot 数对比 (3+4=?)\n{'='*50}")
examples = [("2+3","5"), ("4+1","5"), ("7+2","9"), ("6+3","9"), ("8+1","9"), ("5+4","9")]
for k in [0, 1, 2, 4, 6]:
    p = "".join(f"{i}={o}\n" for i, o in examples[:k]) + "3+4="
    r = generate(p, 5).strip()
    print(f"  {k}-shot: 3+4={r}  ({'correct' if r.startswith('7') else 'wrong'})")

print("\n注意: GPT-2 (124M) 的 ICL 能力非常有限。"
      "GPT-3 (175B) 才真正展现 Few-shot 涌现——这正是 Scaling 的力量。")
```

### 示例 5：Scaling Laws 外推——用小模型预测大模型 Loss

```python
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def power_law(x, a, alpha, L_inf):
    return a * np.power(x, -alpha) + L_inf

# 模拟: 训练了 5 个小模型 (10M~1B)
np.random.seed(42)
small_sizes = np.array([10e6, 50e6, 100e6, 500e6, 1e9])
true_a, true_alpha, true_Linf = 6.0, 0.076, 1.69
observed = power_law(small_sizes, true_a, true_alpha, true_Linf) + np.random.normal(0, 0.005, 5)

# 拟合幂律
popt, pcov = curve_fit(power_law, small_sizes, observed,
                       p0=[5, 0.1, 1.5], bounds=([0,0,0], [100,1,5]))

# 外推到大模型
print(f"{'模型':>6s} | {'预测 Loss':>10s} | {'真实 Loss':>10s} | {'误差':>7s}")
print("-" * 42)
for name, size in [("7B", 7e9), ("13B", 13e9), ("70B", 70e9), ("175B", 175e9)]:
    pred = power_law(size, *popt)
    true = power_law(size, true_a, true_alpha, true_Linf)
    print(f"{name:>6s} | {pred:>10.4f} | {true:>10.4f} | {abs(pred-true)/true*100:>6.2f}%")

# 可视化
all_s = np.logspace(7, 12, 200)
plt.figure(figsize=(9, 5))
plt.loglog(all_s, power_law(all_s, *popt)-popt[2], "b-", lw=2, label="拟合曲线 (from 5 small models)")
plt.loglog(all_s, power_law(all_s, true_a, true_alpha, true_Linf)-true_Linf,
           "r--", lw=2, alpha=0.7, label="真实 Scaling Law")
plt.scatter(small_sizes, observed-popt[2], s=100, c="black", zorder=5, label="观测点")
for nm, sz in [("7B", 7e9), ("70B", 70e9), ("175B", 175e9)]:
    plt.scatter(sz, power_law(sz, *popt)-popt[2], s=80, marker="*", c="gold", edgecolors="k", zorder=6)
    plt.annotate(nm, (sz, power_law(sz, *popt)-popt[2]), xytext=(8, 5), textcoords="offset points")
plt.xlabel("Parameters (N)"); plt.ylabel("Reducible Loss")
plt.title("用 5 个小模型 (10M~1B) 外推预测大模型 Loss"); plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig("scaling_law_extrapolation.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n核心价值: 用几天的小模型实验，就能预测几个月的大模型训练结果。")
```

---

## 工程师视角

### Scaling Laws 对工程决策的影响

```
传统流程:  确定架构 → 训练 → 评估 → 不行？改架构重训（每轮数周~数月）

Scaling Laws 流程:
  1. 训练 5~10 个小模型 (几小时~几天)
  2. 拟合幂律曲线，外推预测目标规模 Loss  ← 训练前就知道结果
  3. 计算 Chinchilla 最优 N/D 配比
  4. 训练大模型 (高确定性)
```

### 预算分配决策树

```
给定计算预算 C:
├─→ 追求最强性能     → Chinchilla 最优: N = sqrt(C/120), D = 20N
├─→ 追求最低推理成本 → 过度训练小模型: 小 N, 大 D (Mistral 策略)
├─→ 验证假设 (研究)  → 训练多个小模型, 确认 Scaling 后再 scale up
└─→ 快速迭代 (产品)  → 用开源模型 + 微调, 无需从零训练
```

### 用小模型预测大模型：实操步骤

1. **设计实验矩阵**：5 个规模（10M~1B），每个规模独立调优学习率
2. **训练到 Chinchilla 最优点**（D = 20N），记录最终 Loss
3. **拟合** $L(N) = a \cdot N^{-\alpha} + L_\infty$（`scipy.optimize.curve_fit`）
4. **外推并决策**：预测 Loss 是否达标，是否值得投入资源

| 风险 | 缓解措施 |
|------|---------|
| 学习率未调优 | 每个规模独立做 LR sweep |
| 数据质量变化 | 保持数据处理流程一致 |
| 外推过远 (> 2~3 数量级) | 至少覆盖 1/10 目标规模 |

### 过拟合 Scaling Laws 的风险

Scaling Laws 是强大的工程工具，但有其局限性。它预测的是**交叉熵损失**，不是下游任务准确率。

**1. Loss 不等于任务表现**——更低的 Loss 通常意味着更强的能力，但两者关系并非线性。涌现能力的存在就说明：Loss 平滑下降时，某些任务可以突变。

```
Loss (可预测)                    任务准确率 (不完全可预测)
  ▲                                ▲
  │ ╲                              │           ·───
  │   ╲  平滑幂律                   │          ╱  可能存在阶跃
  │     ╲                          │ · · · ·╱
  └──────╲──→ 规模                 └─────────────→ 规模
```

**2. 数据质量未被捕捉**——Scaling Laws 假设"数据量"是一维标量 D。但 1T 高质量数据和 1T 低质量数据效果天差地别。

**3. Post-training 改变排名**——SFT/RLHF/DPO 可显著改变实际任务表现。Loss 相近的基座模型，经不同 Post-training 后可能表现迥异。

**4. 推理时计算不在预测范围内**——CoT、Self-Consistency、Tree-of-Thoughts 等推理时技巧大幅提升能力，但不在 Scaling Laws 覆盖范围内。

```
模型最终表现 = f(预训练Loss + 数据质量 + Post-training + 推理时计算)
                  ↑ 可预测        ↑ 不可预测  ↑ 不可预测    ↑ 不可预测
```

### 从 Scaling Laws 到实际部署

| 阶段 | 关键决策 | Scaling Laws 的作用 |
|------|---------|-------------------|
| 立项 | 值不值得训练？ | 预测目标规模的 Loss |
| 预算规划 | N 和 D 怎么分配？ | Chinchilla 最优配比 |
| 训练监控 | 训练是否正常？ | Loss 曲线是否符合预测 |
| 架构选型 | 哪些改进值得？ | 小模型验证后外推 |
| 部署优化 | 大模型 vs 过度训练小模型？ | 推理成本 vs 训练成本分析 |

### 2024~2025 新趋势

| 方向 | 要点 |
|------|------|
| **推理时 Scaling Laws** | 更多采样/更长 CoT 也遵循幂律 |
| **数据质量 Scaling** | 筛选/合成数据的质量对 Scaling 的影响 |
| **多模态 Scaling** | 文本+图像+代码的联合 Scaling 行为 |
| **过度训练 Scaling** | D/N >> 20 时的 Scaling 行为特征 |

---

## 本章小结

| 概念 | 一句话总结 |
|------|-----------|
| **Kaplan Scaling Laws** | Loss 与 N/D/C 各自服从幂律关系，偏向增大模型 |
| **Chinchilla Scaling Laws** | 纠正 Kaplan：N 和 D 应等比例增长（D/N $\approx$ 20） |
| **最优配比** | $N_{\text{opt}} \propto C^{0.5}$, $D_{\text{opt}} \propto C^{0.5}$ |
| **In-Context Learning** | 推理时通过 prompt 示例学新任务，不更新参数 |
| **ICL 机制假说** | 隐式梯度下降 / 任务识别 / 贝叶斯推理——不互斥 |
| **涌现能力** | 小模型不会、大模型突然会——但可能是评估指标的阶跃效应 |
| **工程核心价值** | 用小模型实验预测大模型表现，训练前就能做出决策 |
| **过度训练趋势** | D/N >> 20，牺牲训练成本换推理效率（Mistral 策略） |

**下一章**：[对齐技术](3_对齐技术.md)——预训练好的大模型虽然能力强大，但并不"听话"。SFT、RLHF、DPO 等对齐技术如何让模型既有能力又安全可控？
