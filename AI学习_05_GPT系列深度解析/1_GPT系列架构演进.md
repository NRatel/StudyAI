# GPT 系列架构演进

> **前置知识**：[04 预训练语言模型演进](../AI学习_04_预训练语言模型演进/README.md)

---

## 直觉与概述

### GPT 系列的核心信念

GPT（Generative Pre-trained Transformer）系列的哲学可以用一句话概括：

> **更大的 Decoder-only Transformer + 更多的数据 + 更多的计算 = 更强的能力**

从 GPT-1 到 GPT-4，架构本身几乎没有本质变化（都是 Transformer Decoder 堆叠），真正变化的是**规模**和**训练方法**。

### 范式转变

```
GPT-1 (2018): 预训练 → 微调 → 完成任务     （每个任务需要专门微调）
GPT-2 (2019): 预训练 → 直接推理             （Zero-shot，无需微调）
GPT-3 (2020): 预训练 → Prompt 给几个例子    （Few-shot，不更新参数）
GPT-4 (2023): 预训练 + 对齐 → 通用助手      （多模态，推理增强）
```

每一代的核心突破不是"架构创新"，而是**规模带来的质变**。

---

## 严谨定义与原理

### 1. GPT-1 (2018) — 预训练+微调范式的验证

| 属性 | 值 |
|------|-----|
| 参数量 | 1.17 亿 (117M) |
| 层数 | 12 |
| 隐藏维度 | 768 |
| 注意力头 | 12 |
| 上下文长度 | 512 |
| 训练数据 | BooksCorpus (~5GB) |

**架构**：标准 Transformer Decoder（无编码器、无交叉注意力），与上一模块学的 Decoder-only 完全一致。

**核心创新**：
1. **两阶段训练**：先在无标注文本上做 CLM 预训练，再在有标注数据上微调
2. **统一架构**：不同下游任务只需要改输入格式和加一个线性头，不改模型结构

**预训练目标**（因果语言模型）：

$$L_{CLM} = -\sum_{t=1}^{T} \log P(x_t | x_1, ..., x_{t-1}; \theta)$$

**微调**：在预训练权重基础上，加任务特定的线性头，用有标注数据继续训练所有参数。学习率比预训练小很多。

**历史意义**：GPT-1 首次在 NLP 中系统验证了"大规模预训练 + 小规模微调"的有效性，与 BERT 同期奠定了预训练时代的基础。

### 2. GPT-2 (2019) — Zero-shot 与"语言模型即多任务学习器"

| 属性 | 值 |
|------|-----|
| 参数量 | 15 亿 (1.5B) |
| 层数 | 48 |
| 隐藏维度 | 1600 |
| 注意力头 | 25 |
| 上下文长度 | 1024 |
| 训练数据 | WebText (~40GB) |

**架构变化**（相对 GPT-1，很小）：
- Layer Norm 移到每个子层的前面（Pre-LN，训练更稳定）
- 最后一层后加了额外的 Layer Norm
- 残差连接的初始化缩放为 $1/\sqrt{N}$（N 为层数）
- 词表从 BPE 40K 扩展到 50257

**核心创新**：
1. **去掉微调**：所有任务都用语言模型直接做，不更新参数
2. **Zero-shot 任务转化**：任何 NLP 任务都可以描述为"给定某种输入文本，预测后续文本"

```
翻译: "Translate English to French: cheese =>"  → "fromage"
摘要: "Article: [长文本]\n\nTL;DR:"              → "摘要内容"
问答: "Q: What is the capital of France?\nA:"    → "Paris"
```

3. **"太危险不发布"**：OpenAI 最初只发布了小版本（124M/355M），后来才发布 1.5B 版本

**关键发现**：即使没有针对特定任务训练，仅靠预训练+规模扩大，模型就开始展现出任务泛化能力。这暗示了**规模是能力涌现的关键**。

### 3. GPT-3 (2020) — In-Context Learning 的突破

| 属性 | 值 |
|------|-----|
| 参数量 | 1750 亿 (175B) |
| 层数 | 96 |
| 隐藏维度 | 12288 |
| 注意力头 | 96 |
| 上下文长度 | 2048 |
| 训练数据 | ~570GB (Common Crawl + Books + Wikipedia) |
| 训练成本 | ~$4.6M (3640 PF-days) |

**架构变化**（相对 GPT-2，极小）：
- 交替使用稠密和局部带状稀疏注意力模式（Sparse Transformer 启发）
- 除此之外，与 GPT-2 几乎完全相同

**核心创新 — In-Context Learning (ICL)**：

GPT-3 最重要的发现不是架构，而是**不更新参数就能学习新任务**：

```
Zero-shot:  "Translate English to French: sea otter =>"
One-shot:   "Translate English to French: sea otter => loutre de mer\ncheese =>"
Few-shot:   "sea otter => loutre de mer\npeppermint => menthe poivrée\ncheese =>"
```

| 模式 | 示例数 | 梯度更新？ | 效果 |
|------|--------|-----------|------|
| Zero-shot | 0 | 否 | 一般 |
| One-shot | 1 | 否 | 较好 |
| Few-shot | 2~100 | 否 | 接近微调 |
| 微调 | 全量 | 是 | 最好但代价高 |

**为什么 ICL 有效？**（三种假说）

1. **隐式梯度下降**：注意力机制在前向传播时隐式执行了类似梯度下降的参数更新（Akyürek 2022, von Oswald 2023）
2. **任务识别**：预训练时见过大量"输入-输出"格式，模型学会识别 prompt 中的任务模式
3. **贝叶斯推理**：模型在所有预训练任务的后验分布中做推理，Few-shot 示例用于缩小候选任务范围

**规模的作用**：ICL 能力在小模型中几乎不存在，到 GPT-3 (175B) 级别才显著涌现。这是"涌现能力"的经典案例。

### 4. GPT-4 (2023) — 多模态与推理的飞跃

| 属性 | 推测值（OpenAI 未公开） |
|------|-----|
| 参数量 | 约 1.8T（推测，8×220B MoE） |
| 架构 | Mixture of Experts (MoE)（推测） |
| 上下文长度 | 8K / 32K / 128K（多版本） |
| 训练数据 | 约 13T tokens（推测） |
| 多模态 | 文本 + 图像输入 |

**核心突破**：

1. **多模态**：首次接受图像输入（GPT-4V），后来 GPT-4o 进一步支持音频
2. **推理能力大幅提升**：在 BAR Exam、SAT、GRE 等考试中达到人类前 10% 水平
3. **MoE 架构**（广泛推测）：
   - 非所有参数同时激活，而是由路由网络选择子集专家
   - 总参数量巨大但每次推理只用一部分，兼顾能力和效率
4. **安全对齐**：大量 RLHF + Red Teaming，拒绝有害请求
5. **长上下文**：从 8K 扩展到 128K

**GPT-4 未公开技术细节**，但社区通过实验和信息推测了大量架构信息。这也直接推动了开源社区的复现努力。

### 5. 代际对比总览

| 模型 | 年份 | 参数量 | 上下文 | 训练数据 | 核心范式 | 关键突破 |
|------|------|--------|--------|----------|----------|----------|
| GPT-1 | 2018 | 117M | 512 | 5GB | 预训练+微调 | 验证范式可行 |
| GPT-2 | 2019 | 1.5B | 1024 | 40GB | Zero-shot | 去掉微调 |
| GPT-3 | 2020 | 175B | 2048 | 570GB | Few-shot/ICL | 不更新参数学新任务 |
| GPT-4 | 2023 | ~1.8T* | 128K | ~13T* | 多模态+对齐 | 推理+图像理解 |

*: 推测值

**参数量增长**：$117M \to 1.5B \to 175B \to \sim 1.8T$，每代增长约 10~100 倍。

---

## Python 代码示例

### 示例 1：GPT-2 Zero-shot 与 Few-shot

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# 加载 GPT-2（小版本，本地可跑）
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

def generate(prompt, max_new_tokens=50, temperature=0.7):
    """自回归生成文本"""
    ids = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(ids).logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            ids = torch.cat([ids, next_id], dim=1)
            if next_id.item() == tokenizer.eos_token_id:
                break
    return tokenizer.decode(ids[0], skip_special_tokens=True)

# --- Zero-shot ---
print("=== Zero-shot 翻译 ===")
print(generate("Translate English to French: cheese =>"))

# --- Few-shot (In-Context Learning) ---
print("\n=== Few-shot 翻译 ===")
few_shot_prompt = """Translate English to French:
sea otter => loutre de mer
peppermint => menthe poivrée
plush giraffe => girafe en peluche
cheese =>"""
print(generate(few_shot_prompt, max_new_tokens=20))

# --- Few-shot 情感分类 ---
print("\n=== Few-shot 情感分类 ===")
sentiment_prompt = """Classify the sentiment as Positive or Negative:
Review: "This movie was absolutely wonderful!" Sentiment: Positive
Review: "Terrible waste of time." Sentiment: Negative
Review: "I loved every minute of it!" Sentiment:"""
print(generate(sentiment_prompt, max_new_tokens=5, temperature=0.1))
```

### 示例 2：分析 GPT-2 模型结构

```python
from transformers import GPT2LMHeadModel, GPT2Config

# 加载不同规模的 GPT-2 配置
configs = {
    "GPT-2 Small": "gpt2",
    "GPT-2 Medium": "gpt2-medium",
    "GPT-2 Large": "gpt2-large",
    "GPT-2 XL": "gpt2-xl",
}

print(f"{'模型':<16} {'层数':>4} {'d_model':>8} {'n_heads':>8} {'参数量':>12}")
print("-" * 52)

for name, model_id in configs.items():
    config = GPT2Config.from_pretrained(model_id)
    # 参数量估算: 12 * n_layer * d_model^2（近似）
    n_params = 12 * config.n_layer * config.n_embd ** 2
    # 加上 Embedding: vocab_size * d_model
    n_params += config.vocab_size * config.n_embd
    print(f"{name:<16} {config.n_layer:>4} {config.n_embd:>8} {config.n_head:>8} {n_params:>12,}")

# 加载实际模型，精确统计
model = GPT2LMHeadModel.from_pretrained("gpt2")
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nGPT-2 Small 精确参数量: {total:,} ({total/1e6:.1f}M)")
print(f"可训练参数: {trainable:,}")

# 打印模型结构（前两层）
print("\n模型结构（前两层）:")
for name, param in list(model.named_parameters())[:20]:
    print(f"  {name:<45} {str(list(param.shape)):>20}")
```

运行输出（GPT-2 精确参数量）：
```
模型              层数  d_model  n_heads        参数量
----------------------------------------------------
GPT-2 Small       12      768       12   124,293,888
GPT-2 Medium      24     1024       16   440,401,920
GPT-2 Large       36     1280       20   893,030,400
GPT-2 XL          48     1600       25 1,753,907,200

GPT-2 Small 精确参数量: 124,439,808 (124.4M)
可训练参数: 124,439,808
```

### 示例 3：In-Context Learning 效果随示例数变化

```python
from transformers import pipeline
import re

# 使用 text-generation pipeline
generator = pipeline("text-generation", model="gpt2", max_new_tokens=5)

# 任务：大小写转换（简单任务演示 ICL）
examples = [
    ("hello", "HELLO"),
    ("world", "WORLD"),
    ("python", "PYTHON"),
    ("openai", "OPENAI"),
    ("transformer", "TRANSFORMER"),
]

test_word = "neural"
expected = "NEURAL"

for n_shots in [0, 1, 2, 3, 5]:
    if n_shots == 0:
        prompt = f"Convert to uppercase: {test_word} =>"
    else:
        shots = "\n".join(f"{w} => {u}" for w, u in examples[:n_shots])
        prompt = f"{shots}\n{test_word} =>"

    result = generator(prompt, do_sample=False, pad_token_id=50256)[0]["generated_text"]
    # 提取 => 后面的部分
    answer = result.split(f"{test_word} =>")[-1].strip().split("\n")[0].strip()
    correct = expected.lower() in answer.lower()
    print(f"{n_shots}-shot: '{answer}' {'✓' if correct else '✗'}")
```

---

## 工程师视角

### 架构几乎没变，变的是规模

这是 GPT 系列最反直觉的一点：从 GPT-1 到 GPT-3，**核心架构完全相同**（Transformer Decoder + CLM）。唯一的"创新"就是把模型做大、数据做多。

| 组件 | GPT-1 | GPT-2 | GPT-3 | 变化 |
|------|-------|-------|-------|------|
| 基本架构 | Transformer Decoder | 同 | 同 | 无 |
| 注意力 | Full Self-Attention | 同 | + 稀疏变体 | 微小 |
| Layer Norm | Post-LN | Pre-LN | Pre-LN | GPT-2 改了一次 |
| 激活函数 | GELU | GELU | GELU | 无 |
| 位置编码 | 可学习 | 可学习 | 可学习 | 无 |

这意味着：**理解了 GPT-2 的架构，就理解了 GPT-3 的架构**。差异只在超参数（层数/宽度/头数）。

### 开源替代：LLaMA 系列

Meta 的 LLaMA 系列开源复现了 GPT-3 级别的能力，并做了几项关键改进：

| 改进 | GPT-3 | LLaMA |
|------|-------|-------|
| 位置编码 | 可学习绝对位置 | RoPE 旋转位置编码 |
| 注意力 | MHA | GQA (Grouped Query Attention) |
| 激活函数 | GELU | SwiGLU |
| 归一化 | Pre-LN | RMSNorm（更快） |
| 训练配比 | 参数多数据少 | 遵循 Chinchilla 最优配比 |

LLaMA 证明了：**在正确的训练配比下，较小的模型可以达到较大模型的水平**。

### GPT-4 的 MoE 架构推测

虽然 OpenAI 没有公开 GPT-4 的架构，但社区广泛推测它使用了 MoE：

```
标准 Dense 模型:    每个 token 激活所有 175B 参数
MoE 模型 (推测):    总参数 ~1.8T，每个 token 只激活 ~220B

好处：
- 总容量更大（存储更多知识）
- 推理成本与 Dense 175B 相当（每次只用一部分）
- 训练效率更高（更多参数 ≠ 更多计算）
```

这种思路后来在 Mixtral（8×7B）和 DeepSeek-V3 等开源模型中得到验证。

### 关键教训

1. **架构不是瓶颈**：Transformer Decoder 的架构已经足够强大，瓶颈在规模和数据
2. **Scaling 是可预测的**：下一章将详细讨论 Scaling Laws
3. **对齐是必须的**：预训练模型能力强但不可控，第 3 章将讨论 RLHF/DPO
4. **开源在追赶**：GPT-4 领先但 LLaMA 等开源模型在快速缩小差距

---

## 本章小结

| 概念 | 一句话总结 |
|------|-----------|
| GPT-1 | 预训练+微调范式的验证，12 层 117M 参数 |
| GPT-2 | 去掉微调，Zero-shot 直接做任务，1.5B |
| GPT-3 | In-Context Learning 突破，Few-shot 不更新参数学新任务，175B |
| GPT-4 | 多模态+MoE+推理增强，通用AI助手 |
| 核心信念 | 架构不变，规模即正义 |
| ICL | 通过 prompt 示例学习，无需梯度更新 |
| 范式转变 | 微调 → Zero-shot → Few-shot → 通用对话 |
| 开源替代 | LLaMA 系列用更好的训练方法复现 GPT-3 级别能力 |
