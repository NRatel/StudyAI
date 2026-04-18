# BERT 与 GPT：两大路线

> **前置知识**：[自监督预训练范式](2_自监督预训练范式.md)（MLM 与 CLM 的原理）、[Transformer 架构](../AI学习_03_注意力机制与Transformer/3_Transformer架构.md)（编码器、解码器、注意力机制）

---

## 直觉与概述

### 一个核心分歧

2018 年，NLP 领域几乎同时出现了两篇划时代的论文：

- **BERT**（Google, 2018.10）：用 Transformer **编码器** + **MLM**（掩码语言模型）预训练，然后微调
- **GPT**（OpenAI, 2018.06）：用 Transformer **解码器** + **CLM**（因果语言模型）预训练，然后微调

两者都证明了"预训练 + 微调"范式在 NLP 中的巨大威力，但在架构选择上走向了完全不同的方向：

```
                  Transformer (2017)
                       │
            ┌──────────┴──────────┐
            ▼                     ▼
     Encoder-only              Decoder-only
     (双向理解)                (单向生成)
            │                     │
         BERT (2018)           GPT (2018)
            │                     │
   RoBERTa, ALBERT,          GPT-2, GPT-3,
   DistilBERT, ...           GPT-4, LLaMA, ...
            │                     │
      理解类任务为主           生成类任务为主
   (分类, NER, QA抽取)      (文本生成, 对话, 推理)
```

### 两条路线的哲学差异

**BERT 的哲学——"深度理解"**：看到完整的上下文（双向），才能真正理解每个词的含义。预训练目标是"填空"（MLM）。类比：阅读理解——读完整篇文章，再回答问题。

**GPT 的哲学——"顺序生成"**：语言是从左到右逐词产生的，模型应该学会"接着往下说"。预训练目标是"续写"（CLM）。类比：作家写作——根据已经写好的内容，决定下一个字写什么。

**直觉判断**：如果任务是"理解"（分类、抽取、匹配），BERT 更自然；如果任务是"生成"（续写、对话、翻译），GPT 更自然。但后来的发展表明，Decoder-only 路线通过 scaling 和 prompt 技术，逐渐"统一"了几乎所有任务。

---

## 严谨定义与原理

### BERT：Bidirectional Encoder Representations from Transformers

#### 架构

BERT 使用 **Transformer 编码器堆叠**，没有解码器，没有因果掩码：

```
输入 token IDs → Token Emb + Segment Emb + Position Emb
      │
      ▼
┌─────────────────────────┐
│   Encoder Layer × L     │  Self-Attention (双向, 无掩码) + FFN
│   Self-Attn → Add&Norm  │
│   FFN → Add&Norm        │
└───────────┬─────────────┘
            ▼
     每个位置的上下文表示 (batch, seq_len, d_model)
```

| 配置 | 层数 L | 隐藏维度 d_model | 注意力头数 | 参数量 |
|------|--------|-----------------|-----------|--------|
| BERT-Base | 12 | 768 | 12 | 110M |
| BERT-Large | 24 | 1024 | 16 | 340M |

**关键特征**：每个位置都能看到序列中的**所有**其他位置（双向注意力）。

#### 输入表示

BERT 的输入由三种 Embedding 相加构成：

```
输入:   [CLS] I  love  NLP  [SEP] It  is  great [SEP]
Token:  [101] [1] [23] [45] [102] [3] [7] [89]  [102]
Seg:    [ A ] [A] [A]  [A]  [A]   [B] [B] [B]   [B]
Pos:    [ 0 ] [1] [2]  [3]  [4]   [5] [6] [7]   [8]

最终输入 = Token Emb + Segment Emb + Position Emb
```

- **Token Embedding**：WordPiece 分词后的词向量
- **Segment Embedding**：标记 token 属于句子 A 还是句子 B（用于句对任务）
- **Position Embedding**：**可学习的**位置编码（非正弦余弦），最大长度 512
- `[CLS]`：序列聚合表示（用于分类）；`[SEP]`：分隔两个句子

#### 预训练目标

**1. Masked Language Model (MLM)**：随机遮住 15% 的 token，让模型预测被遮住的词。

```
原始:  The cat sat on the mat
遮掩:  The [MASK] sat on the mat  →  预测 "cat"
```

对被选中的 15% token：80% 替换为 `[MASK]`，10% 替换为随机 token，10% 保持不变。

$$\mathcal{L}_{\text{MLM}} = -\sum_{i \in \text{masked}} \log P(x_i \mid x_{\backslash i})$$

**2. Next Sentence Prediction (NSP)**：预测句子 B 是否是句子 A 的真实下一句（二分类）。

**总损失**：$\mathcal{L} = \mathcal{L}_{\text{MLM}} + \mathcal{L}_{\text{NSP}}$

> **注意**：RoBERTa（2019）发现 NSP 对性能提升有限甚至有害，因此去掉了 NSP。

#### 微调

在下游任务上加一个任务特定的"头"（线性层），微调整个模型：

| 任务 | 使用的表示 | 头结构 |
|------|-----------|--------|
| 文本分类（情感分析） | `[CLS]` 的输出向量 | Linear(d_model, num_classes) |
| 命名实体识别（NER） | 每个 token 的输出向量 | Linear(d_model, num_entity_types) |
| 问答（抽取式） | 每个 token 的输出向量 | 预测 start/end 位置 |
| 句对匹配 | `[CLS]` 的输出向量 | Linear(d_model, 2) |

#### BERT 的重要变体

| 模型 | 年份 | 核心改进 | 要点 |
|------|------|---------|------|
| **RoBERTa** | 2019 | 去掉 NSP，更大 batch，更多数据，动态 masking | BERT 的"正确训练方式" |
| **ALBERT** | 2019 | 参数共享 + Embedding 分解 | 大幅减少参数量 |
| **DistilBERT** | 2019 | 知识蒸馏，6 层 | 保留 97% 性能，速度快 60% |
| **ELECTRA** | 2020 | 用判别任务替代 MLM | 训练效率更高 |

**RoBERTa 的关键发现**：去掉 NSP、动态 Masking、更大 Batch（256→8K）、更多数据（160GB）。结论：BERT 原始论文严重欠训练，用好 BERT 架构就能超越当时所有变体。

---

### GPT：Generative Pre-trained Transformer

#### 架构

GPT 使用 **Transformer 解码器堆叠**，但**没有编码器，没有交叉注意力**——只保留掩码自注意力 + FFN：

```
输入 token IDs → Token Emb + Position Emb
      │
      ▼
┌─────────────────────────┐
│   Decoder Layer × L     │  Masked Self-Attention (因果掩码) + FFN
│   Masked Self-Attn      │
│   → Add&Norm → FFN      │
│   → Add&Norm            │
└───────────┬─────────────┘
            ▼
      Linear + Softmax → 下一个 token 的概率分布
```

**关键区别**：因果掩码确保每个位置只能看到**自己和之前**的 token。

```
因果注意力掩码（GPT）：            双向注意力（BERT）：

位置  1  2  3  4                  位置  1  2  3  4
 1   [✓  ×  ×  × ]               1   [✓  ✓  ✓  ✓]
 2   [✓  ✓  ×  × ]               2   [✓  ✓  ✓  ✓]
 3   [✓  ✓  ✓  × ]               3   [✓  ✓  ✓  ✓]
 4   [✓  ✓  ✓  ✓ ]               4   [✓  ✓  ✓  ✓]
  只能看到当前及之前                  每个位置看到所有位置
```

GPT 初代配置：12 层，d_model=768，12 头，117M 参数，BooksCorpus 训练。

#### 预训练目标

标准的 **Causal Language Modeling (CLM)**——给定前文，预测下一个 token：

$$\mathcal{L}_{\text{CLM}} = -\sum_{t=1}^{T} \log P(x_t \mid x_1, x_2, \ldots, x_{t-1})$$

| 维度 | MLM (BERT) | CLM (GPT) |
|------|-----------|-----------|
| 上下文方向 | 双向（看前后文） | 单向（只看前文） |
| 预测目标 | 被掩码的 token（约 15%） | 每个位置的下一个 token（100%） |
| 训练信号密度 | 低（每序列只预测 15%） | 高（每个位置都产生梯度） |
| 是否可直接生成 | 不能 | 可以（天然自回归） |

#### 微调与演进

GPT 初代微调时同时优化任务损失和 CLM 辅助损失：$\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda \cdot \mathcal{L}_{\text{CLM}}$

GPT 的真正意义不在于初代的性能（很多任务不如 BERT），而在于奠定了整条路线：

| 模型 | 年份 | 参数量 | 核心突破 |
|------|------|--------|---------|
| GPT | 2018 | 117M | 预训练+微调范式 |
| GPT-2 | 2019 | 1.5B | Zero-shot 能力（不微调也能做任务） |
| GPT-3 | 2020 | 175B | In-Context Learning（给几个例子就会做） |
| GPT-4 | 2023 | 未公开 | 多模态，推理能力大幅提升 |

---

### 两条路线对比

| 维度 | BERT (Encoder-only) | GPT (Decoder-only) |
|------|--------------------|--------------------|
| **架构** | Transformer 编码器堆叠 | Transformer 解码器堆叠（无交叉注意力） |
| **注意力模式** | 双向（全连接） | 单向（因果掩码） |
| **预训练目标** | MLM + NSP | CLM |
| **训练信号密度** | 低（15%） | 高（100%） |
| **是否能直接生成** | 不能 | 可以 |
| **典型任务** | 分类、NER、QA、语义匹配 | 文本生成、对话、翻译、代码 |
| **使用方式** | 预训练 → 微调 | 微调 / Prompt / In-Context Learning |
| **代表模型** | BERT, RoBERTa, ELECTRA | GPT-3/4, LLaMA, Mistral, Claude |

### 为什么 Decoder-only 最终胜出

从 2020 年至今，Decoder-only 已成为绝对主流。原因有三：

#### 1. 统一的生成范式

任何 NLP 任务都可以转化为文本生成：

```
分类:  "这部电影太棒了。\n情感：" → "正面"
NER:   "北京是中国的首都。\n找出地名：" → "北京, 中国"
翻译:  "Translate: I love you → 中文：" → "我爱你"
```

BERT 需要为每个任务设计不同的任务头；GPT 只需要不同的 prompt。**一个模型做所有事情**的优势在工程和商业上是压倒性的。

#### 2. Scaling Laws 的发现

Kaplan et al. (2020) 发现了**神经网络缩放定律**：$L(N) \propto N^{-\alpha}$。参数量、数据量、计算量三个维度上，模型性能都遵循平滑的幂律关系。CLM 训练信号密度高，在 scaling 时比 MLM 更高效。

#### 3. In-Context Learning 的涌现

GPT-3（175B）展现了 **In-Context Learning**——给几个示例，模型就能学会新任务：

```
输入: "happy → 开心
      sad → 难过
      angry →"
输出: "生气"
```

这种能力只在模型规模超过阈值后涌现，彻底改变了 AI 使用方式：不再需要标注数据和微调，用自然语言就能"编程"模型。BERT 不能生成文本，天然不支持这种能力。

---

## Python 代码示例

### 示例 1：用 BERT 做 MLM 填空

```python
from transformers import BertTokenizer, BertForMaskedLM
import torch

# 加载 BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")
model.eval()

print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
print(f"配置: {model.config.num_hidden_layers} 层, "
      f"d_model={model.config.hidden_size}, heads={model.config.num_attention_heads}")

# MLM 填空
text = "The capital of France is [MASK]."
print(f"\n输入: {text}")

inputs = tokenizer(text, return_tensors="pt")
mask_idx = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

with torch.no_grad():
    logits = model(**inputs).logits

# 取 [MASK] 位置的 Top-5 预测
probs = torch.softmax(logits[0, mask_idx], dim=-1)
top5 = torch.topk(probs, k=5, dim=-1)

print("Top-5 预测:")
for i, (prob, idx) in enumerate(zip(top5.values[0], top5.indices[0])):
    print(f"  {i+1}. {tokenizer.decode(idx.item()):15s} (概率: {prob:.4f})")

# 多个 [MASK]
text2 = "I want to [MASK] a [MASK] in the park."
print(f"\n输入: {text2}")
inputs2 = tokenizer(text2, return_tensors="pt")
mask_positions = (inputs2["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

with torch.no_grad():
    logits2 = model(**inputs2).logits

for pos in mask_positions:
    top3 = torch.topk(torch.softmax(logits2[0, pos], dim=-1), k=3)
    tokens = [tokenizer.decode(idx.item()) for idx in top3.indices]
    print(f"  位置 {pos.item()}: {tokens}")
```

### 示例 2：用 GPT-2 做文本生成

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import torch.nn.functional as F

# 加载 GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
print(f"配置: {model.config.n_layer} 层, "
      f"d_model={model.config.n_embd}, heads={model.config.n_head}")

def generate(model, tokenizer, prompt, max_new_tokens=30,
             temperature=1.0, top_k=0):
    """自回归生成：手动逐 token 生成以理解原理。
    top_k=0 表示贪心解码，top_k>0 表示 top-k 采样。"""
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    generated = input_ids.clone()

    for step in range(max_new_tokens):
        with torch.no_grad():
            next_logits = model(generated).logits[:, -1, :] / temperature

        if top_k > 0:
            # Top-k 采样
            top_k_logits, top_k_indices = torch.topk(next_logits, k=top_k)
            probs = F.softmax(top_k_logits, dim=-1)
            sampled = torch.multinomial(probs, num_samples=1)
            next_token = top_k_indices.gather(-1, sampled)
        else:
            # 贪心解码
            next_token = next_logits.argmax(dim=-1, keepdim=True)

        generated = torch.cat([generated, next_token], dim=1)

    return tokenizer.decode(generated[0], skip_special_tokens=True)

prompt = "The future of artificial intelligence is"

# 贪心解码（确定性）
print(f"\n--- 贪心解码 ---")
print(generate(model, tokenizer, prompt, top_k=0))

# 采样解码（每次结果不同）
print(f"\n--- Top-k 采样 (temperature=0.8, top_k=50) ---")
for i in range(3):
    result = generate(model, tokenizer, prompt, temperature=0.8, top_k=50)
    print(f"  生成 {i+1}: {result}")
```

### 示例 3：对比两者的注意力模式

```python
from transformers import BertModel, BertTokenizer, GPT2Model, GPT2Tokenizer
import torch
import numpy as np

# 加载两个模型，启用注意力输出
bert_tok = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)
bert_model.eval()

gpt2_tok = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2Model.from_pretrained("gpt2", output_attentions=True)
gpt2_model.eval()

sentence = "The cat sat on the mat"

# 提取注意力权重
with torch.no_grad():
    bert_attn = bert_model(**bert_tok(sentence, return_tensors="pt")).attentions
    gpt2_attn = gpt2_model(**gpt2_tok(sentence, return_tensors="pt")).attentions

bert_tokens = bert_tok.convert_ids_to_tokens(bert_tok(sentence)["input_ids"])
gpt2_tokens = gpt2_tok.convert_ids_to_tokens(gpt2_tok(sentence)["input_ids"])

# 打印第 1 层 Head 0 的注意力矩阵
def print_attention(name, tokens, attn_matrix):
    print(f"\n{'='*55}")
    print(f"{name}（第 1 层, Head 0）")
    print(f"{'='*55}")
    w = attn_matrix[0][0, 0].numpy()  # layer 0, batch 0, head 0
    for i, row in enumerate(w):
        row_str = " ".join(f"{v:.2f}" for v in row)
        print(f"  {tokens[i]:>8s} → [{row_str}]")

print_attention("BERT 注意力 —— 双向（每行都有非零值）", bert_tokens, bert_attn)
print_attention("GPT-2 注意力 —— 单向（上三角全为 0）", gpt2_tokens, gpt2_attn)

# 验证因果掩码：GPT-2 所有层的上三角应严格为 0
print(f"\n验证 GPT-2 因果掩码（12 层 × 12 头）:")
for layer_idx in range(12):
    w = gpt2_attn[layer_idx][0].numpy()  # (heads, seq, seq)
    # 用 numpy 批量提取上三角
    upper_sum = sum(
        np.triu(w[h], k=1).sum() for h in range(w.shape[0])
    )
    if layer_idx in [0, 1, 11]:
        print(f"  Layer {layer_idx+1:>2d}: 上三角元素之和 = {upper_sum:.2e}  "
              f"{'OK' if upper_sum < 1e-6 else 'FAIL'}")
    elif layer_idx == 2:
        print(f"  ... (中间层均为 0，省略) ...")

# 注意力熵对比（高熵=分散，低熵=集中）
def avg_entropy(attn_tuple, layer):
    w = attn_tuple[layer][0].numpy()
    return -(w * np.log(w + 1e-10)).sum(axis=-1).mean()

print(f"\n注意力熵对比:")
print(f"  {'模型':>6s} | {'Layer 1':>8s} | {'Layer 6':>8s} | {'Layer 12':>8s}")
for name, attn in [("BERT", bert_attn), ("GPT-2", gpt2_attn)]:
    e1, e6, e12 = avg_entropy(attn, 0), avg_entropy(attn, 5), avg_entropy(attn, 11)
    print(f"  {name:>6s} | {e1:>8.3f} | {e6:>8.3f} | {e12:>8.3f}")
print("  BERT 熵通常更高（双向关注，信息源更多）")
```

---

## 工程师视角

### BERT 仍然活跃的场景

尽管 Decoder-only 是主流，BERT 在以下场景中仍然大量使用：

**1. 搜索与信息检索**：BERT 擅长将文本编码为语义向量，用于查询-文档匹配和重排序。

**2. 命名实体识别（NER）**：token 级分类任务，BERT 的双向注意力能同时利用前后文判断实体。

**3. 文本分类**：情感分析、意图识别等，BERT-Base（110M）就能达到很好效果。

| 优势 | 说明 |
|------|------|
| 模型小 | 110M 参数，推理成本远低于 7B+ 的 LLM |
| 推理快 | 一次前向传播，不需要自回归生成 |
| 双向表示 | 理解类任务天然更有效 |
| 部署简单 | CPU 即可快速推理 |

### GPT 路线的优势

**"一个模型做所有任务"** 是最大的工程优势：

```
BERT 时代: 6+ 个任务 → 6+ 个微调模型 → 6+ 套训练流程
GPT 时代:  所有任务 → 1 个 LLM + 不同 Prompt
```

### Encoder-Decoder 路线的定位（T5/BART）

```
Encoder-only (BERT)        Decoder-only (GPT)       Encoder-Decoder (T5)
┌─────────────┐            ┌─────────────┐          ┌──────┐  ┌──────┐
│ 双向注意力   │            │ 因果注意力   │          │双向  │→│因果+交叉│
│ (全连接)     │            │ (下三角)     │          │      │  │       │
└─────────────┘            └─────────────┘          └──────┘  └──────┘
```

T5 把所有任务统一为"文本到文本"格式。编码器双向理解输入（像 BERT），解码器自回归生成输出（像 GPT），在翻译、摘要中效果优异。但因架构复杂且 Decoder-only 通过 scaling 已能处理同类任务，未成为主流。

| 路线 | 代表模型 | 适用场景 | 当前状态 |
|------|---------|---------|---------|
| Encoder-only | BERT, RoBERTa | 理解、分类、NER、搜索 | 特定场景仍活跃 |
| Decoder-only | GPT, LLaMA, Claude | 几乎所有任务 | 绝对主流 |
| Encoder-Decoder | T5, BART | 翻译、摘要 | 小众但有价值 |

### 工程选型决策树

```
你的任务是什么？
│
├─→ 需要生成文本（对话、写作、代码）
│   └─→ Decoder-only (GPT/LLaMA/API)
│
├─→ 文本分类 / 情感分析 / 语义匹配
│   ├─→ 有标注数据 + 追求低延迟低成本 → BERT/RoBERTa 微调
│   └─→ 标注数据少 / 任务多变 → LLM + Prompt
│
├─→ NER / 序列标注 → BERT/RoBERTa 微调（仍是首选）
├─→ 搜索 / 语义检索 → BERT 系列（sentence-transformers）
├─→ 翻译 / 摘要 → T5/BART 或 Decoder-only LLM
└─→ 不确定 → 先试 LLM API，不够再微调 BERT
```

### 参数量与计算成本对比

| 模型 | 参数量 | 推理延迟 (128 token) | 典型硬件 |
|------|--------|---------------------|---------|
| BERT-Base | 110M | ~5ms (GPU) / ~50ms (CPU) | 消费级 GPU 或 CPU |
| BERT-Large | 340M | ~15ms (GPU) | 单张 GPU |
| GPT-2 | 1.5B | ~100ms (GPU) | 单张 GPU |
| LLaMA-2-7B | 7B | ~500ms (GPU) | 1x A100 |
| GPT-3 | 175B | API 调用 | 多节点 GPU 集群 |

### 历史时间线

```
2017.06  Transformer ("Attention Is All You Need")
    │
2018.06  GPT (OpenAI) ← 首次大规模预训练+微调 NLP 模型
    │
2018.10  BERT (Google) ← MLM + 双向注意力，刷新 11 项纪录
    │
2019.02  GPT-2 (1.5B) ← Zero-shot 能力涌现
    │
2019.07  RoBERTa ← "BERT 应该这样训练"
    │
2019.10  T5 ← Encoder-Decoder, Text-to-Text 统一范式
    │
2020.01  Scaling Laws ← 发现幂律关系
    │
2020.05  GPT-3 (175B) ← In-Context Learning 涌现
    │
2023~    GPT-4, LLaMA, Mistral, Claude ← Decoder-only 绝对主流
```

---

## 本章小结

| 概念 | 一句话总结 |
|------|-----------|
| **BERT** | Transformer 编码器 + MLM + NSP，双向理解，适合分类/NER/搜索 |
| **GPT** | Transformer 解码器 + CLM，单向生成，适合文本生成和通用任务 |
| **MLM vs CLM** | MLM 双向但训练信号稀疏（15%），CLM 单向但训练信号密集（100%） |
| **输入表示** | BERT: Token + Segment + Position；GPT: Token + Position |
| **微调** | 两者都支持加任务头微调，GPT 路线逐渐转向 Prompt/ICL |
| **RoBERTa** | BERT 的"正确训练方式"：去 NSP + 更多数据 + 动态 Masking |
| **Scaling Laws** | 参数量、数据量、计算量与性能存在可预测的幂律关系 |
| **In-Context Learning** | 大模型涌现能力：给几个示例就能学会新任务，无需微调 |
| **Decoder-only 胜出** | 统一生成范式 + Scaling Laws + ICL 涌现 = 一个模型做所有事 |
| **BERT 仍有价值** | 搜索、NER、分类场景，小而快，性价比极高 |
| **Encoder-Decoder** | T5/BART，兼具双向理解和自回归生成，翻译摘要有优势 |

**下一章**：[微调与迁移学习](4_微调与迁移学习.md)——预训练好的 BERT/GPT 如何适配具体任务？从全参数微调到 LoRA、Prompt Tuning，再到 In-Context Learning，迁移学习的方式在不断演进。
