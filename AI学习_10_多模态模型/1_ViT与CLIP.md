# ViT 与 CLIP

> **前置知识**：[03 注意力机制与Transformer](../AI学习_03_注意力机制与Transformer/README.md)（Self-Attention、Multi-Head Attention、Transformer 架构）、[05 GPT系列](../AI学习_05_GPT系列深度解析/README.md)（预训练范式）
>
> **本节目标**：理解 Vision Transformer 如何将图像建模为 token 序列、CLIP 如何通过对比学习实现图文对齐、以及它们为什么是多模态 AI 的基石。
>
> **内容时效**：截至 2026 年 4 月。

---

## 1. 风趣易懂的直觉讲解
### 1.1 核心问题：图像怎么进入 Transformer？

Transformer 最初是为序列数据（文本）设计的——输入是一个 token 序列，每个 token 是一个向量。但图像不是序列，它是一个二维像素网格。

CNN 是传统的图像处理方案：卷积核在图像上滑动，逐层提取局部特征。但 CNN 有一个根本局限——**感受野是局部的**。即使通过多层卷积扩大感受野，CNN 也很难高效捕捉图像中远距离区域之间的关系。

**ViT 的核心思路**：把图像切成一个个小方块（patch），把每个 patch 当作一个"token"，然后直接送入标准 Transformer。

类比：**ViT = 把图像当作一篇文章来读**。每个 patch 就像一个"词"，Transformer 的 Self-Attention 让每个"词"都能关注到文章中的任何其他"词"——这就是全局感受野。

### 1.2 核心问题：图像和文本怎么"对齐"？

ViT 解决了"图像怎么用 Transformer 处理"，但还有一个更根本的问题：**图像和文本活在完全不同的特征空间里**。一张猫的照片的特征向量，和"一只猫"这个文本的特征向量，毫无关系。

**CLIP 的核心思路**：训练一个图像编码器和一个文本编码器，让同一概念的图文描述在**共享向量空间**中靠近，不相关的图文描述远离。

类比：**CLIP = 给图像和文本建立一本共用的字典**。训练后，"一只猫"的文本向量和猫图片的图像向量，在同一个向量空间中距离很近。

### 1.3 为什么 ViT 和 CLIP 是多模态的基石？

```
传统路线:  图像 → CNN → 图像特征     文本 → RNN/Transformer → 文本特征
                   ↓                            ↓
              两个独立空间，无法直接比较

ViT + CLIP:  图像 → ViT → 图像特征 ─┐
                                      ├──→ 共享向量空间（可比较、可检索、可迁移）
             文本 → Transformer → 文本特征 ─┘
```

CLIP 训练出的"图文共享语义空间"成为后续多模态大模型的重要地基。LLaVA、早期 InternVL 等开源模型常直接使用或借鉴 CLIP/SigLIP 风格的视觉编码器；GPT-4V、GPT-4o 这类闭源模型没有公开具体视觉编码器，不能断言它们直接使用 CLIP，只能说它们延续了"视觉特征与语言语义对齐"这条路线。

> **为什么 CLIP 能成为后续多模态模型的桥接基础？** 根本原因在于 CLIP 通过对比学习建立了一个**图文共享的向量空间**——在这个空间中，图像和文本的语义是可比较、可对齐的。正是因为有了这个共享空间，后续的多模态大模型才能通过一个简单的投影层（如 MLP）将视觉特征"翻译"到 LLM 能理解的语义空间。如果没有 CLIP 预先建立的图文对齐基础，视觉编码器的输出对 LLM 来说就是毫无意义的噪声。可以说，CLIP 的图文共享向量空间是整个"桥接式多模态"技术路线的地基。

---

## 2. 准确概念定义与核心原理
### 2.1 Vision Transformer（ViT）

#### 2.1.1 架构总览

ViT 的完整流程（Dosovitskiy et al., 2020）：

```
输入图像 (H × W × C)
    │
    ▼
┌─────────────────────────────────┐
│ Step 1: Patch Embedding          │
│                                  │
│ 将图像切成 N = (H/P) × (W/P)    │
│ 个 patch，每个 patch 大小 P × P  │
│                                  │
│ 每个 patch 展平后线性投影为       │
│ D 维向量 → 得到 N 个 patch token │
└───────────┬─────────────────────┘
            │
            ▼
┌─────────────────────────────────┐
│ Step 2: 添加 [CLS] token         │
│ + 位置编码（Positional Embedding）│
│                                  │
│ [CLS], patch_1, patch_2, ...,   │
│ patch_N  → 共 N+1 个 token       │
└───────────┬─────────────────────┘
            │
            ▼
┌─────────────────────────────────┐
│ Step 3: 标准 Transformer Encoder │
│                                  │
│ L 层 (MSA + FFN + LayerNorm)     │
└───────────┬─────────────────────┘
            │
            ▼
┌─────────────────────────────────┐
│ Step 4: 取 [CLS] token 的输出    │
│ 作为整张图像的全局表示            │
│ → 接分类头（MLP）进行分类         │
└─────────────────────────────────┘
```

#### 2.1.2 Patch Embedding 的数学

给定输入图像 $\mathbf{x} \in \mathbb{R}^{H \times W \times C}$，patch 大小为 $P \times P$：

**Step 1**：将图像分割为 $N = \frac{H \cdot W}{P^2}$ 个 patch：

$$\mathbf{x}_p^{(i)} \in \mathbb{R}^{P^2 \cdot C}, \quad i = 1, \ldots, N$$

例如：224 x 224 的图像，patch 大小 16 x 16，得到 $N = 14 \times 14 = 196$ 个 patch。每个 patch 展平后是 $16 \times 16 \times 3 = 768$ 维向量。

**Step 2**：线性投影到 $D$ 维嵌入空间：

$$\mathbf{z}_0^{(i)} = \mathbf{x}_p^{(i)} \mathbf{E} + \mathbf{e}_{pos}^{(i)}, \quad \mathbf{E} \in \mathbb{R}^{(P^2 C) \times D}$$

其中 $\mathbf{e}_{pos}^{(i)}$ 是可学习的位置编码（position embedding），$D$ 是 Transformer 的隐藏维度。

**Step 3**：在序列开头添加一个可学习的 [CLS] token $\mathbf{z}_0^{(0)}$：

$$\mathbf{Z}_0 = [\mathbf{z}_0^{(0)},\; \mathbf{z}_0^{(1)},\; \ldots,\; \mathbf{z}_0^{(N)}] \in \mathbb{R}^{(N+1) \times D}$$

**Step 4**：送入 $L$ 层标准 Transformer Encoder：

$$\mathbf{Z}_l = \text{TransformerBlock}(\mathbf{Z}_{l-1}), \quad l = 1, \ldots, L$$

最终图像表示为 $\mathbf{Z}_L^{(0)}$（最后一层 [CLS] token 的输出）。

#### 2.1.3 ViT 的关键设计选择

| 设计 | ViT 的做法 | 为什么 |
|------|-----------|--------|
| **Patch 大小** | 16×16 或 14×14 | 太小 → token 数暴增（计算量平方级增长）；太大 → 分辨率损失 |
| **位置编码** | 可学习的 1D 位置编码 | 比固定 sin/cos 效果略好，且无需假设 2D 结构 |
| **[CLS] token** | 沿用 BERT 设计 | 让全局信息汇聚到一个 token |
| **预训练策略** | 大规模监督预训练（JFT-300M） | ViT 在小数据集上不如 CNN，需要大数据 |

**关键发现**：ViT 在小数据集（如 ImageNet-1K）上不如 CNN（缺少卷积的归纳偏置），但在大数据集（JFT-300M，3 亿图像）上显著超越 CNN。这印证了 Transformer 的核心优势——在充分数据下的扩展性（scalability）。

#### 2.1.4 ViT 家族演进（截至 2026.4）

| 模型 | 年份 | 核心改进 |
|------|------|---------|
| ViT | 2020 | 原始 patch + Transformer |
| DeiT | 2021 | 知识蒸馏，小数据也能训好 ViT |
| Swin Transformer | 2021 | 层级窗口注意力，适合密集预测 |
| MAE | 2022 | 自监督预训练（掩码补全） |
| DINOv2 | 2023 | 自蒸馏自监督，通用视觉特征 |
| SigLIP | 2023 | sigmoid 对比学习，替代 softmax |
| InternViT-6B | 2024 | 60 亿参数视觉编码器 |
| ViT-22B | 2023 | Google 220 亿参数 ViT |

### 2.2 CLIP（Contrastive Language-Image Pre-training）

#### 2.2.1 CLIP 的训练范式

CLIP（Radford et al., 2021）的核心是**对比学习**（contrastive learning）：

```
训练数据：4 亿个 (图像, 文本描述) 对

              图像编码器（ViT）          文本编码器（Transformer）
                    │                           │
              图像 → I_1                   文本 → T_1
              图像 → I_2                   文本 → T_2
              图像 → I_3                   文本 → T_3
                    │                           │
                    └─────────┬─────────────────┘
                              │
                   计算 N×N 相似度矩阵
                              │
                    ┌─────────────────────┐
                    │  T_1   T_2   T_3    │
                    │ ┌─────┬─────┬─────┐ │
              I_1   │ │ ✓   │ ✗   │ ✗   │ │  ← 对角线是正样本对
              I_2   │ │ ✗   │ ✓   │ ✗   │ │
              I_3   │ │ ✗   │ ✗   │ ✓   │ │  ← 非对角线是负样本对
                    │ └─────┴─────┴─────┘ │
                    └─────────────────────┘
                              │
                    最大化对角线相似度
                    最小化非对角线相似度
```

#### 2.2.2 对比学习损失函数

给定一个 batch 中的 $N$ 个 (图像, 文本) 对，CLIP 同时优化两个方向的交叉熵：

**图像 → 文本方向**（给定图像，找正确文本）：

$$\mathcal{L}_{\text{i2t}} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\text{sim}(\mathbf{I}_i, \mathbf{T}_i) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(\mathbf{I}_i, \mathbf{T}_j) / \tau)}$$

**文本 → 图像方向**（给定文本，找正确图像）：

$$\mathcal{L}_{\text{t2i}} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\text{sim}(\mathbf{T}_i, \mathbf{I}_i) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(\mathbf{T}_i, \mathbf{I}_j) / \tau)}$$

**总损失**：

$$\mathcal{L}_{\text{CLIP}} = \frac{1}{2}(\mathcal{L}_{\text{i2t}} + \mathcal{L}_{\text{t2i}})$$

其中：
- $\text{sim}(\mathbf{I}, \mathbf{T}) = \frac{\mathbf{I} \cdot \mathbf{T}}{|\mathbf{I}| \cdot |\mathbf{T}|}$ 是余弦相似度
- $\tau$ 是**可学习的温度参数**（初始约 0.07），控制分布的尖锐程度

#### 2.2.3 CLIP 的零样本分类

CLIP 最惊人的能力是**零样本分类**——无需任何标注数据，就能对新的类别进行图像分类。

原理：

```
Step 1: 将所有候选类别转化为文本 prompt
        "a photo of a cat"
        "a photo of a dog"
        "a photo of a car"
        ...

Step 2: 用文本编码器编码所有类别 → 得到 K 个文本向量

Step 3: 用图像编码器编码待分类图像 → 得到 1 个图像向量

Step 4: 计算图像向量与所有文本向量的余弦相似度

Step 5: 相似度最高的类别即为预测结果
```

**这为什么有效？** 因为 CLIP 在 4 亿图文对上训练后，已经学会了一个**通用的视觉-语言语义空间**。在这个空间里，"猫的照片"和"a photo of a cat"天然靠近。这意味着任何能用语言描述的视觉概念，都可以被零样本识别。

#### 2.2.4 CLIP 的关键设计

| 设计 | 具体做法 | 为什么 |
|------|---------|--------|
| **数据** | 4 亿网络图文对（WIT-400M） | 网络数据虽然有噪声，但规模弥补了质量 |
| **图像编码器** | ViT-L/14（最优配置） | ViT 在大规模数据上优于 ResNet |
| **文本编码器** | 12 层 Transformer（63M 参数） | 不需要特别大，能编码语义即可 |
| **Batch Size** | 32,768 | 对比学习需要大 batch 产生足够负样本 |
| **温度参数 τ** | 可学习，初始 0.07 | 自适应调节分布尖锐度 |
| **Prompt 设计** | "a photo of a {class}" | 手工 prompt 比裸类别名效果显著提升 |

---

## 本节小结

| 概念 | 一句话总结 |
|------|-----------|
| **ViT** | 将图像切为 patch 序列，用标准 Transformer 处理——证明了图像也不需要 CNN |
| **Patch Embedding** | 图像切块 → 展平 → 线性投影 → 位置编码，将 2D 图像转为 1D token 序列 |
| **[CLS] Token** | 沿用 BERT 设计，汇聚全局信息，其最终输出作为整图表示 |
| **CLIP** | 用 4 亿图文对做对比学习，让图像和文本在共享空间对齐 |
| **对比学习** | 最大化匹配图文对的相似度，最小化不匹配对的相似度 |
| **零样本分类** | 用文本描述作为分类器，无需标注数据即可分类新类别 |
| **CLIP 的遗产** | 其视觉编码器成为几乎所有多模态大模型的"眼睛" |

**下一节**：基于 ViT/CLIP 这个"眼睛"，多模态大模型如何把视觉能力接入 LLM？

---

> **时效性说明**（截至 2026 年 4 月）：ViT 和 CLIP 作为基础架构已相当稳定，核心原理不会过时。但其变体（如 SigLIP、EVA-CLIP、InternViT 等）仍在快速迭代。在选择具体的视觉编码器时，建议查阅最新的 Open CLIP benchmark 和 MMMU/MMBench 等多模态评测排行榜。2025-2026 年的重要趋势包括：(1) 视觉编码器参数量持续增大（InternViT-6B、ViT-22B）；(2) 自监督预训练（DINOv2）与对比学习的融合；(3) SigLIP 逐步替代原始 CLIP 成为新的默认选择。
