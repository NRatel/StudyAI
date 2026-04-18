# ViT 与 CLIP

> **前置知识**：[03 注意力机制与Transformer](../AI学习_03_注意力机制与Transformer/README.md)（Self-Attention、Multi-Head Attention、Transformer 架构）、[05 GPT系列](../AI学习_05_GPT系列深度解析/README.md)（预训练范式）
>
> **本节目标**：理解 Vision Transformer 如何将图像建模为 token 序列、CLIP 如何通过对比学习实现图文对齐、以及它们为什么是多模态 AI 的基石。
>
> **内容时效**：截至 2026 年 4 月。

---

## 1. 直觉与概述

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

CLIP 训练好的视觉编码器（通常就是 ViT），成为了后续几乎所有多模态大模型的"眼睛"——GPT-4V、LLaVA、InternVL 等都直接使用或基于 CLIP 的视觉编码器。

---

## 2. 严谨定义与原理

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

## 3. Python 代码实战

### 3.1 从零实现简化版 ViT

```python
"""
简化版 ViT：理解 Patch Embedding → Transformer → 分类 的完整流程。
依赖: pip install torch torchvision
"""
import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """将图像切分为 patch 并投影为 token 向量"""

    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2  # 14*14 = 196
        # 用卷积实现 "切 patch + 线性投影" 一步到位
        # 卷积核大小 = patch 大小，步长 = patch 大小 → 不重叠地切割
        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x: (B, C, H, W) → (B, embed_dim, H/P, W/P) → (B, embed_dim, N) → (B, N, embed_dim)
        x = self.projection(x)           # (B, 768, 14, 14)
        x = x.flatten(2)                 # (B, 768, 196)
        x = x.transpose(1, 2)            # (B, 196, 768)
        return x


class SimpleViT(nn.Module):
    """
    简化版 Vision Transformer。
    省略了 DropPath 等细节，聚焦核心架构。
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embed_dim=768,
        num_heads=12,
        num_layers=12,
    ):
        super().__init__()

        # 1. Patch Embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # 2. [CLS] Token（可学习）
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # 3. 位置编码（可学习）
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))

        # 4. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True,      # Pre-Norm（ViT 使用的变体）
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 5. 分类头
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]

        # Patch Embedding: (B, 3, 224, 224) → (B, 196, 768)
        x = self.patch_embed(x)

        # 添加 [CLS] Token: (B, 196, 768) → (B, 197, 768)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # 添加位置编码
        x = x + self.pos_embed

        # Transformer Encoder
        x = self.transformer(x)

        # 取 [CLS] Token 的输出作为图像表示
        cls_output = x[:, 0]              # (B, 768)
        cls_output = self.norm(cls_output)

        # 分类
        logits = self.head(cls_output)    # (B, num_classes)
        return logits


# ---------- 验证 ----------
if __name__ == "__main__":
    model = SimpleViT(
        img_size=224, patch_size=16, num_classes=10,
        embed_dim=384, num_heads=6, num_layers=6,  # 小配置，便于测试
    )
    dummy_img = torch.randn(2, 3, 224, 224)  # batch_size=2
    logits = model(dummy_img)
    print(f"输入: {dummy_img.shape}")          # (2, 3, 224, 224)
    print(f"输出: {logits.shape}")             # (2, 10)
    print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # 查看中间维度
    patches = model.patch_embed(dummy_img)
    print(f"\nPatch Embedding 输出: {patches.shape}")  # (2, 196, 384)
    print(f"Patch 数量: {model.patch_embed.num_patches}")  # 196 = 14*14
```

### 3.2 使用 CLIP 进行零样本图像分类

```python
"""
CLIP 零样本分类：无需训练数据，用文本描述即可分类图像。
依赖: pip install torch transformers pillow requests
"""
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import requests
from io import BytesIO

# 加载 CLIP 模型（首次运行会下载约 600MB）
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)
model.eval()

# ================================================================
# 示例 1：零样本分类
# ================================================================

# 下载一张测试图片（也可以用本地图片）
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
try:
    image = Image.open(BytesIO(requests.get(url, timeout=10).content))
except Exception:
    # 如果下载失败，创建一个模拟图像
    image = Image.new("RGB", (224, 224), color="gray")
    print("(使用模拟图像)")

# 定义候选类别（用自然语言描述）
candidate_labels = [
    "a photo of a cat",
    "a photo of a dog",
    "a photo of a bird",
    "a photo of a car",
    "a photo of a person",
]

# 编码图像和文本
inputs = processor(
    text=candidate_labels,
    images=image,
    return_tensors="pt",
    padding=True,
)

with torch.no_grad():
    outputs = model(**inputs)

# CLIP 返回图文相似度 logits（未归一化）
logits_per_image = outputs.logits_per_image  # (1, 5)
probs = logits_per_image.softmax(dim=-1)     # 归一化为概率

print("=== 零样本分类结果 ===")
for label, prob in sorted(zip(candidate_labels, probs[0]), key=lambda x: -x[1]):
    bar = "█" * int(prob * 40)
    print(f"  {prob:.4f} {bar} {label}")

# ================================================================
# 示例 2：图文相似度检索
# ================================================================

# 假设我们有一组图片描述（在实际应用中，这些是图像的 embedding）
texts = [
    "a cat sleeping on a sofa",
    "a dog playing in the park",
    "sunset over the ocean",
    "a city skyline at night",
]

# 计算文本 embedding
text_inputs = processor(text=texts, return_tensors="pt", padding=True)
with torch.no_grad():
    text_features = model.get_text_features(**text_inputs)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# 计算图像 embedding
image_inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    image_features = model.get_image_features(**image_inputs)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

# 计算余弦相似度
similarities = (image_features @ text_features.T).squeeze(0)
print("\n=== 图文相似度检索 ===")
for text, sim in sorted(zip(texts, similarities), key=lambda x: -x[1]):
    print(f"  {sim:.4f}  {text}")

# ================================================================
# 示例 3：查看 embedding 维度
# ================================================================
print(f"\n=== Embedding 维度 ===")
print(f"图像 embedding: {image_features.shape}")  # (1, 512)
print(f"文本 embedding: {text_features.shape}")    # (4, 512)
print(f"图文共享空间维度: {image_features.shape[-1]}")  # 512
```

### 3.3 CLIP 对比学习损失实现

```python
"""
CLIP 对比损失的核心实现：理解 InfoNCE 损失如何对齐图文空间。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPLoss(nn.Module):
    """
    CLIP 使用的对称对比损失（Symmetric InfoNCE）。

    核心思想：
    - 一个 batch 中有 N 个 (图像, 文本) 对
    - 对角线位置是正样本对（匹配的图文）
    - 非对角线位置是负样本对（不匹配的图文）
    - 同时从"图像找文本"和"文本找图像"两个方向优化
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        # 温度参数：可学习（CLIP 原文做法）
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(1.0 / temperature)))

    def forward(self, image_features, text_features):
        """
        Args:
            image_features: (N, D) L2 归一化后的图像特征
            text_features:  (N, D) L2 归一化后的文本特征
        Returns:
            loss: 标量
        """
        # 获取温度（限制范围防止数值不稳定）
        temperature = torch.clamp(self.log_temperature.exp(), min=0.01, max=100.0)

        # 计算 N×N 相似度矩阵（余弦相似度，因为输入已 L2 归一化）
        # logits[i][j] = sim(image_i, text_j) / temperature
        logits = image_features @ text_features.T / temperature  # (N, N)

        # 标签：对角线位置是正样本 → labels = [0, 1, 2, ..., N-1]
        N = logits.shape[0]
        labels = torch.arange(N, device=logits.device)

        # 图像→文本方向：每行做 softmax，正确答案在第 i 列
        loss_i2t = F.cross_entropy(logits, labels)

        # 文本→图像方向：每列做 softmax，正确答案在第 i 行
        loss_t2i = F.cross_entropy(logits.T, labels)

        # 对称损失
        loss = (loss_i2t + loss_t2i) / 2
        return loss


# ---------- 演示 ----------
if __name__ == "__main__":
    torch.manual_seed(42)
    N, D = 8, 512  # 8 个图文对，512 维特征

    # 模拟编码器输出（L2 归一化）
    img_feat = F.normalize(torch.randn(N, D), dim=-1)
    txt_feat = F.normalize(torch.randn(N, D), dim=-1)

    loss_fn = CLIPLoss(temperature=0.07)
    loss = loss_fn(img_feat, txt_feat)
    print(f"对比损失: {loss.item():.4f}")

    # 让正样本对靠近后，损失应该下降
    # 模拟训练后：让 img_feat[i] ≈ txt_feat[i]
    txt_feat_aligned = img_feat + 0.1 * torch.randn(N, D)
    txt_feat_aligned = F.normalize(txt_feat_aligned, dim=-1)

    loss_aligned = loss_fn(img_feat, txt_feat_aligned)
    print(f"对齐后损失: {loss_aligned.item():.4f}")
    print(f"损失下降: {loss.item() - loss_aligned.item():.4f}")

    # 查看相似度矩阵（对齐后对角线应该最大）
    sim_matrix = img_feat @ txt_feat_aligned.T
    print(f"\n对齐后相似度矩阵对角线均值: {sim_matrix.diag().mean():.4f}")
    print(f"对齐后相似度矩阵非对角线均值: {(sim_matrix.sum() - sim_matrix.diag().sum()) / (N*N - N):.4f}")
```

---

## 4. 工程师视角

### 4.1 ViT 的工程实践要点

#### 4.1.1 Patch 大小的工程权衡

| patch_size | Token 数（224×224） | 计算量 | 细粒度 | 适用场景 |
|-----------|---------------------|--------|--------|---------|
| 32 | 49 | 很低 | 低 | 快速原型、资源受限 |
| 16 | 196 | 中等 | 中等 | 通用分类（最常用） |
| 14 | 256 | 较高 | 较高 | 高精度分类 |
| 8 | 784 | 很高 | 高 | 密集预测（分割、检测） |

**关键**：Transformer 的计算复杂度是 $O(N^2)$，$N$ 是 token 数。patch 大小从 16 减到 8，token 数增加 4 倍，计算量增加约 16 倍。

#### 4.1.2 位置编码的工程启示

ViT 的可学习位置编码有一个实用特性：通过**双线性插值**，可以将低分辨率训练的位置编码迁移到高分辨率推理。

```
训练：224×224，patch=16 → 14×14 = 196 个位置编码
推理：384×384，patch=16 → 24×24 = 576 个位置编码

做法：将 14×14 的位置编码网格双线性插值为 24×24
```

这使得 ViT 可以在低分辨率上训练（节省计算），在高分辨率上推理（获得更好效果）。

### 4.2 CLIP 的工程应用模式

#### 4.2.1 CLIP 的四种典型应用

| 应用 | 做法 | 典型场景 |
|------|------|---------|
| **零样本分类** | 类别文本 embedding 作为分类器权重 | 快速原型、动态类别 |
| **图文检索** | 图像/文本 embedding 存入向量库 | 搜索引擎、电商推荐 |
| **视觉编码器** | 取 CLIP ViT 作为下游模型的"眼睛" | GPT-4V、LLaVA 等多模态大模型 |
| **图像过滤/审核** | 计算图像与违规描述的相似度 | 内容审核、数据清洗 |

#### 4.2.2 Prompt Engineering for CLIP

CLIP 的零样本效果高度依赖文本 prompt 的设计：

```python
# 差的 prompt → 准确率低
labels_bad = ["cat", "dog", "car"]

# 好的 prompt → 准确率显著提升
labels_good = ["a photo of a cat", "a photo of a dog", "a photo of a car"]

# 更好的 prompt（ensemble of prompts）→ 进一步提升
labels_ensemble = [
    ["a photo of a cat", "a close-up of a cat", "a picture of a cat"],
    ["a photo of a dog", "a close-up of a dog", "a picture of a dog"],
    # ...
]
# 对每个类别的多个 prompt 的 embedding 取平均，再用于分类
```

OpenAI 发现，80 个精心设计的 prompt 模板的 ensemble，可以在 ImageNet 上提升约 3.5% 的零样本准确率。

#### 4.2.3 CLIP 的局限性

| 局限 | 表现 | 原因 | 解决方案 |
|------|------|------|---------|
| **细粒度识别弱** | 区分不了"藏獒"和"松狮" | 文本描述粒度不够 | 在具体领域数据上微调 |
| **计数能力差** | "三只猫"和"五只猫"分不清 | 对比学习不关注数量 | 用检测模型辅助 |
| **空间关系弱** | "左边猫右边狗"理解不好 | 文本编码器未编码空间 | 用多模态大模型 |
| **域外泛化** | 在医学/卫星图像上效果差 | 训练数据以自然图像为主 | 领域适配微调 |

### 4.3 从 CLIP 到多模态大模型的桥梁

CLIP 训练好的视觉编码器是后续多模态大模型的核心组件：

```
CLIP 的遗产（2021 → 至今）
│
├── CLIP ViT 作为视觉编码器
│   ├── LLaVA：CLIP ViT-L/14 → MLP 投影 → LLM
│   ├── GPT-4V：（推测）改进的 ViT → 适配层 → GPT-4
│   └── InternVL：InternViT-6B（基于 CLIP 训练范式）→ LLM
│
├── CLIP 的对比学习范式被继承
│   ├── SigLIP：用 sigmoid 替代 softmax，去掉全局归一化
│   ├── EVA-CLIP：稳定大规模 CLIP 训练
│   └── MetaCLIP：改进训练数据策略
│
└── CLIP embedding 空间被复用
    ├── Stable Diffusion：CLIP 文本编码器指导图像生成
    ├── DALL-E 2：用 CLIP 空间做图文桥接
    └── 图文检索系统：直接用 CLIP embedding 做索引
```

### 4.4 常见面试 / 系统设计问题

| 问题 | 核心要点 |
|------|---------|
| ViT 和 CNN 有什么本质区别？ | CNN 靠局部卷积叠加感受野，归纳偏置强但扩展性有限；ViT 用全局注意力，需要更多数据但扩展性更好 |
| CLIP 为什么能做零样本分类？ | 对比学习让图文在共享空间对齐；分类时用文本 embedding 当分类器权重，无需见过具体类别 |
| CLIP 的 batch size 为什么要 32K？ | 对比学习依赖负样本数量；batch 越大，负样本越多，学到的表示越好 |
| 如何把 CLIP 用于特定领域？ | 在领域数据上继续对比学习微调（如 BiomedCLIP 用于医学图像） |

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
