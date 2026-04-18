# AI学习：03 注意力机制与 Transformer

## 本模块导读

**学什么**：从注意力机制的直觉出发，完整拆解 Transformer 架构——现代 AI 最核心的基础设施。几乎所有大语言模型（GPT、BERT、LLaMA）和视觉模型（ViT）都基于 Transformer。

**为什么学**：Transformer 是理解后续所有模块（GPT、扩散模型中的 DiT、多模态模型等）的绝对前提。没有 Transformer 的知识，后面的内容完全无法理解。

**学完能做什么**：
- 理解 Self-Attention 的计算过程和直觉（为什么 QKV？为什么除以 √d？）
- 理解 Multi-Head Attention 的设计动机
- 从零理解 Transformer 编码器和解码器的完整架构
- 理解位置编码（正弦、RoPE）为什么需要以及如何工作
- 能手写一个简化版 Transformer

**前置知识**：[02 经典网络架构](../AI学习_02_经典网络架构/README.md)（特别是编码器-解码器框架和 Bahdanau 注意力）

**预估学习时间**：10~12 小时

**本模块常见卡点**：
- **QKV 的物理含义**：Query 决定"要找什么"，Key 决定"怎么被找到"，Value 决定"找到后取什么"。三者通过各自的线性投影从同一输入分化而来，切忌混为一谈。
- **掩码方向**：不同框架对掩码的布尔约定不同——PyTorch `nn.MultiheadAttention` 中 `True` = 禁止关注，HuggingFace 中 `0` = 禁止关注。切换框架时务必确认掩码方向，这是最常见的 bug 来源。
- **位置编码为什么需要**：Self-Attention 天然具有置换不变性，打乱输入顺序不影响输出。语言是有序的，必须通过位置编码显式注入"我在哪"的信息，否则模型无法区分"狗咬人"和"人咬狗"。

## 目录

| 序号 | 文件 | 内容 |
|------|------|------|
| 1 | [注意力机制](1_注意力机制.md) | 注意力的直觉与数学、缩放点积注意力、为什么需要 QKV |
| 2 | [Self-Attention与Multi-Head](2_SelfAttention与MultiHead.md) | 自注意力、多头注意力、掩码注意力 |
| 3 | [Transformer架构](3_Transformer架构.md) | 完整的编码器和解码器、FFN、残差连接、Layer Norm |
| 4 | [位置编码](4_位置编码.md) | 正弦位置编码、可学习位置编码、RoPE 旋转位置编码 |
| - | [论文与FAQ](论文与FAQ.md) | 关键论文、常见误区、延伸资源 |
