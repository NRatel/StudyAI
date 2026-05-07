# Transformer 架构

> **前置知识**：[Self-Attention 与 Multi-Head](2_SelfAttention与MultiHead.md)
>
> **本节目标**：理解 Transformer 由哪些核心部件组成，以及为什么它适合训练大模型。

---

## 1. Transformer 是一套可堆叠的文本加工流水线

Transformer 不是只有注意力。它把几个模块稳定地组合在一起，然后一层层堆叠。

每一层大致做两件事：

- 用注意力让 token 之间交换信息。
- 用前馈网络对每个 token 的表示再加工。

{{img:ch03_03_transformer_block_stack}}

## 2. 注意力层：先和上下文交换信息

Self-Attention 让每个 token 根据上下文更新自己。

比如“它”会从前面的“新手机”那里拿信息，“发布”会和“苹果”形成关系。

这一步解决的是：一句话里不同位置如何互相影响。

## 3. 前馈网络：再单独加工每个位置

注意力层负责“互相看”。前馈网络负责“自己想一想”。

它会对每个 token 的表示做非线性变换，让模型能提取更复杂的特征。

可以粗略理解为：

```text
注意力：收集上下文
前馈网络：加工当前表示
```

## 4. 残差连接和归一化：让深层网络更稳

Transformer 往往堆很多层。层数一多，训练容易不稳定。

残差连接像一条快捷通道，让原信息可以绕过某些加工步骤继续往后传。归一化则像把数值尺度整理一下，避免训练过程忽大忽小。

它们不是最显眼的模块，但对稳定训练非常关键。

## 5. 为什么 Transformer 适合大模型

和 RNN 相比，Transformer 的优势很明显：

- 不必严格按顺序一步步读，更容易并行训练。
- 注意力可以直接连接远距离 token。
- 层堆叠后表达能力强。
- 架构统一，能扩展到文本、图像、语音、多模态。

这也是为什么 GPT、BERT、T5、ViT 等模型都围绕 Transformer 展开。

## 6. Encoder、Decoder、Decoder-only

Transformer 可以有不同用法：

- **Encoder-only**：擅长理解，如 BERT。
- **Encoder-Decoder**：适合输入输出都较复杂的任务，如翻译、摘要。
- **Decoder-only**：擅长自回归生成，如 GPT。

大语言模型主流多采用 Decoder-only，因为它和“预测下一个 token”的训练方式天然匹配。
