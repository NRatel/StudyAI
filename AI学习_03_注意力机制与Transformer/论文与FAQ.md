# 论文与FAQ

> 本文件用于论文脉络、误区澄清与延伸阅读，不替代主章节正文。

> 本文档汇总注意力机制与 Transformer 领域的关键论文、常见误区与延伸资源，供系统性学习和查阅使用。

---

## 一、关键论文与里程碑（延伸阅读）

以下论文构成了从注意力机制诞生到 Transformer 架构成熟、再到高效注意力与位置编码优化的核心脉络。建议按时间顺序阅读，体会这条"注意力即一切"的技术演化路线。

### 1. Bahdanau, Cho & Bengio, 2014 — *Neural Machine Translation by Jointly Learning to Align and Translate*

**简评：** 注意力机制的开山之作，让解码器学会"在正确的时间关注正确的位置"，彻底突破了 Seq2Seq 固定长度上下文向量的信息瓶颈。

- **核心贡献：** 提出加性注意力（Additive Attention）机制。在经典 Seq2Seq 中，编码器必须把整个输入序列压缩成一个固定长度的向量，这对长句子来说信息损失严重。Bahdanau 的方案是：解码器每生成一个词时，通过一个小型对齐网络（alignment model）计算当前解码状态与编码器每个位置的"相关性得分"，对编码器隐藏状态做加权求和，得到动态的上下文向量。具体来说，对齐得分 $e_{ij} = v^T \tanh(W_1 s_{i-1} + W_2 h_j)$ 经 softmax 归一化后得到注意力权重 $\alpha_{ij}$，上下文向量 $c_i = \sum_j \alpha_{ij} h_j$。
- **历史意义：** 这篇论文不仅大幅提升了机器翻译质量（尤其是长句子），更深远的影响在于：它提出的"动态加权聚合信息"思想直接启发了后续的自注意力机制和 Transformer 架构。可以说，没有 Bahdanau Attention，就没有后来的 "Attention Is All You Need"。同时，注意力权重的可视化还提供了模型的可解释性——我们可以看到模型在翻译每个词时"注意"了源语言的哪些位置。

### 2. Luong, Pham & Manning, 2015 — *Effective Approaches to Attention-based Neural Machine Translation*

**简评：** 系统比较了多种注意力计算方式，提出了更高效的乘性注意力（Multiplicative Attention），并区分了全局注意力与局部注意力两种策略。

- **核心贡献：** 提出三种注意力得分计算方式——点积（dot）$s^T h$、通用（general）$s^T W h$、拼接（concat，即 Bahdanau 的加性方式）。其中点积注意力计算最简单、速度最快，且效果不亚于加性注意力。同时提出局部注意力（Local Attention）的概念，只关注源序列的一个窗口，降低计算开销。
- **历史意义：** Luong 的乘性注意力（特别是点积形式）直接成为了 Transformer 中"缩放点积注意力"（Scaled Dot-Product Attention）的前身。论文中对不同注意力变体的系统对比，也为后续研究者选择注意力机制提供了重要参考。

### 3. Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser & Polosukhin, 2017 — *Attention Is All You Need*

**简评：** 抛弃循环和卷积，提出完全基于注意力机制的 Transformer 架构——这篇论文改变了整个 AI 领域的走向。

- **核心贡献：**
  - **自注意力（Self-Attention）：** 序列中的每个位置都与所有其他位置计算注意力，直接建模全局依赖关系，不再需要 RNN 的逐步传递。计算公式 $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$，其中 $\sqrt{d_k}$ 的缩放因子防止点积值过大导致 softmax 梯度消失。
  - **多头注意力（Multi-Head Attention）：** 将 Q、K、V 投影到多个低维子空间分别做注意力，再拼接合并。不同的头可以学习不同类型的关系模式（如语法关系、语义关系、位置关系等）。
  - **位置编码（Positional Encoding）：** 由于自注意力本身对位置不敏感（排列不变性），使用正弦/余弦函数生成的位置编码注入序列的位置信息。
  - **编码器-解码器架构：** 编码器由 N 层相同的块堆叠而成，每块包含自注意力层和前馈网络（FFN），配合残差连接和 Layer Normalization。解码器额外增加了交叉注意力层（Cross-Attention）和因果掩码（Causal Mask）。
- **历史意义：** Transformer 不仅在机器翻译上取得了 SOTA，更催生了整个现代 AI 生态：BERT（仅编码器）、GPT 系列（仅解码器）、T5（编码器-解码器）、Vision Transformer（ViT）、扩散模型中的 DiT 等。Transformer 已经成为跨越 NLP、CV、多模态等几乎所有 AI 领域的通用基础架构。

### 4. Shaw, Uszkoreit & Vaswani, 2018 — *Self-Attention with Relative Position Representations*

**简评：** 将位置信息从"绝对坐标"改为"相对距离"，让注意力机制原生理解"谁离谁近"，增强了模型的泛化能力。

- **核心贡献：** 提出在自注意力计算中直接引入相对位置偏置。具体做法是在注意力得分和加权求和中分别加入可学习的相对位置嵌入 $a_{ij}^K$ 和 $a_{ij}^V$，使得注意力不仅依赖内容相似度，还依赖两个位置之间的相对距离。修改后的注意力得分为 $e_{ij} = \frac{x_i W^Q (x_j W^K + a_{ij}^K)^T}{\sqrt{d_k}}$。同时引入了距离裁剪（clipping），超过一定距离的位置共享相同的位置嵌入，既限制了参数量，也反映了"距离太远则区分意义不大"的直觉。
- **历史意义：** 相对位置编码是后续所有位置编码改进工作的重要起点。它揭示了一个关键洞察：对于自然语言来说，"两个词之间的距离"往往比"每个词在句子中的绝对位置"更重要。这一思想直接影响了后续的 T5 相对位置偏置、ALiBi（Attention with Linear Biases）、以及 RoPE 等方案的设计。

### 5. Su, Lu, Pan, Murtadha, Wen & Liu, 2021 — *RoFormer: Enhanced Transformer with Rotary Position Embedding*

**简评：** 用旋转矩阵将位置信息优雅地编码进注意力计算中，兼顾了绝对位置编码的简洁性和相对位置编码的泛化性。

- **核心贡献：** 提出旋转位置编码（Rotary Position Embedding, RoPE）。核心思想极其精巧：对查询向量 $q$ 和键向量 $k$ 的每一对相邻维度施加一个与位置相关的旋转操作。具体地，位置 $m$ 处的向量被旋转角度 $m\theta_i$（$\theta_i$ 随维度变化），使得两个位置 $m$ 和 $n$ 的内积自然包含了相对位置信息 $m - n$。数学上，$\langle R_m q, R_n k \rangle = \langle R_{m-n} q, k \rangle$，旋转操作天然将绝对位置编码转化为了相对位置关系。
- **历史意义：** RoPE 已经成为当前大语言模型的事实标准位置编码方案。LLaMA、Qwen、Mistral、Gemma 等主流开源模型均采用 RoPE。其核心优势在于：（1）不引入额外参数；（2）天然支持相对位置感知；（3）通过频率外推或 NTK-aware 缩放等技术，可以扩展到远超训练长度的序列。

### 6. Dao, Fu, Ermon, Rudra & Re, 2022 — *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*

**简评：** 不改变注意力的数学计算，仅通过重新组织 GPU 内存访问模式，实现了 2-4 倍的加速和显著的显存节省——一个纯系统优化带来巨大影响的经典案例。

- **核心贡献：** 标准注意力的实现需要将 $N \times N$ 的注意力矩阵完整写入 GPU 高带宽内存（HBM），这在长序列时既慢又费内存。FlashAttention 采用分块（tiling）策略：将 Q、K、V 切成小块，在 GPU 片上 SRAM 中完成注意力计算，避免将完整注意力矩阵写回 HBM。通过在线 softmax 技巧（online softmax，维护运行中的最大值和求和值），实现了精确的（非近似的）注意力计算。内存复杂度从 $O(N^2)$ 降至 $O(N)$，同时由于减少了 HBM 读写，实际速度也大幅提升。
- **历史意义：** FlashAttention 揭示了一个重要事实：现代 GPU 上的性能瓶颈往往不是计算量（FLOPs），而是内存访问（IO）。这一 IO-aware 的设计理念影响深远。FlashAttention 已被 PyTorch 2.0+ 原生集成，成为训练和推理大模型的标准基础设施。后续的 FlashAttention-2 和 FlashAttention-3 进一步优化了并行度和硬件利用率。

### 7. Xiong, Yang, He, Zheng, Zheng, Xing, Zhang, Lan, Wang & Liu, 2020 — *On Layer Normalization in the Transformer Architecture*

**简评：** 系统分析了 Transformer 中 Layer Normalization 的放置位置对训练稳定性的影响，为 Pre-LN 设计提供了理论依据。

- **核心贡献：** 原始 Transformer 使用 Post-LN（先做子层计算，再做 LayerNorm），这种配置在深层 Transformer 中容易出现训练不稳定问题，需要精心的学习率预热。论文分析了 Post-LN 中梯度在初始化阶段的放大效应，证明了 Pre-LN（先做 LayerNorm，再做子层计算）可以使梯度范数在各层间保持稳定，从而允许更大的学习率和更少的预热步数。
- **历史意义：** Pre-LN 已成为现代大模型（GPT-2/3、LLaMA 等）的默认选择。这篇论文是理解"为什么现代 Transformer 的 LayerNorm 位置与原始论文不同"的关键参考。

### 论文时间线表格

| 年份 | 论文 | 核心概念 | 解决的核心问题 |
|------|------|----------|----------------|
| 2014 | Bahdanau et al. — 加性注意力 | **Attention 机制** | 突破 Seq2Seq 固定上下文向量的信息瓶颈 |
| 2015 | Luong et al. — 乘性注意力 | **点积注意力 / 局部注意力** | 更高效的注意力计算方式 |
| 2017 | Vaswani et al. — Attention Is All You Need | **Transformer** | 完全基于注意力的并行化架构，替代 RNN |
| 2018 | Shaw et al. — 相对位置表示 | **相对位置编码** | 让注意力感知位置间距离而非绝对坐标 |
| 2020 | Xiong et al. — Pre-LN 分析 | **Pre-LN vs Post-LN** | 解决深层 Transformer 训练不稳定问题 |
| 2021 | Su et al. — RoFormer | **RoPE 旋转位置编码** | 用旋转矩阵统一绝对/相对位置编码 |
| 2022 | Dao et al. — FlashAttention | **IO-aware 注意力** | 通过内存访问优化实现精确注意力的加速和省内存 |

> **演化脉络一览**：Bahdanau Attention(2014) -> Luong Attention(2015) -> Transformer(2017) 构成了注意力机制从配角到主角的演进主线；正弦位置编码(2017) -> 相对位置编码(2018) -> RoPE(2021) 构成了位置编码的演进主线；标准注意力实现 -> FlashAttention(2022) 构成了工程优化的主线。Pre-LN(2020) 则是架构细节的重要修正。

---

## 二、常见误区与 FAQ

### FAQ 1：Transformer 真的不需要 RNN 吗？没有循环结构，怎么处理序列的先后顺序？

**误区：** "Transformer 完全不关心顺序，所以处理序列一定不如 RNN。"

**澄清：** Transformer 确实完全抛弃了循环结构，但这不意味着它忽略了顺序信息。

**RNN 处理顺序的方式：** RNN 通过逐步处理（$t=1, 2, 3, \ldots$）隐式地编码了序列顺序——第 3 个时间步的隐状态自然"知道"自己是第 3 个，因为它是由前两步的信息递归得来的。

**Transformer 处理顺序的方式：** 自注意力本身是排列不变的（permutation invariant）——如果你打乱输入序列的顺序，注意力的输出也只是相应地打乱，不会改变注意力权重。为了注入位置信息，Transformer 依赖**位置编码**（Positional Encoding）。位置编码被加到输入嵌入上，让模型"看到"每个 token 在序列中的位置。

**Transformer 不用 RNN 的核心优势：**

1. **并行化**：RNN 必须按时间步顺序计算（$h_t$ 依赖 $h_{t-1}$），无法并行。Transformer 的自注意力可以一次性计算所有位置之间的关系，充分利用 GPU 的并行能力。
2. **长距离依赖**：RNN 中距离为 $L$ 的两个位置需要信息经过 $L$ 步传递，梯度容易消失。Transformer 中任意两个位置之间只需一步注意力计算，路径长度为 $O(1)$。

**需要注意的代价：** Transformer 的自注意力计算复杂度为 $O(N^2)$（$N$ 为序列长度），而 RNN 为 $O(N)$。这在超长序列场景下是 Transformer 的劣势，也是 FlashAttention、线性注意力等后续工作试图解决的问题。

**结论：** Transformer 不是"不需要顺序"，而是用位置编码这种显式的方式替代了 RNN 隐式的顺序建模，并因此换来了并行化和长距离建模的巨大优势。

---

### FAQ 2：为什么 QKV 要做线性变换，而不是直接用输入？

**误区：** "Q、K、V 的线性变换只是多了几个参数，去掉也差不多。"

**澄清：** 线性变换是自注意力机制中至关重要的设计，去掉它会导致模型能力急剧下降。原因可以从三个角度理解。

**角度一：功能分离**

在注意力机制中，Q（查询）、K（键）、V（值）扮演的角色完全不同：
- **Q 和 K** 共同决定"谁应该关注谁"（注意力权重的计算）
- **V** 决定"关注之后获取什么信息"（加权求和的内容）

如果 Q、K、V 都直接使用输入 $X$，那么同一个向量需要同时承担三种不同的功能。通过各自独立的线性变换 $W^Q$、$W^K$、$W^V$，模型可以为每种功能学习最合适的表示空间。

**角度二：打破对称性**

如果 Q = K = X（没有线性变换），那么注意力得分 $X X^T$ 是一个对称矩阵——位置 $i$ 对位置 $j$ 的注意力等于位置 $j$ 对位置 $i$ 的注意力。但在自然语言中，关系往往是不对称的。例如在"猫追老鼠"中，"追"对"猫"的关注程度（找主语）和"猫"对"追"的关注程度（找谓语）应该是不同的。线性变换打破了这种对称性：$Q K^T = (XW^Q)(XW^K)^T = XW^Q {W^K}^T X^T$，一般不再对称。

**角度三：维度灵活性**

线性变换允许将输入映射到不同的维度。在多头注意力中，每个头的 Q、K、V 维度是 $d_k = d_{\text{model}} / h$（$h$ 为头数），比输入维度低得多。如果没有线性变换，就无法实现这种维度压缩。

**实验验证：** 多项消融实验（ablation study）表明，去掉 QKV 的线性变换后，模型性能显著下降。这不仅仅是"多几个参数"的问题，而是自注意力能否有效工作的关键。

---

### FAQ 3：Multi-Head Attention 和单头大维度有什么区别？为什么要分成多头？

**误区：** "多头注意力只是把大矩阵拆成几个小矩阵算，本质上和一个大头一样。"

**澄清：** 多头注意力不是简单的矩阵分块，它在表达能力和学习行为上与单头有本质区别。

**参数量对比：** 假设模型维度 $d_{\text{model}} = 512$，多头注意力用 8 个头，每个头维度 $d_k = 64$。多头注意力和一个 $d_k = 512$ 的单头注意力的参数量几乎相同（都是 $3 \times d_{\text{model}}^2 + d_{\text{model}}^2$ 加上输出投影）。所以区别不在参数量，而在计算结构。

**多头注意力的核心优势：**

1. **多子空间关注**：每个头在自己的低维子空间中独立计算注意力模式。不同的头可以学习关注不同类型的关系。研究发现，Transformer 的不同头确实会自发地学到不同功能：有的头关注相邻位置（局部语法），有的头关注长距离依赖（跨句指代），有的头关注特定的语法结构（主谓关系）。
2. **正则化效果**：多头结构迫使模型在多个不同的子空间中都能有效地建模关系，避免过度依赖单一的注意力模式，起到了隐式正则化的作用。
3. **训练稳定性**：单头注意力的 softmax 在高维空间中容易产生非常尖锐的分布（趋近于 one-hot），导致梯度不稳定。多头机制通过降低每个头的维度缓解了这一问题。

**一个直觉类比：** 想象一群人在审阅一篇文章。一个人（单头）虽然很厉害，但视角单一。八个人（多头）分别从语法、逻辑、风格、用词等不同角度审阅，最后综合意见，结果往往更全面、更稳健。

**工程考量：** 多头注意力的每个头可以在 GPU 上并行计算，不增加计算时间。这是"免费"的表达能力提升。

---

### FAQ 4：Pre-LN 和 Post-LN 有什么区别？为什么现代模型大多用 Pre-LN？

**误区：** "LayerNorm 的位置无所谓，反正都是归一化。"

**澄清：** LayerNorm 的放置位置对 Transformer 的训练稳定性有重大影响，这是一个看似微小但实际关键的架构选择。

**两种结构的定义：**

```
Post-LN（原始 Transformer，Vaswani 2017）:
  output = LayerNorm(x + SubLayer(x))

Pre-LN（GPT-2、LLaMA 等现代模型）:
  output = x + SubLayer(LayerNorm(x))
```

**Post-LN 的问题：**
- 在深层 Transformer 中（例如 24 层以上），Post-LN 的梯度在初始化阶段会出现严重的放大效应。Xiong et al. (2020) 证明：Post-LN 中靠近输出层的梯度范数远大于靠近输入层的，导致训练初期参数更新极不均衡。
- 因此 Post-LN 必须使用精心设计的学习率预热（warmup），否则训练容易发散。

**Pre-LN 的优势：**
- LayerNorm 在子层计算之前执行，对每层的输入做了归一化，使梯度范数在各层间保持稳定。
- 可以使用更大的学习率、更少的预热步数，甚至在某些情况下不需要预热。
- 训练更稳定，尤其对于深层和大规模模型。

**Pre-LN 的代价：**
- 一些研究（Liu et al., 2020）发现，Post-LN 在训练充分的情况下最终性能可能略优于 Pre-LN。这可能是因为 Pre-LN 的残差路径没有经过归一化，在极深的网络中可能导致各层输出尺度不一致。
- 为此，后续出现了一些折中方案，如 Sandwich-LN（在子层前后各做一次 LayerNorm）、DeepNorm（通过缩放残差连接来稳定 Post-LN）等。

**实践结论：** 对于大多数场景，Pre-LN 是更安全的选择。除非你有特定的理由和充足的训练调优经验，否则不建议使用 Post-LN。

---

### FAQ 5：Transformer 的参数量怎么算？

**误区：** "Transformer 的参数主要在注意力层"或"参数量和序列长度有关"。

**澄清：** Transformer 的参数量与序列长度无关（序列长度只影响计算量和内存），参数量完全由模型维度 $d_{\text{model}}$、层数 $L$、注意力头数 $h$、前馈网络维度 $d_{\text{ff}}$ 决定。并且 FFN 层的参数往往占大头。

**单层 Transformer 编码器块的参数量：**

| 组件 | 参数矩阵 | 参数量 |
|------|----------|--------|
| 多头注意力 (MHA) | $W^Q, W^K, W^V$：各 $d_{\text{model}} \times d_{\text{model}}$ | $3 \times d_{\text{model}}^2$ |
| MHA 输出投影 | $W^O$：$d_{\text{model}} \times d_{\text{model}}$ | $d_{\text{model}}^2$ |
| 前馈网络 (FFN) | $W_1$：$d_{\text{model}} \times d_{\text{ff}}$，$W_2$：$d_{\text{ff}} \times d_{\text{model}}$ | $2 \times d_{\text{model}} \times d_{\text{ff}}$ |
| LayerNorm (x2) | $\gamma, \beta$：各 $d_{\text{model}}$ | $4 \times d_{\text{model}}$（可忽略） |
| 偏置项 | 各线性层的偏置 | 较小（可忽略） |

**单层合计（忽略偏置和 LN）：** $4 d_{\text{model}}^2 + 2 d_{\text{model}} \times d_{\text{ff}}$

**通常 $d_{\text{ff}} = 4 d_{\text{model}}$**，代入得：$4d^2 + 8d^2 = 12d^2$（其中 $d = d_{\text{model}}$）

也就是说，FFN 的参数量（$8d^2$）是注意力层（$4d^2$）的两倍。

**一个实际例子——GPT-3 175B：**
- $d_{\text{model}} = 12288$，$L = 96$ 层，$h = 96$ 头，$d_{\text{ff}} = 4 \times 12288 = 49152$
- 词表嵌入：$V \times d_{\text{model}} = 50257 \times 12288 \approx 6.2$ 亿
- 每层 Transformer 块：$12 \times 12288^2 \approx 18.1$ 亿
- 96 层总计：$18.1 \times 96 \approx 1738$ 亿
- 加上嵌入层，总计约 $1750$ 亿，与公开数据吻合

**关键结论：**
1. FFN 参数占总参数的约 2/3，注意力层只占约 1/3
2. 参数量与序列长度无关，与 $d_{\text{model}}^2 \times L$ 成正比
3. 计算量（FLOPs）则同时与序列长度和参数量相关

---

### FAQ 6：$\sqrt{d_k}$ 缩放因子到底为什么重要？不除会怎样？

**误区：** "除以 $\sqrt{d_k}$ 只是一个工程 trick，不除也能训。"

**澄清：** 这个缩放因子有严格的数学动机，不除的话训练大概率会失败。

**问题来源：** 假设 $Q$ 和 $K$ 的每个元素独立采样自均值为 0、方差为 1 的分布，那么 $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ 的均值为 0，但方差为 $d_k$（$d_k$ 个独立随机变量之积的方差之和）。当 $d_k$ 较大时（比如 64 或 128），点积的值会很大。

**不缩放的后果：** 大的点积值送入 softmax 后，输出会趋近于 one-hot 分布——最大值对应的位置概率接近 1，其他位置接近 0。在这种饱和区域，softmax 的梯度几乎为零，导致训练时梯度消失，注意力权重无法有效更新。

**缩放的效果：** 除以 $\sqrt{d_k}$ 后，点积的方差被归一化为 1，softmax 的输入处于一个梯度较大的合理范围内，训练可以正常进行。

**直觉理解：** 把 $\sqrt{d_k}$ 缩放想象成"温度控制"。不缩放相当于用很低的温度做 softmax（分布尖锐），缩放后相当于恢复到正常温度（分布平滑）。

---

### FAQ 7：位置编码为什么不直接用 1, 2, 3, ... 这样的数字？

**误区：** "直接用整数编号作为位置编码不是最直观吗？"

**澄清：** 直接用整数编号有几个严重的问题。

**问题一：数值尺度不可控。** 如果序列长度为 1000，那么最后一个位置的编码值是 1000，而第一个位置是 1。这会导致位置信息的数值范围远大于词嵌入的数值范围（词嵌入通常在 [-1, 1] 附近），位置信号会淹没语义信号。

**问题二：无法泛化到训练时未见过的长度。** 如果训练时最长序列是 512，那么模型从未见过位置编码值为 513 的输入，无法处理更长的序列。

**问题三：不包含相对位置信息。** 位置 100 和 101 之间的关系，与位置 1 和 2 之间的关系，应该是类似的（都相邻）。但整数编码中，100 和 101 在数值上离 1 和 2 很远，模型难以学到这种平移不变性。

**正弦位置编码的设计思路：** Vaswani (2017) 使用不同频率的正弦和余弦函数：$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d})$，$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})$。这种设计的巧妙之处在于：（1）值域有界（始终在 [-1, 1]）；（2）不同维度使用不同频率，低维度变化快（编码精细位置），高维度变化慢（编码粗略位置），形成多尺度的位置表示；（3）任意两个位置的编码之间的关系可以通过线性变换表达，天然包含相对位置信息。

---

## 三、延伸资源

> 以下资源为补充参考，本教材已覆盖注意力机制与 Transformer 的核心内容。如果希望进一步深入，可以选择性阅读。

### 1. Jay Alammar — *The Illustrated Transformer*

- **链接**：https://jalammar.github.io/illustrated-transformer/
- **推荐理由**：以精美的可视化图解 Transformer 的完整工作流程，从输入嵌入到最终输出，每一步都有清晰的示意图。是建立 Transformer 架构直觉的最佳视觉资源，适合在阅读本教材的同时对照参考。同系列的 *The Illustrated GPT-2* 和 *Visualizing A Neural Machine Translation Model* 也值得阅读。

### 2. Andrej Karpathy — *Let's build GPT: from scratch, in code, spelled out*

- **链接**：https://www.youtube.com/watch?v=kCc8FmEb1nY
- **推荐理由**：前 Tesla AI 总监从零开始用纯 Python/PyTorch 实现一个 GPT 模型，全程两小时手写代码并逐行讲解。涵盖自注意力、多头注意力、位置编码、残差连接、LayerNorm 等全部核心组件。适合在理解本教材理论后动手实践，真正做到"从零手写 Transformer"。

### 3. *Dive into Deep Learning* — 注意力机制与 Transformer 章节

- **链接**：https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html（英文）/ https://zh.d2l.ai/（中文）
- **推荐理由**：李沐等人编写的交互式教材，每个概念都配有可运行的 PyTorch 代码。特别推荐其中"注意力提示""注意力汇聚""多头注意力""自注意力和位置编码""Transformer"等章节，理论推导和代码实现一一对应，适合深入理解细节。
