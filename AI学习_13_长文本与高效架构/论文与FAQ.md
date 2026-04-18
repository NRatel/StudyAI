# 论文与FAQ

> 本文档汇总长文本与高效架构领域的关键论文、常见误区与延伸资源，供系统性学习和查阅使用。
>
> **内容时效**：截至 2026 年 4 月

---

## 一、关键论文与里程碑

以下论文构成了从高效注意力到状态空间模型、从 MoE 稀疏架构到长文本扩展的核心技术脉络。按时间顺序阅读可以清晰看到"如何让 Transformer 又长又快"这一主题的演进。

### 1. Kitaev, Kaiser & Levskaya, 2020 — *Reformer: The Efficient Transformer*

**简评**：首次系统性地挑战 Transformer 的 O(n^2) 效率问题，提出了用局部敏感哈希（LSH）降低注意力复杂度的方案。

- **核心贡献**：提出两个关键改进——（1）**LSH 注意力**：通过局部敏感哈希将 token 分桶，只在同一桶内计算注意力，复杂度从 $O(n^2)$ 降到 $O(n \log n)$；（2）**可逆残差网络**：借鉴 RevNet 的思想，不存储中间激活值而是在反向传播时重新计算，将内存占用从 $O(L)$（层数）降到 $O(1)$。
- **核心思想**：注意力矩阵通常是稀疏的——大多数注意力权重接近零。与其计算全部 $n^2$ 个注意力分数再发现大部分接近零，不如直接找出那些"会有高注意力权重"的 token 对。LSH 通过哈希函数将相似的向量映射到同一桶中，从而高效地找到这些高权重的 token 对。
- **局限与后续**：LSH 的哈希分桶引入了随机性，需要多轮哈希来保证精度；工程实现复杂度高。后续被更简洁的方案（如 Longformer 的窗口注意力、FlashAttention 的 IO 优化）在实践中取代。但 Reformer 开创了"高效 Transformer"这个重要的研究方向。

### 2. Beltagy, Peters & Cohan, 2020 — *Longformer: The Long-Document Transformer*

**简评**：用滑动窗口 + 全局注意力的简洁组合，让 Transformer 高效处理长文档，是稀疏注意力在实际应用中最成功的范例。

- **核心贡献**：提出三种注意力模式的组合——（1）**滑动窗口注意力**：每个 token 只关注左右 $w/2$ 个邻居，复杂度 $O(n \cdot w)$；（2）**扩张窗口注意力**：类似 dilated convolution，窗口中每隔 $g$ 个位置取一个，扩大感受野而不增加计算量；（3）**全局注意力**：少数任务相关的 token（如分类任务的 [CLS]、问答任务的问题 token）可以关注全部位置。
- **核心优势**：与 Reformer 的 LSH 注意力相比，Longformer 的实现更简单、更确定性、更易于在现有深度学习框架中实现。多层堆叠后，滑动窗口的感受野随层数线性增长（$L$ 层 × 窗口 $w$ = 感受野 $Lw$），可以覆盖非常长的序列。
- **实际影响**：Longformer 在长文档分类、问答等任务上取得了优异表现，处理序列长度可达 4096~16384 token（在 2020 年已很长）。其思想直接影响了后续的 BigBird、LED（Longformer Encoder-Decoder）等模型。

### 3. Fedus, Zoph & Shazeer, 2021 — *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*

**简评**：通过极简的 Top-1 路由将 MoE 成功扩展到万亿参数规模，证明了"稀疏激活"是扩展模型容量的有效路径。

- **核心贡献**：（1）将 MoE 的路由策略简化为 **Top-1**——每个 token 只去一个专家，大幅降低通信和实现复杂度；（2）提出**容量因子**（capacity factor）和**辅助负载均衡损失**两个工程机制，解决 MoE 训练中的负载不均问题；（3）在 T5 架构上将 FFN 替换为 MoE，训练了最大 1.6 万亿参数的模型。
- **关键发现**：在相同的计算预算（FLOP）下，Switch Transformer 的预训练速度比 Dense T5 快 4-7 倍。这意味着 MoE 不仅是"更大"，而且在效率上也有巨大优势——用更少的计算获得更好的性能。
- **历史意义**：Switch Transformer 重新点燃了 MoE 的研究热潮。之前 MoE（如 Shazeer 2017）因训练不稳定和通信开销而未被广泛采用。Switch 的简化设计证明了 MoE 可以稳定、高效地扩展到极大规模，直接催生了后续的 Mixtral、DeepSeek-V2/V3 等模型。

### 4. Gu, Goel & Re, 2022 — *Efficiently Modeling Long Sequences with Structured State Spaces (S4)*

**简评**：首次让状态空间模型在长序列建模上取得突破性成果，开创了 SSM 这一全新的序列建模范式。

- **核心贡献**：（1）提出用**连续时间线性状态空间模型**（$h' = Ah + Bx$, $y = Ch$）替代注意力机制进行序列建模；（2）通过 **HiPPO 初始化**（一种基于正交多项式最优逼近理论设计的 $A$ 矩阵结构），解决了线性递归难以记忆长期依赖的问题；（3）利用线性递归可以展开为卷积的性质，训练时用 FFT 实现 $O(n \log n)$ 的并行计算，推理时用递归实现 $O(1)$ 每步。
- **关键突破**：在 Long Range Arena 基准测试上，S4 大幅超越了所有之前的高效 Transformer 变体（包括 Reformer、Performer、Longformer 等），尤其在需要建模极长依赖（序列长度 16K）的 Path-X 任务上，S4 是第一个超过随机基线的模型。
- **历史意义**：S4 证明了不依赖注意力机制也可以有效建模长序列，开辟了从 RNN → Transformer → SSM 的第三条技术路线，直接催生了 Mamba 等后续工作。

### 5. Gu & Dao, 2023 — *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*

**简评**：让 SSM 首次在语言建模上比肩同规模 Transformer，是 SSM 领域最具影响力的工作。

- **核心贡献**：（1）提出**选择性 SSM**——让 $B$、$C$、$\Delta$ 成为输入的函数（而非 S4 中的固定参数），使模型能根据输入内容动态决定记忆和遗忘；（2）设计了高效的**选择性扫描**（selective scan）GPU 算法，采用类似 FlashAttention 的 IO-aware 策略，在 SRAM 中完成递归计算；（3）将选择性 SSM 与门控 MLP 结合为 "Mamba Block"，无需注意力和 MLP 的分离设计。
- **关键性能**：Mamba-3B 在语言建模上超越了同规模的 Transformer++（配备了 FlashAttention、RoPE 等现代技巧的 Transformer），并匹配 2 倍规模的 Transformer。推理吞吐量是 Transformer 的 5 倍（序列长度越长优势越大）。
- **局限**：在需要精确检索序列中特定位置信息的任务（如 in-context learning 的精确复制、多跳推理）上，Mamba 仍不如 Transformer。这催生了 Mamba+Attention 混合架构的研究方向。

### 6. Munkhdalai, Faruqui & Gopal, 2024 — *Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention*

**简评**：在标准注意力中嵌入压缩记忆，让 Transformer 以有限的资源处理无限长的上下文。

- **核心贡献**：（1）在每个注意力层维护一个**压缩记忆矩阵** $M \in \mathbb{R}^{d_k \times d_v}$（大小固定，与序列长度无关），使用线性注意力风格的更新规则：$M_t = M_{t-1} + K_t^T V_t$；（2）将序列分段，段内做标准注意力，段间通过查询压缩记忆获取历史信息；（3）用可学习的门控参数 $\beta$ 在局部注意力和记忆检索之间做平衡。
- **核心思想**：将 Transformer 的"有限窗口、精确注意力"与"无限长度、近似记忆"统一在一个框架中。段内享受标准注意力的精度，段间通过压缩记忆保持信息传递，门控机制让模型自行学习两者的最佳比例。
- **实际效果**：在 1M token 长度的书籍摘要任务上，Infini-Attention 用远少于全注意力的计算量取得了可比的性能。但截至 2026 年初，该方法仍主要停留在研究阶段，大规模工程部署的案例有限。

### 7. Dao & Gu, 2024 — *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality (Mamba-2)*

**简评**：揭示了 SSM 与注意力之间的深层数学联系，统一了两大序列建模范式。

- **核心贡献**：（1）证明了选择性 SSM 等价于一种**半可分离矩阵**（semiseparable matrix）的结构化矩阵乘法，而标准注意力对应的矩阵在某些约束下也是半可分离的；（2）线性注意力实际上是 SSM 的一个特例；（3）利用这种对偶性（duality），设计了新的 SSD（State Space Duality）算法，在 GPU 上比 Mamba-1 快 2-8 倍。
- **重大意义**：Transformer 和 SSM 不再是截然对立的两个范式，而是同一数学框架下的不同特例。这为设计混合架构（如 Jamba、Zamba、Samba）提供了理论基础。

### 补充论文

**8. Chen, Lin et al., 2023** — *Extending Context Window of Large Language Models via Positional Interpolation*
**简评**：提出 Position Interpolation (PI)，通过线性缩放 RoPE 频率将 LLaMA 的上下文从 2K 扩展到 32K，仅需少量微调。是 RoPE 扩展技术的开创性工作，后续 NTK-Aware 和 YaRN 在此基础上改进。

**9. Jiang et al., 2024** — *Mixtral of Experts*
**简评**：8×7B MoE 架构，每次推理激活 2 个专家（12.9B 参数），总参数 46.7B。在多数基准上匹配或超越 LLaMA 2 70B，推理速度快 6 倍。首个被广泛采用的开源 MoE 大模型。

**10. Choromanski et al., 2020** — *Rethinking Attention with Performers*
**简评**：提出用随机正交特征（FAVOR+）近似 softmax 注意力，将复杂度降至 O(n·d)。虽然实际效果不如 FlashAttention，但线性注意力的核函数近似思路影响了后续研究。

**11. Peng et al., 2023** — *RWKV: Reinventing RNNs for the Transformer Era*
**简评**：结合 Transformer 的并行训练优势和 RNN 的 O(1) 推理优势，提出 RWKV 架构。通过 WKV 算子实现线性复杂度，是非 Transformer 架构的重要代表。

### 论文时间线

| 年份 | 论文 | 核心概念 | 复杂度 |
|------|------|----------|--------|
| 2020 | Reformer | LSH 注意力 + 可逆残差 | $O(n \log n)$ |
| 2020 | Longformer | 滑动窗口 + 全局注意力 | $O(n \cdot w)$ |
| 2020 | Performer | 随机特征线性注意力 | $O(n \cdot d)$ |
| 2021 | Switch Transformer | Top-1 MoE 路由 | $O(n \cdot d)$ per token |
| 2022 | S4 | 结构化状态空间模型 | $O(n \log n)$ 训练 / $O(1)$ 推理 |
| 2023 | Mamba | 选择性 SSM | $O(n \cdot d)$ |
| 2023 | Mixtral | 8 专家 Top-2 MoE | 46.7B 总参 / 12.9B 激活 |
| 2024 | Mamba-2 | SSM-Attention 对偶性 | $O(n \cdot d)$，更快常数 |
| 2024 | Infini-Attention | 压缩记忆 + 标准注意力 | $O(n \cdot s)$，$s$ 为段长 |
| 2024 | DeepSeek-V3 | 细粒度 MoE + 无辅助损失均衡 | 671B / 37B |

> **演化脉络**：两条主线并行发展——
> - **高效注意力线**：Reformer(2020) → Longformer(2020) → FlashAttention(2022) → Infini-Attention(2024)
> - **替代架构线**：S4(2022) → Mamba(2023) → Mamba-2(2024) → 混合架构(2024-2025)
> - **稀疏参数线**：MoE(2017) → Switch(2021) → Mixtral(2023) → DeepSeek-V3(2024)
>
> 三条线在 2024-2025 年开始融合：混合架构（Mamba+Attention）+ MoE 成为新趋势。

---

## 二、常见误区与FAQ

### FAQ 1：有了 128K 甚至 1M 的上下文窗口，RAG 是不是没用了？

**误区**："长上下文可以替代 RAG，把所有文档塞进上下文就行了。"

**澄清**：长上下文和 RAG 解决的是不同层面的问题，它们是互补关系。

**长上下文的局限**：

1. **成本**：128K token 的推理成本是 4K token 的约 32 倍（即使用 FlashAttention，计算量仍为 $O(n^2 \cdot d)$）。把所有文档都塞进上下文在经济上不可行。
2. **"大海捞针"退化**：研究表明，当上下文极长时，模型对中间位置信息的检索能力下降（"Lost in the Middle" 现象）。虽然 2025-2026 年的模型已大幅改善，但在多针测试中仍不完美。
3. **知识更新**：长上下文需要每次都重新输入全部文档；RAG 只需更新向量数据库中的相关条目。

**最佳实践**：RAG 负责从海量文档中**检索**最相关的片段（可能只有几百个 token），长上下文负责在这些片段之间**推理**和**综合**。两者结合的效果优于单独使用。

---

### FAQ 2：Mamba 会取代 Transformer 吗？

**误区**："Mamba 是 O(n) 的，Transformer 是 O(n^2) 的，所以 Mamba 一定会赢。"

**澄清**：截至 2026 年，答案是**不会完全取代，但会深度融合**。

**Mamba 的优势确实存在**：推理速度快（无 KV Cache 膨胀）、长序列计算高效、显存占用可控。

**但 Transformer 有 Mamba 无法完全复制的能力**：

1. **精确检索**：Transformer 的注意力可以让任意位置直接"查看"任意其他位置。Mamba 的信息必须通过有限维度的隐状态传递，存在信息瓶颈。
2. **In-context learning**：Transformer 通过注意力直接"模式匹配"上下文中的示例。Mamba 在这方面能力较弱。
3. **生态优势**：Transformer 有数年的工程优化积累（FlashAttention、KV Cache 优化、量化、分布式训练框架等），Mamba 的工程生态仍在建设中。

**2024-2025 的共识**：混合架构是最优方案——用 Mamba 层处理大部分序列建模（高效），在关键位置插入注意力层（精确检索）。Jamba、Zamba、Samba 等模型均验证了这一方向。

---

### FAQ 3：MoE 的总参数量那么大，推理时不是要把所有专家都加载到 GPU 上吗？

**误区**："MoE 模型虽然每个 token 只激活少数专家，但所有专家的参数都要在 GPU 显存中，所以实际显存需求并没有减少。"

**澄清**：这个说法**部分正确**——确实需要加载所有专家的参数。但有几个重要的补充：

1. **推理计算量确实减少了**：虽然显存占用大（需要放全部专家参数），但每个 token 的计算量只涉及 Top-K 个专家。在高吞吐场景（大 batch），计算量的减少直接转化为速度提升。

2. **专家并行分摊显存**：在多 GPU 部署时，不同 GPU 持有不同的专家。每个 GPU 的显存只需要放自己的那部分专家。

3. **专家卸载（Expert Offloading）**：对于显存有限的场景，可以将不常用的专家卸载到 CPU 内存或 SSD，需要时再加载。代价是增加延迟，但可以大幅降低 GPU 显存需求。

4. **实际对比**：Mixtral 8x7B（46.7B 总参数）在 2xA100-80G 上可以运行，推理速度接近 Mistral 7B（因为每 token 只激活 12.9B）。而一个 46.7B 的 Dense 模型不仅需要更多显存，推理也慢得多。

---

### FAQ 4：稀疏注意力（Longformer）在 LLM 时代还有用吗？

**误区**："有了 FlashAttention，直接用全注意力就行了，稀疏注意力过时了。"

**澄清**：在 **Decoder-only LLM**（如 GPT、LLaMA）场景中，Longformer/BigBird 风格的稀疏注意力确实不是主流选择。原因是：

1. FlashAttention 通过 IO 优化将全注意力的实际速度大幅提升，在 128K 以内的序列上性能已经很好
2. 稀疏注意力的掩码模式增加了实现复杂性，且在 GPU 上的实际加速不如理论值
3. LLM 的因果掩码已经是一种"半稀疏"模式

但在以下场景中，稀疏注意力仍有价值：
- **Encoder 模型**（BERT 家族）处理超长文档（如法律文件、学术论文）
- **特殊任务**：如基因组序列、蛋白质序列等超长序列的分析
- **滑动窗口注意力**在 LLM 中被 Mistral 成功采用（作为 KV Cache 的优化策略）

---

### FAQ 5：线性注意力为什么没有被广泛采用？

**误区**："线性注意力是 O(n) 的，应该比 O(n^2) 的标准注意力好得多。"

**澄清**：线性注意力（如 Performer）在理论上很优雅，但在实践中面临几个严重问题：

1. **近似质量**：softmax 注意力具有"赢者通吃"（winner-take-all）的尖锐分布特性，这对 Transformer 的表达能力至关重要。线性注意力的核函数近似很难准确复制这种特性。

2. **常数因子**：虽然理论复杂度从 $O(n^2 d)$ 降到了 $O(n d^2)$，但在 $n$ 不是特别大、$d$ 不是特别小的常见情况下，实际速度提升有限。配合 FlashAttention 的标准注意力可能更快。

3. **性能差距**：在语言建模等核心任务上，线性注意力的困惑度（perplexity）始终与标准注意力有可观的差距。

4. **SSM 的替代**：Mamba 等 SSM 模型以更优雅的方式实现了线性复杂度，并且性能远好于 Performer 等线性注意力方案。可以说 SSM 是"做对了的线性注意力"。

因此，线性注意力作为一个独立方案基本被放弃，但其核心思想（改变计算顺序、压缩记忆）被 SSM 和 Infini-Attention 等后续工作所继承和发展。

---

### FAQ 6：DeepSeek-V3 训练只花了 $5.6M，是真的吗？

**误区**："DeepSeek-V3 一定有大量未公开的预训练成本，$5.6M 不可能。"

**澄清**：$5.6M 是 DeepSeek 在技术报告中公开的**最终训练运行**的 GPU 租用成本（使用约 2048 张 H800，训练约 2 个月）。需要注意几点：

1. 这个数字**不包含**前期的架构搜索、超参调优、小规模实验、数据准备等成本
2. $5.6M 已经非常低，原因在于 MoE 架构的效率——671B 总参数但每 token 只激活 37B，计算效率远高于同规模 Dense 模型
3. 同期训练一个同性能水平的 Dense 模型（如 Llama-3-70B 级别），成本估计在数千万美元
4. MoE 的核心价值之一就是**训练效率**——Switch Transformer 论文就指出，MoE 在相同 FLOP 预算下比 Dense 模型收敛更快

这说明 MoE 不仅在推理时高效，在训练成本上也有巨大优势。

---

## 三、延伸资源

### 1. Lilian Weng — *The Transformer Family Version 2.0*

- **链接**：https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/
- **推荐理由**：OpenAI 研究员 Lilian Weng 对 Transformer 家族的全面综述，涵盖稀疏注意力、线性注意力、MoE、长文本扩展等所有主题。图文并茂，数学推导清晰，是本模块最佳的补充阅读材料。

### 2. Albert Gu & Tri Dao — *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* (原论文 + 官方代码)

- **论文**：https://arxiv.org/abs/2312.00752
- **代码**：https://github.com/state-spaces/mamba
- **推荐理由**：Mamba 论文的前半部分对 SSM 的动机和演进做了非常清晰的综述（从 RNN → S4 → Mamba），即使不深究数学细节也值得阅读。官方 PyTorch 实现可以直接运行实验。

### 3. DeepSeek-V3 技术报告

- **链接**：https://arxiv.org/abs/2412.19437
- **推荐理由**：截至 2025 年初最详细的 MoE 大模型技术报告之一。详细描述了细粒度 MoE 路由、无辅助损失负载均衡、多 Token 预测 (MTP) 等创新。对理解工业级 MoE 模型的设计选择非常有价值。

---

## 时效性说明

本文内容截至 **2026 年 4 月**。论文列表反映了截至此日期的重要工作。以下方面可能发生变化：

- 新的 SSM 变体或全新的序列建模范式可能出现
- MoE 的路由策略和负载均衡方法仍在快速迭代
- 混合架构（Mamba+Attention+MoE）的最优组合方式尚未确定
- 更高效的注意力实现（如 FlashAttention 的后续版本）可能改变各方案间的性价比对比

**建议**：定期查阅 arXiv 上 cs.CL 和 cs.LG 分类的最新论文，以及各大模型发布时的技术报告。
