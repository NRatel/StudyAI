# 论文与FAQ

> 本文件用于论文脉络、误区澄清与延伸阅读，不替代主章节正文。

---

## 关键论文

### 时间线总览

| 年份 | 作者 | 简称 | 核心贡献 |
|------|------|------|----------|
| 2013 | Kingma & Welling | VAE | 提出变分自编码器，将生成建模转化为变分推断问题 |
| 2014 | Goodfellow et al. | GAN | 提出生成对抗网络，开创对抗训练范式 |
| 2020 | Ho, Jain & Abbeel | DDPM | 将扩散概率模型工程化落地，证明可生成高质量图像 |
| 2020 | Song et al. | Score SDE / DDIM | 统一 score-based 与扩散模型，提出确定性采样加速 |
| 2021 | Dhariwal & Nichol | ADM / Classifier Guidance | 扩散模型首次在 FID 上超越 GAN |
| 2022 | Ho & Salimans | Classifier-Free Guidance | 去掉外部分类器，仅用单个条件/无条件模型实现引导 |
| 2022 | Rombach et al. | LDM / Stable Diffusion | 将扩散过程迁移到潜空间，大幅降低计算开销 |
| 2023 | Peebles & Xie | DiT | 用 Transformer 替换 U-Net 作为扩散模型骨干网络 |

---

### 1. VAE — 变分自编码器

- **作者/年份**：Diederik P. Kingma, Max Welling, 2013
- **标题**：*Auto-Encoding Variational Bayes*
- **一句话简评**：首次将深度学习与变分推断结合，通过重参数化技巧（reparameterization trick）使得含连续潜变量的生成模型可以端到端训练。
- **核心贡献**：
  - 提出 ELBO（Evidence Lower Bound）作为训练目标，将不可解的后验推断转化为优化问题。
  - 重参数化技巧让梯度可以穿过随机采样节点反向传播。
  - 建立了"编码器-潜空间-解码器"的生成模型基本范式，后续几乎所有潜变量生成模型都受其影响。

### 2. GAN — 生成对抗网络

- **作者/年份**：Ian J. Goodfellow et al., 2014
- **标题**：*Generative Adversarial Nets*
- **一句话简评**：通过生成器和判别器的博弈训练生成模型，无需显式建模数据分布，生成质量在当时远超其他方法。
- **核心贡献**：
  - 提出对抗训练框架：生成器试图骗过判别器，判别器试图区分真假样本。
  - 理论上证明在最优判别器下，生成器最小化的目标等价于 Jensen-Shannon 散度。
  - 开启了生成模型的"质量竞赛"时代，后续衍生出 DCGAN、WGAN、StyleGAN 等大量变体。

### 3. DDPM — 去噪扩散概率模型

- **作者/年份**：Jonathan Ho, Ajay Jain, Pieter Abbeel, 2020
- **标题**：*Denoising Diffusion Probabilistic Models*
- **一句话简评**：将 2015 年 Sohl-Dickstein 提出的扩散概率模型进行了工程化改进，证明扩散模型可以生成与 GAN 质量相当的图像。
- **核心贡献**：
  - 将训练目标简化为"预测每一步添加的噪声"，即简化的变分下界（simplified ELBO）。
  - 采用线性噪声调度（linear noise schedule）和 U-Net 架构。
  - 首次在无条件图像生成上展示了扩散模型的竞争力，为后续爆发奠定基础。

### 4. Score SDE / DDIM — 基于得分的生成建模与确定性采样

- **作者/年份**：Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, Ben Poole, 2020
- **标题**：*Score-Based Generative Modeling through Stochastic Differential Equations*（Score SDE）；Jiaming Song, Chenlin Meng, Stefano Ermon, 2020, *Denoising Diffusion Implicit Models*（DDIM）
- **一句话简评**：Score SDE 将离散扩散步骤推广到连续随机微分方程框架，DDIM 则提出确定性采样路径，使得采样步数可以大幅缩减。
- **核心贡献**：
  - Score SDE 统一了 SMLD（score matching with Langevin dynamics）和 DDPM 两大流派，证明它们都是同一 SDE 的特例。
  - DDIM 发现扩散模型的生成过程不必是随机的，通过非马尔可夫推断过程实现确定性映射，10-50 步即可出图。
  - 确定性映射使得潜空间插值和语义编辑成为可能。

### 5. ADM / Classifier Guidance — 带分类器引导的扩散模型

- **作者/年份**：Prafulla Dhariwal, Alexander Quinn Nichol, 2021
- **标题**：*Diffusion Models Beat GANs on Image Synthesis*
- **一句话简评**：通过精心的架构改进和分类器引导（classifier guidance），扩散模型在 FID 指标上首次全面超越 GAN。
- **核心贡献**：
  - 系统性地改进了 U-Net 架构（adaptive group normalization、更多注意力头、BigGAN 式上下采样等）。
  - 提出 classifier guidance：利用一个在加噪图像上训练的分类器的梯度来引导采样方向，提升条件生成质量。
  - 证明了扩散模型在多样性-保真度权衡上优于 GAN，终结了"GAN 生成质量不可超越"的认知。

### 6. Classifier-Free Guidance — 无分类器引导

- **作者/年份**：Jonathan Ho, Tim Salimans, 2022
- **标题**：*Classifier-Free Diffusion Guidance*
- **一句话简评**：去除了对外部分类器的依赖，通过在训练时随机丢弃条件信息（unconditional dropout），让同一个模型同时学习有条件和无条件生成。
- **核心贡献**：
  - 推断时将条件预测和无条件预测做线性外推：`output = unconditional + guidance_scale * (conditional - unconditional)`。
  - 更简洁的流程，不再需要额外训练分类器；且在实践中效果优于 classifier guidance。
  - 成为后续几乎所有条件扩散模型（Stable Diffusion、DALL-E 2、Imagen 等）的标准引导方式。

### 7. LDM / Stable Diffusion — 潜空间扩散模型

- **作者/年份**：Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Bjorn Ommer, 2022
- **标题**：*High-Resolution Image Synthesis with Latent Diffusion Models*
- **一句话简评**：将扩散过程从像素空间迁移到预训练自编码器的潜空间，在大幅降低计算量的同时保持甚至提升了生成质量。
- **核心贡献**：
  - 两阶段架构：先训练 VQ-VAE / KL-VAE 将图像压缩到低维潜空间（通常 4x 或 8x 空间下采样），再在潜空间上训练扩散模型。
  - 引入 cross-attention 机制对接文本编码器（CLIP text encoder），实现灵活的文本-图像条件生成。
  - 计算效率提升约 4-16 倍，使得高分辨率文生图在消费级 GPU 上成为可能，直接催生了 Stable Diffusion 开源生态。

### 8. DiT — 基于 Transformer 的扩散模型

- **作者/年份**：William Peebles, Saining Xie, 2023
- **标题**：*Scalable Diffusion Models with Transformers*
- **一句话简评**：用标准 Vision Transformer（ViT）替换 U-Net 作为扩散模型的去噪骨干网络，证明扩散模型同样遵循 Transformer 的 scaling law。
- **核心贡献**：
  - 将潜空间 patch 化后输入标准 Transformer，使用 adaLN-Zero（adaptive layer norm with zero-initialization）注入时间步和类别条件信息。
  - 实验证明 DiT 的 FID 随模型参数量和训练计算量平滑下降，具备良好的可扩展性（scalability）。
  - 为后续大规模扩散模型（如 Sora、SD3、FLUX）采用 Transformer 架构铺平了道路。

---

## 常见误区与FAQ

### Q1：扩散模型和 GAN 哪个更好？

**没有绝对的"更好"，取决于应用场景。**

| 维度 | 扩散模型 | GAN |
|------|----------|-----|
| 生成质量（FID） | 目前 SOTA，尤其在多样性上占优 | 高保真但易出现模式坍缩 |
| 训练稳定性 | 非常稳定，标准的去噪回归损失 | 对抗训练不稳定，需精心调参 |
| 采样速度 | 较慢（需迭代去噪，通常 20-50 步） | 极快（单次前向传播） |
| 模式覆盖 | 优秀，几乎不会模式坍缩 | 容易遗漏数据分布的部分模式 |
| 可控性 | 天然支持 guidance 和条件生成 | 需要额外设计条件注入机制 |

**总结**：如果追求生成多样性、训练稳定性和可控性，选扩散模型；如果对实时性要求极高（如游戏、视频实时渲染），GAN 或蒸馏后的扩散模型可能更合适。当前主流趋势是扩散模型占主导，GAN 在某些实时场景仍有一席之地。

### Q2：为什么预测噪声而不是直接预测图像？

这是 DDPM 论文中的关键设计选择，有几个重要原因：

1. **目标更简单、方差更低**：噪声是从标准正态分布中采样的，值域固定且均匀，模型学起来更容易收敛。而直接预测原始图像意味着需要从几乎纯噪声直接跳到复杂图像，回归目标的方差极大。

2. **数学等价但数值更优**：从变分下界推导出的损失函数中，预测噪声 epsilon 和预测原始图像 x_0 在数学上是等价的（可以互相线性变换），但实践中预测噪声时各时间步的损失量级更一致，不需要额外的加权系数。

3. **与 score function 的联系**：预测噪声本质上等价于估计数据分布的 score function（对数概率密度的梯度）。这建立了 DDPM 与 score-based generative model 的理论桥梁。

4. **实际效果更好**：Ho et al. 在论文中通过消融实验验证，预测噪声的简化目标比加权变分下界目标生成质量更高。

> 注意：后续研究中也出现了其他预测目标，如 v-prediction（预测速度向量）在某些场景下表现更优，说明"预测噪声"并非唯一正确答案，而是一个在多数情况下效果好且实现简单的选择。

### Q3：Stable Diffusion 为什么在潜空间而不是像素空间？

**核心原因是计算效率。**

- 一张 512x512 的 RGB 图像有 512 x 512 x 3 = 786,432 个维度。如果直接在像素空间做扩散，U-Net 每一步去噪都要处理这么大的张量，显存和计算量巨大。
- 通过预训练的自编码器（如 KL-VAE），图像被压缩到例如 64x64x4 = 16,384 维的潜空间，维度降低了约 **48 倍**。
- 自编码器承担了"学习像素级细节"的任务（高频信息），扩散模型只需要在语义丰富的低维空间中学习"全局结构和语义组成"。

**具体优势**：

| 对比项 | 像素空间扩散 | 潜空间扩散 |
|--------|-------------|-----------|
| 输入分辨率 | 512x512x3 | 64x64x4 |
| 计算量 | 极高 | 降低约 4-16 倍 |
| 训练设备要求 | 需要大量高端 GPU | 消费级 GPU 可训练 |
| 生成质量 | 高 | 相当甚至更好（低维空间更利于学习语义） |
| 推理速度 | 慢 | 快得多 |

**代价**：引入了自编码器的重建误差，极细粒度的细节可能在压缩过程中丢失。但实践表明，对于大多数应用来说这种损失可以忽略。

### Q4：Guidance scale 越大越好吗？

**不是。Guidance scale 存在最优区间，过大会严重损害生成质量。**

Classifier-free guidance 的公式为：

```
output = unconditional + s * (conditional - unconditional)
```

其中 `s` 就是 guidance scale（通常记为 `w` 或 `cfg_scale`）。

- **s = 1**：等价于不使用引导，模型完全按自身条件分布采样。生成多样性高但可能与条件不够匹配。
- **s = 7-8**（常见默认值）：在条件一致性（图文匹配度）和图像质量/多样性之间取得良好平衡。
- **s > 15-20**：图像开始出现过饱和、高对比度、细节失真等伪影。因为本质上是在做"外推"，把条件方向放大过头会脱离真实数据分布。
- **s 极大**（如 50+）：图像严重失真，颜色爆炸，结构崩坏。

**直觉理解**：guidance scale 相当于"对条件文本的强调程度"。适度强调让生成更贴合文本描述，过度强调则把图像推向了分布边缘的极端区域。

**实践建议**：
- 文生图一般用 7-12。
- 图生图或控制精细程度的场景可以适当降低到 3-7。
- 具体最优值因模型和任务而异，建议实验调节。

### Q5：扩散模型能生成文本吗？

**可以，但并非其最擅长的领域，且方法与图像生成有本质区别。**

1. **直接生成文本中的文字（图像层面）**：早期扩散模型在图像中渲染文字的质量很差（拼写错误、变形），因为扩散模型在连续像素空间操作，而文字本质上是离散符号。随着模型规模和训练数据增大（如 SDXL、FLUX），文字渲染能力已显著改善，但仍不如专门的字体渲染引擎精确。

2. **生成自然语言文本（文本序列）**：扩散模型也被探索用于生成离散文本序列，如：
   - **D3PM**（Austin et al., 2021）：定义了离散状态空间上的扩散过程。
   - **Diffusion-LM**（Li et al., 2022）：在连续词嵌入空间上做扩散，再映射回离散 token。
   - 但这类方法在文本生成质量上目前仍远不及自回归语言模型（GPT、LLaMA 等），因为自回归模型天然适合建模离散序列的条件概率。

3. **多模态生成**：一些前沿工作尝试在统一框架中同时生成图像和文本（如 Chameleon、Transfusion），通常对图像部分用扩散、对文本部分用自回归。

**总结**：扩散模型的核心优势在连续数据（图像、音频、视频、3D）的生成上。对于离散文本生成，自回归模型仍是更自然和更强的选择。

---

## 阅读建议

初次接触扩散模型，建议**先读 Ho et al. 2020（DDPM）**，这是最清晰的入门论文。它把扩散模型的训练目标简化为"预测噪声 + MSE 损失"，数学推导自洽且工程实现直观，是后续所有工作的基础。读完 DDPM 后，再按需阅读 DDIM（加速采样）→ Classifier-Free Guidance（条件生成）→ LDM（潜空间扩散）→ DiT（Transformer 骨干）的顺序，逐步构建完整知识体系。

---

## 延伸资源

1. **Lilian Weng 的博客 — "What are Diffusion Models?"**
   - 链接：https://lilianweng.github.io/posts/2021-07-11-diffusion-models/
   - 从 VAE、Flow 到 Score-based 和 DDPM 的完整数学推导，是理解扩散模型理论最推荐的入门资料之一。文风清晰，公式详尽。

2. **Calvin Luo — "Understanding Diffusion Models: A Unified Perspective" (arXiv: 2208.11970)**
   - 一篇综述性质的技术报告，从 ELBO 出发统一推导 VAE、层级 VAE、DDPM 和 Score SDE 之间的数学联系，适合希望深入理解不同生成模型之间关系的读者。

3. **Hugging Face Diffusion Models Course**
   - 链接：https://github.com/huggingface/diffusion-models-class
   - 配套代码的实战课程，覆盖从 DDPM 基础实现到 Stable Diffusion 微调的完整流程，适合动手实践。
