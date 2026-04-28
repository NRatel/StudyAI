# 论文与 FAQ

> 本文件用于论文脉络、误区澄清与延伸阅读，不替代主章节正文。

> 本文件汇总模块 07 的关键论文和常见问题。

---

## 关键论文

> **论文分三组理解**：
> 1. **分布式训练**：ZeRO (Rajbhandari 2020) → Megatron-LM (Shoeybi 2019) → 3D 并行 (Narayanan 2021)，解决"怎么把大模型分到多卡上训练"
> 2. **混合精度**：Mixed Precision Training (Micikevicius 2018)，解决"怎么用半精度省显存加速而不损失精度"
> 3. **对齐训练**：InstructGPT (Ouyang 2022)，解决"怎么让训出来的模型听话"
>
> 建议按这三组分别阅读，每组内部按发表时间顺序。

### 1. ZeRO: Memory Optimizations Toward Training Trillion Parameter Models

- **作者**: Rajbhandari, Rasley, Ruwase, He (Microsoft, 2020)
- **论文**: [arXiv:1910.02054](https://arxiv.org/abs/1910.02054)
- **核心贡献**:
  - 系统分析了数据并行中的显存冗余问题
  - 提出 ZeRO 三个阶段：分别切分优化器状态（Stage 1）、梯度（Stage 2）、模型参数（Stage 3）
  - 在不引入模型并行复杂度的前提下，将显存需求降低到 1/N（N 为 GPU 数）
  - 通信量分析：Stage 1/2 与 DDP 相同，Stage 3 多 50%
- **关键数字**: 使用 ZeRO-3 在 400 张 V100 上训练了 1000 亿参数模型
- **影响**: ZeRO 成为 DeepSpeed 的核心技术，是目前最广泛使用的大模型训练方案之一

```
ZeRO 的核心洞察:
  DDP 中每张卡存完整的 {参数, 梯度, 优化器状态} → 巨大冗余
  ZeRO 将这些状态分片到 N 张卡 → 需要时通信获取 → 用完就丢
  代价是多了通信 → 但可以与计算 overlap
```

**读后思考**: ZeRO 的成功说明，很多分布式训练的显存问题不需要改模型架构就能解决。核心是"用通信换显存"。

---

### 2. Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism

- **作者**: Shoeybi, Patwary, Puri, LeGresley, Casper, Catanzaro (NVIDIA, 2019)
- **论文**: [arXiv:1909.08053](https://arxiv.org/abs/1909.08053)
- **核心贡献**:
  - 提出了 Transformer 层内的高效张量并行方案
  - MLP 层：A 按列切分 + GeLU + B 按行切分 → 只需 1 次 AllReduce
  - 注意力层：按注意力头切分 → 天然适合并行
  - 在 512 张 V100 上训练了 83 亿参数的 GPT 模型
- **关键技巧**:
  - 利用 GeLU 的非线性性质巧妙安排切分方式
  - 每个 Transformer 层只需要 2 次 AllReduce（MLP 一次 + Attention 一次）
  - 通信发生在层内，要求高带宽（NVLink）

```
Megatron 张量并行的精妙之处:
  朴素做法: 每个矩阵乘法后都通信 → 通信次数多
  Megatron:  利用 GeLU 的性质，两个矩阵乘法之间不需要通信
             GeLU(X @ A_col) @ B_row → 一次 AllReduce
```

**读后思考**: 张量并行的核心限制是需要高带宽互连，所以通常限制在单节点（8 GPU）内。跨节点的并行用流水线或 ZeRO。

---

### 3. Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM

- **作者**: Narayanan, Shoeybi, Casper, LeGresley, Patwary, Puri, et al. (NVIDIA, 2021)
- **论文**: [arXiv:2104.04473](https://arxiv.org/abs/2104.04473)
- **核心贡献**:
  - 提出 **3D 并行** (PTD-P): 将数据并行、张量并行、流水线并行组合
  - 提出 **交错式流水线调度** (Interleaved 1F1B)：每个 GPU 持有多个非连续的层，进一步减少气泡
  - 系统分析了不同并行策略的通信模式和最优配置
  - 在 3072 张 A100 上训练了 1 万亿参数模型，达到 52% MFU

```
3D 并行的配置原则:
  1. 张量并行 (TP) → 放在节点内 (NVLink, 高带宽)
  2. 流水线并行 (PP) → 放在节点间 (通信量较小)
  3. 数据并行 (DP) → 跨所有副本

  TP × PP × DP = 总 GPU 数
```

- **交错式流水线**:

```
传统 1F1B: 每个 GPU 持有连续的层 (如 GPU 0 = 层 0-23)
交错式:    每个 GPU 持有多组非连续的层 (如 GPU 0 = 层 0-5 和层 24-29)
  优势: 流水线阶段数增加 → 微批次更多 → 气泡更少
  代价: 点对点通信量增加
```

**读后思考**: 3D 并行是当前训练超大模型的标准方案。理解这篇论文，就理解了"如何在千卡集群上高效训练"。

---

### 4. Mixed Precision Training

- **作者**: Micikevicius, Sharan, Gitman, Ber, Beirami, et al. (NVIDIA, Baidu, 2018)
- **论文**: [arXiv:1710.03740](https://arxiv.org/abs/1710.03740)
- **核心贡献**:
  - 系统提出混合精度训练方法：前向反向用 FP16，参数更新用 FP32
  - 提出 **Loss Scaling** 技术解决 FP16 梯度下溢问题
  - 提出 **FP32 主权重** (Master Weights) 保证更新精度
  - 在 CNN 和 RNN 上验证了混合精度训练不损失精度，速度提升 2-3x

```
混合精度三件套:
  1. FP32 Master Weights: 参数用 FP32 存储和更新
  2. FP16 计算: 前向和反向使用 FP16
  3. Loss Scaling: 放大 loss 防止梯度下溢

为什么有效:
  - 前向/反向中的数值精度要求不高 → FP16 够用
  - 参数更新中 lr × grad 可能很小 → 需要 FP32 精度
  - Tensor Core 对 FP16 有硬件加速 → 速度翻倍
```

**后续发展**:
- BF16（2020年+）: 保持 FP32 的指数范围，不需要 Loss Scaling
- FP8（2023年+）: H100 引入 FP8 Tensor Core，进一步提速
- 目前主流: BF16 混合精度（Llama-2/3、GPT-4 等都使用）

---

### 5. Training Language Models to Follow Instructions with Human Feedback (InstructGPT)

- **作者**: Ouyang, Wu, Jiang, Almeida, Wainwright, Mishkin, et al. (OpenAI, 2022)
- **论文**: [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
- **核心贡献**:
  - 提出完整的 RLHF 三阶段流程：SFT → RM 训练 → PPO 优化
  - 证明了 1.3B InstructGPT (RLHF) > 175B GPT-3 (纯预训练)
  - 系统研究了人类偏好数据的收集和标注方法
  - 分析了对齐训练对不同能力的影响（对齐税 Alignment Tax）

```
InstructGPT 的关键发现:
  1. 小模型 + 对齐 >> 大模型不对齐
     1.3B InstructGPT 在人类评估中优于 175B GPT-3

  2. RLHF 数据量不需要很多
     SFT: ~13K 条  |  RM: ~33K 条  |  PPO: ~31K prompts

  3. 对齐有代价 (Alignment Tax)
     在某些学术基准上，InstructGPT 比 GPT-3 略差
     但在人类偏好和安全性上大幅提升

  4. 数据质量极其重要
     标注者培训、质量控制、一致性检查 都很关键
```

**读后思考**: InstructGPT 论文开创了"对齐训练"的范式，此后几乎所有商用 LLM（GPT-4、Claude、Gemini）都采用了类似的流程。

---

### 6. 补充论文

| 论文 | 年份 | 核心贡献 |
|------|------|---------|
| **Chinchilla** (Hoffmann et al., 2022) | 2022 | 计算最优的数据/参数比例（约 20:1） |
| **DPO** (Rafailov et al., 2023) | 2023 | 直接用偏好数据优化，不需要 RM 和 RL |
| **LIMA** (Zhou et al., 2023) | 2023 | 仅 1000 条高质量数据就能有效对齐 |
| **FSDP** (Zhao et al., 2023) | 2023 | PyTorch 原生的 ZeRO-3 实现 |
| **Llama-2 技术报告** (Touvron et al., 2023) | 2023 | 完整的预训练+对齐训练工程细节 |
| **Llama-3 技术报告** (Meta, 2024) | 2024 | 15T tokens 训练、128K 词表、数据工程 |

---

## 常见问题 (FAQ)

### Q1: ZeRO-3 和张量并行有什么区别？什么时候用哪个？

**ZeRO-3** 和**张量并行** (TP) 都能让"放不下单卡的模型"训练起来，但机制完全不同：

| 特性 | ZeRO-3 | 张量并行 |
|------|--------|---------|
| 切什么 | 参数/梯度/优化器的存储 | 矩阵计算本身 |
| 通信内容 | 参数 (All-Gather) | 激活值 (AllReduce) |
| 通信量 | 与模型大小成正比 | 与激活值大小成正比 |
| 对带宽要求 | 中 | **高**（层内通信） |
| 代码修改 | 几乎不改 | 需要改模型代码 |
| 适用场景 | 通用 | 节点内 NVLink 互连 |

**选型建议**：
- **资源有限、快速上手** → ZeRO-3（DeepSpeed 一行配置）
- **大规模训练、追求效率** → 张量并行（配合 Megatron-LM）
- **超大模型（100B+）** → 3D 并行（TP + PP + ZeRO-DP）

---

### Q2: BF16 和 FP16 到底选哪个？

**简单结论**：如果你的 GPU 支持 BF16（A100、H100、RTX 3090+），**选 BF16**。

原因：
1. BF16 的指数位和 FP32 相同（8位），范围足够大，**不需要 Loss Scaling**
2. FP16 虽然精度更高（10位尾数），但范围太小（最大 65504），容易溢出
3. BF16 训练更稳定，几乎是现代大模型训练的标准配置

唯一选 FP16 的场景：V100 等不支持 BF16 的老卡。

---

### Q3: 梯度累积会影响训练效果吗？

**数学上完全等价**——梯度累积 8 步、每步 batch=4，等价于 batch=32 训一步。

但有三个实践注意点：
1. **BatchNorm 层**：如果使用 BatchNorm（Transformer 通常不用），统计量是基于 micro-batch 计算的，可能与大 batch 不同。LayerNorm 没有这个问题。
2. **学习率调度**：step 数变少了（累积 8 步才算 1 step），warmup 步数和总步数都要相应调整。
3. **Loss 缩放**：记得 `loss = loss / accumulation_steps`，否则梯度会被放大。

---

### Q4: DPO 效果真的能比 RLHF 好吗？

**目前的共识**：
- DPO 在离线偏好数据上的效果可以接近甚至匹配 RLHF（PPO）
- 但 RLHF 有在线探索的优势——模型可以生成新回复、获取新反馈
- 实践中，**迭代式 DPO**（用当前模型生成 → 收集偏好 → 再训练）可以缩小差距

**Llama-2 的经验**：
- 先 RLHF (PPO) 训练多轮
- 后来发现 Rejection Sampling + DPO 的效果也很好
- 最终版本结合了多种方法

**建议**：大多数场景先用 DPO，简单有效；如果有足够资源和经验，再尝试 PPO。

---

### Q5: 预训练数据的 tokens 数量应该是参数量的多少倍？

**Chinchilla 法则**（2022）：计算最优的 tokens 数约为参数量的 **20 倍**。

```
模型参数量    Chinchilla 推荐 tokens    实际趋势 (2024+)
1B            20B                        ~100B+ (过度训练)
7B            140B                       ~2T+
70B           1.4T                       ~15T
```

但实际中，2023 年之后的趋势是**过度训练**（Over-training）：
- 推理成本远大于训练成本
- 训练更小但"吃更多数据"的模型，推理更便宜
- Llama-3 8B 用了 15T tokens（比 Chinchilla 推荐多 ~100 倍！）

**结论**：Chinchilla 法则是"计算最优"，但如果关注推理效率，可以用更多数据训更小的模型。

---

### Q6: 激活检查点的 33% 额外计算开销值得吗？

**几乎总是值得**。

```
不用激活检查点:
  - 显存: 参数 + 梯度 + 优化器 + 全部激活值
  - 激活值可能比参数还大! (与 batch_size × seq_len × n_layers 成正比)

用激活检查点:
  - 显存: 参数 + 梯度 + 优化器 + 少量激活值
  - 计算: 增加 ~33% (每个 block 的前向算两遍)

实际影响:
  - 省下来的显存可以用更大 batch → 训练更快
  - 或者可以训练更大的模型
  - 33% 的额外计算通常被更大 batch 的吞吐量收益抵消
```

所以**几乎所有大模型训练都开启激活检查点**。

---

### Q7: SFT 数据量多少合适？更多数据一定更好吗？

**不一定！** LIMA 论文证明 1000 条精选数据就能达到很好的效果。

```
经验法则:
  < 1K:    如果质量极高（人工精心编写），可以有效
  1K-10K:  大多数场景的推荐范围
  10K-100K: 覆盖更多任务类型
  > 100K:  可能开始过拟合，除非数据多样性足够

质量 vs 数量:
  1000 条人工精写 > 100K 条 GPT 生成
  关键不是数量，而是:
  1. 多样性 (覆盖不同任务类型)
  2. 准确性 (事实正确)
  3. 格式规范 (一致的回复风格)
```

---

### Q8: 单机 8 卡训练 7B 模型，ZeRO 应该选 Stage 几？

```
7B 模型的显存需求:
  FP16 参数:      14 GB
  FP16 梯度:      14 GB
  FP32 优化器:    84 GB (Adam: 4+4+4 bytes × 7B)
  ──────────────
  总计:           112 GB

8× A100-80GB 方案:
  DDP:    每卡 112 GB → 超！
  ZeRO-1: 参数 14 + 梯度 14 + 优化器 84/8 = 38.5 GB → 可行!
  ZeRO-2: 参数 14 + 梯度 14/8 + 优化器 84/8 = 26.25 GB → 更宽裕
  ZeRO-3: (14+14+84)/8 = 14 GB → 最省

推荐:
  训练 (大 batch): ZeRO-2 (够用且通信少)
  训练 (小 batch): ZeRO-2
  微调 (LoRA):    ZeRO-1 甚至 DDP (参数量小)
  显存不够时:     ZeRO-3 + CPU Offload
```

**通用建议**：从 ZeRO-2 开始尝试，如果 OOM 再升到 ZeRO-3。

---

### Q9: 训练中出现 loss spike（损失突刺）怎么办？

这是大模型训练中的常见问题，尤其在 100B+ 规模。

```
轻微 spike (loss 升高 2-3x，几十步后恢复):
  → 继续训练，通常自动恢复
  → 可能是某个 batch 的数据质量差

中度 spike (loss 升高 5-10x，不恢复):
  → 从最近的 checkpoint 重启
  → 降低学习率 (如减半)
  → 跳过导致 spike 的数据 batch

严重 spike (loss 发散):
  → 停止训练
  → 检查数据质量
  → 降低学习率
  → 增加 warmup 步数
  → 检查梯度裁剪是否生效

预防措施:
  1. 每 100-1000 步保存 checkpoint
  2. 监控梯度范数 (正常范围 0.1-10)
  3. 使用梯度裁剪 (max_norm=1.0)
  4. 充分的 warmup (1000-2000 步)
  5. 数据充分去重和清洗
```

---

### Q10: 如何估算训练所需的 GPU 时间？


---

## 延伸资源

### 开源工具

| 工具 | 用途 | 链接 |
|------|------|------|
| DeepSpeed | 分布式训练框架 | github.com/microsoft/DeepSpeed |
| Megatron-LM | 张量/流水线并行 | github.com/NVIDIA/Megatron-LM |
| trl | SFT/RLHF/DPO 训练 | github.com/huggingface/trl |
| tokenizers | 高速 Tokenizer 库 | github.com/huggingface/tokenizers |
| SentencePiece | Google Tokenizer | github.com/google/sentencepiece |
| datatrove | 数据处理流水线 | github.com/huggingface/datatrove |
| text-dedup | 文本去重工具 | github.com/ChenghaoMou/text-dedup |

### 推荐阅读顺序

```
入门:
  1. Mixed Precision Training (Micikevicius 2018) — 理解为什么用半精度
  2. ZeRO (Rajbhandari 2020) — 理解显存优化
  3. InstructGPT (Ouyang 2022) — 理解对齐训练

进阶:
  4. Megatron-LM (Shoeybi 2019) — 理解张量并行
  5. 3D 并行 (Narayanan 2021) — 理解完整并行策略
  6. DPO (Rafailov 2023) — 理解偏好优化

工程:
  7. Llama-2 技术报告 — 完整的训练工程细节
  8. Llama-3 技术报告 — 最新的数据实际应用
```
