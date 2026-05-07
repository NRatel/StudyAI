# 论文与 FAQ

这个模块的核心论文是 Transformer 和注意力机制。读论文时先看结构图和动机，不必一开始追公式细节。

## 推荐了解的经典工作

| 主题 | 可以关注什么 |
|------|--------------|
| Attention in Seq2Seq | 解码器如何回看输入 |
| Attention Is All You Need | Transformer 为什么能替代 RNN 成为主流 |
| BERT / GPT | Encoder-only 和 Decoder-only 的不同路线 |
| RoPE | 位置信息如何进入注意力计算 |

## FAQ

### Attention 是不是等于 Transformer？

不是。注意力是机制，Transformer 是把注意力、前馈网络、残差、归一化等模块组合起来的架构。

### Q、K、V 必须按数据库类比理解吗？

类比有帮助，但别太死板。Q、K、V 本质是从 token 表示变换出来的向量，用来打分和汇总信息。

### Multi-Head 一定能学到不同语法关系吗？

不保证每个头都可解释，但多头确实给模型提供了多个表示子空间，有助于捕捉不同关系。

### Transformer 为什么适合并行？

因为一层里的所有 token 可以同时计算注意力，不像 RNN 必须等前一步算完。
