# CNN 卷积神经网络

## 直觉与概述

### 核心问题：全连接网络处理图像有什么问题？

你已经学了全连接网络（前一模块）。假设要处理一张 $224 \times 224$ 的彩色图像（3 通道），输入向量的维度是：

$$224 \times 224 \times 3 = 150{,}528$$

如果第一个隐藏层有 1024 个神经元，仅这一层的权重矩阵就有 $150{,}528 \times 1024 \approx 1.54$ 亿个参数。这带来三个致命问题：

1. **参数爆炸**：一亿多参数的第一层，不仅训练慢，还极易过拟合。ImageNet 训练集也才 120 万张图——参数比数据还多。
2. **丢失空间结构**：把图像展平成一维向量后，像素 $(0, 0)$ 和像素 $(0, 1)$ 的"相邻"关系完全丢失了。全连接网络看到的只是一个 15 万维的数字列表，它不知道哪些像素在空间上靠近。
3. **无平移不变性**：如果一只猫从图像左上角移到右下角，全连接网络需要在不同的权重位置**重新学习**"猫"的概念——因为左上角的像素和右下角的像素连接的是完全不同的权重。

### 人类视觉的启示

想象你在看一张照片。你不会同时关注所有像素——你的视觉系统先在**局部区域**检测边缘和纹理，然后逐步组合成更大的模式（角、轮廓），再组合成物体部件（眼睛、耳朵），最终识别出完整的物体（猫）。

```
像素层        →    边缘/纹理    →    部件        →    物体
[][][][][]         / | \ ─         眼 耳 鼻        [猫]
[][][][][]         检测方向、      在局部区域       高层语义
[][][][][]         颜色渐变        组合边缘         组合部件
```

CNN 模仿的正是这个层级化、局部化的处理过程。

### CNN 的三个核心思想

**1. 局部连接（Local Connectivity）**

每个神经元只看输入的一小块区域（称为**感受野**），而不是整张图。一个 $3 \times 3$ 的卷积核只连接 $3 \times 3 = 9$ 个输入像素，而不是全部 $150{,}528$ 个。

**2. 权值共享（Weight Sharing）**

同一个卷积核在整张图上**滑动**使用，所有位置共享同一组权重。这意味着：
- 参数量从"输入大小 $\times$ 输出大小"降到"卷积核大小"——一个 $3 \times 3$ 的核只有 9 个参数
- 在左上角学到的"边缘检测"能力，自动适用于右下角

**3. 平移不变性（Translation Invariance）**

由于权值共享，无论猫出现在图像的什么位置，同一个卷积核都能检测到它的特征。CNN 天然具备平移不变性。

### 卷积核 = 滑动探测器

把卷积核想象成一个小型"探测器"，它在输入图像上从左到右、从上到下滑动。每到一个位置，就与覆盖的局部区域做一次"匹配计算"（逐元素相乘再求和）。如果这个区域恰好包含卷积核要找的模式（比如一条竖直边缘），匹配得分就高；否则得分低。

```
输入图像 (5x5)              卷积核 (3x3)             输出特征图 (3x3)
┌─────────────────┐        ┌───────────┐            ┌───────────┐
│  1  0  1  0  1  │        │  1  0  1  │            │  ?  ?  ?  │
│  0  1  0  1  0  │   *    │  0  1  0  │     =      │  ?  ?  ?  │
│  1  0  1  0  1  │        │  1  0  1  │            │  ?  ?  ?  │
│  0  1  0  1  0  │        └───────────┘            └───────────┘
│  1  0  1  0  1  │
└─────────────────┘
     卷积核在图像上滑动，每个位置输出一个标量
```

### 一句话总结

> **CNN = 局部连接 + 权值共享 + 层级特征提取。** 卷积核作为滑动探测器在输入上扫描，低层检测边缘/纹理，高层组合出语义特征，参数量极少却能捕获空间结构。

---

## 严谨定义与原理

### 卷积运算（实际上是互相关）

#### 数学定义

在数学中，**卷积**（convolution）的定义是：

$$(f * g)(t) = \int_{-\infty}^{\infty} f(\tau) \cdot g(t - \tau) \, d\tau$$

注意其中 $g(t - \tau)$ 有一个**翻转**操作。但在深度学习中，我们使用的其实是**互相关**（cross-correlation），不做翻转：

$$(f \star g)(t) = \int_{-\infty}^{\infty} f(\tau) \cdot g(t + \tau) \, d\tau$$

由于卷积核的权重是学出来的，翻不翻转对最终效果没有区别（翻转后的最优权重 = 不翻转的最优权重的镜像），所以深度学习社区直接把互相关叫做"卷积"。本文也遵循这个惯例。

> **直觉补充——参数共享为什么天然适合图像？** 同一个边缘检测器（卷积核）在图像的任何位置都有用：左上角的竖直边缘和右下角的竖直边缘本质上是同一种模式，不需要用不同的参数分别学习。这正是权值共享的物理合理性所在。

#### 2D 卷积的计算过程

给定输入矩阵 $\mathbf{X}$（大小 $W \times W$）和卷积核 $\mathbf{K}$（大小 $K \times K$），2D 卷积的输出 $\mathbf{Y}$ 的第 $(i, j)$ 个元素为：

$$Y_{i,j} = \sum_{m=0}^{K-1} \sum_{n=0}^{K-1} X_{i+m, \, j+n} \cdot K_{m,n} + b$$

其中 $b$ 是偏置项。直觉：在位置 $(i, j)$，把卷积核覆盖的区域与核做逐元素乘积，然后求和。

**具体数值示例**：

```
输入 X (4x4):                卷积核 K (3x3):
┌─────────────────┐          ┌───────────┐
│  1   2   3   0  │          │  0   1   2 │
│  0   1   2   3  │          │  2   2   0 │
│  3   0   1   2  │          │  0   1   2 │
│  2   3   0   1  │          └───────────┘
└─────────────────┘          偏置 b = 0

位置 (0,0):
Y[0,0] = 1×0 + 2×1 + 3×2 + 0×2 + 1×2 + 2×0 + 3×0 + 0×1 + 1×2 = 12

  ┌───────────┐
  │ [1  2  3] │ 0         对应核:  0  1  2
  │ [0  1  2] │ 3                  2  2  0
  │ [3  0  1] │ 2                  0  1  2
  │  2  3  0  │ 1
  └───────────┘
逐元素相乘再求和 = 0+2+6+0+2+0+0+0+2 = 12

位置 (0,1):
Y[0,1] = 2×0 + 3×1 + 0×2 + 1×2 + 2×2 + 3×0 + 0×0 + 1×1 + 2×2 = 14

输出 Y (2x2):
┌───────┐
│ 12 14 │
│ 14 12 │
└───────┘
```

### 关键参数与输出尺寸公式

#### 三个核心参数

| 参数 | 符号 | 含义 |
|------|:----:|------|
| **卷积核大小** | $K$ | 卷积核的宽和高（常见：$3 \times 3$、$5 \times 5$） |
| **步长** | $S$ | 卷积核每次滑动的像素数（stride） |
| **填充** | $P$ | 在输入边缘补零的圈数（padding） |

#### 输出尺寸公式

给定输入大小 $W$、卷积核大小 $K$、步长 $S$、填充 $P$，输出特征图的大小为：

$$O = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1$$

**常见场景验证**：

| 输入 $W$ | 核 $K$ | 步长 $S$ | 填充 $P$ | 输出 $O$ | 说明 |
|:---:|:---:|:---:|:---:|:---:|------|
| 5 | 3 | 1 | 0 | 3 | 无填充，尺寸缩小 |
| 5 | 3 | 1 | 1 | 5 | "same" 填充，输出与输入同大小 |
| 5 | 3 | 2 | 0 | 2 | 步长 2，空间下采样 |
| 32 | 5 | 1 | 2 | 32 | $\lfloor(32-5+4)/1\rfloor+1 = 32$（same padding for $5 \times 5$）|
| 224 | 7 | 2 | 3 | 112 | AlexNet/ResNet 第一层 |

> **"same" 填充的规律**：当 $S = 1$ 时，设 $P = \lfloor K/2 \rfloor$（例如 $K=3 \Rightarrow P=1$，$K=5 \Rightarrow P=2$），输出大小等于输入大小。

### 多通道卷积

#### 多输入通道

实际图像有 3 个通道（RGB）。多通道卷积的规则是：**卷积核的通道数必须等于输入的通道数**。

假设输入有 $C_{in}$ 个通道，卷积核的形状就是 $C_{in} \times K \times K$。计算时，对每个通道分别做 2D 卷积，然后将结果**逐元素相加**，得到一个单通道的输出：

$$Y_{i,j} = \sum_{c=0}^{C_{in}-1} \sum_{m=0}^{K-1} \sum_{n=0}^{K-1} X_{c, \, i+m, \, j+n} \cdot K_{c, \, m, \, n} + b$$

```
输入 (3通道, 4x4):     一个卷积核 (3通道, 3x3):     输出 (1通道, 2x2):
  R通道 [4x4]             核R [3x3]
  G通道 [4x4]     *       核G [3x3]          =      [2x2]
  B通道 [4x4]             核B [3x3]

每个通道单独卷积 → 三个 2x2 结果 → 逐元素相加 → 一个 2x2 输出
```

#### 多输出通道（多个卷积核）

一个卷积核只能检测一种模式（比如"竖直边缘"）。要检测多种模式，就用多个卷积核。如果有 $C_{out}$ 个卷积核，就输出 $C_{out}$ 个特征图（feature maps）。

卷积层的完整参数形状：

$$\text{权重: } C_{out} \times C_{in} \times K \times K \qquad \text{偏置: } C_{out}$$

**参数量计算示例**：输入 3 通道，输出 64 通道，$3 \times 3$ 核：
$$\text{参数量} = 64 \times 3 \times 3 \times 3 + 64 = 1{,}792$$

对比全连接层处理同样的 $224 \times 224 \times 3$ 输入到 64 维输出：
$$\text{参数量} = 150{,}528 \times 64 + 64 = 9{,}633{,}856$$

卷积层的参数量只有全连接层的 $0.019\%$。

### 特征图（Feature Map）

卷积层的输出叫做**特征图**。每个特征图对应一个卷积核检测到的模式。直觉上：

- 第 1 层特征图：各种方向的边缘、颜色斑块
- 第 2-3 层特征图：角、纹理、简单形状
- 第 4-5 层特征图：物体部件（眼睛、轮子、窗户）
- 更深层特征图：整个物体、场景

```
输入图像 → [卷积层1: 64个核] → 64张特征图 → [卷积层2: 128个核] → 128张特征图 → ...
             检测边缘/纹理           组合边缘为部件              组合部件为物体
```

随着网络深度增加，特征图的**空间尺寸**逐渐缩小（通过步长或池化下采样），而**通道数**逐渐增大（更多种类的高级特征）。这是 CNN 的经典设计模式。

### 池化层（Pooling）

池化层的目的是**降低空间分辨率**，减少计算量，同时增强特征的平移不变性。

#### 最大池化（Max Pooling）

在每个池化窗口中取最大值：

```
输入 (4x4):              Max Pool (2x2, stride=2):     输出 (2x2):
┌─────────────────┐                                    ┌───────┐
│ [1  3] [2  4]   │      每个 2x2 窗口取最大值          │  6  8 │
│ [5  6] [7  8]   │      ──────────────────→           │  3  4 │
│ [3  2] [1  0]   │                                    └───────┘
│ [1  2] [3  4]   │
└─────────────────┘

窗口 [1,3,5,6] → max = 6
窗口 [2,4,7,8] → max = 8
窗口 [3,2,1,2] → max = 3
窗口 [1,0,3,4] → max = 4
```

**最大池化的含义**：只保留每个区域中最显著的特征激活。如果一个卷积核检测到了"竖直边缘"，最大池化保留的是该区域中"最像竖直边缘"的那个位置的响应——不管边缘具体在窗口内的哪个像素上。这就是为什么池化能提供额外的平移不变性。

#### 平均池化（Average Pooling）

在每个池化窗口中取平均值。平均池化更"温和"，保留了区域的整体激活水平，而不仅仅是最强响应。

常见用法：在网络最后一层使用**全局平均池化**（Global Average Pooling），把 $H \times W$ 的特征图直接压缩为 $1 \times 1$，替代全连接层。

```
特征图 (7x7x512)  ──全局平均池化──→  向量 (512,)  ──全连接──→  类别概率
```

#### 池化层不含可学习参数

池化是一个固定的操作（取最大值或平均值），没有权重需要学习。池化层的输出尺寸公式和卷积相同：

$$O = \left\lfloor \frac{W - K}{S} \right\rfloor + 1$$

（池化通常不使用填充，且 $K = S$。最常见的配置是 $K = 2, S = 2$，将空间尺寸减半。）

### 经典模型

#### LeNet-5 (1998) -- CNN 的奠基之作

**核心创新**：首次成功将卷积 + 池化 + 全连接的流水线应用于实际任务（手写数字识别）。证明了端到端训练的卷积网络可以自动学习特征，无需手工设计。

**架构**：

```
输入 (1x32x32)
  → Conv1 (6个 5x5核, stride=1)   → 6x28x28   → AvgPool (2x2, stride=2) → 6x14x14
  → Conv2 (16个 5x5核, stride=1)  → 16x10x10  → AvgPool (2x2, stride=2) → 16x5x5
  → Flatten → 400
  → FC1 (400→120) → FC2 (120→84) → FC3 (84→10)
```

放在今天来看，LeNet-5 非常简单：只有 6 万参数，两个卷积层。但它的设计范式——**卷积提取特征 → 池化降维 → 全连接分类**——直到 2012 年的 AlexNet 还在沿用。

#### AlexNet (2012) -- 深度学习的引爆点

**核心创新**：
- **首次在 ImageNet 上用 GPU 训练大规模 CNN**，将 top-5 错误率从 26% 降到 16%，碾压传统方法
- 引入 ReLU 激活函数（替代 sigmoid/tanh，训练速度提升 6 倍）
- 使用 Dropout 正则化（全连接层的 dropout=0.5）
- 数据增强（随机裁剪、水平翻转、颜色抖动）

**架构**（简化版）：

```
输入 (3x227x227)
  → Conv1 (96个 11x11核, stride=4, padding=0)  → 96x55x55  → MaxPool → 96x27x27
  → Conv2 (256个 5x5核, padding=2)              → 256x27x27 → MaxPool → 256x13x13
  → Conv3 (384个 3x3核, padding=1)              → 384x13x13
  → Conv4 (384个 3x3核, padding=1)              → 384x13x13
  → Conv5 (256个 3x3核, padding=1)              → 256x13x13 → MaxPool → 256x6x6
  → Flatten → 9216
  → FC1 (9216→4096, dropout=0.5)
  → FC2 (4096→4096, dropout=0.5)
  → FC3 (4096→1000)
```

AlexNet 的历史意义远超技术本身——它证明了深度学习 + GPU + 大数据的组合是可行的，直接引发了深度学习的爆发式发展。

#### VGGNet (2014) -- "更深更好"

**核心创新**：用大量 $3 \times 3$ 小卷积核堆叠替代大卷积核，证明**网络深度**是提升性能的关键因素。

**为什么 $3 \times 3$ 比 $5 \times 5$ 好？** 两个 $3 \times 3$ 卷积层叠加的感受野等于一个 $5 \times 5$，三个 $3 \times 3$ 等于一个 $7 \times 7$。但参数更少、非线性更强：

| 方案 | 感受野 | 参数量（$C$ 通道） | 非线性层数 |
|------|:------:|:------------------:|:----------:|
| 一个 $5 \times 5$ 核 | $5 \times 5$ | $25C^2$ | 1 |
| 两个 $3 \times 3$ 核 | $5 \times 5$ | $2 \times 9C^2 = 18C^2$ | 2 |
| 一个 $7 \times 7$ 核 | $7 \times 7$ | $49C^2$ | 1 |
| 三个 $3 \times 3$ 核 | $7 \times 7$ | $3 \times 9C^2 = 27C^2$ | 3 |

VGG-16 有 16 个权重层（13 个卷积 + 3 个全连接），约 1.38 亿参数。它的架构极其规整——通道数从 64 翻倍到 128、256、512，空间尺寸每次减半。这种"通道翻倍、尺寸减半"的模式后来被广泛采用。

#### ResNet (2015) -- 残差连接，开启超深网络时代

**核心问题**：网络越深越好吗？实验发现，当网络从 20 层加深到 56 层时，**训练误差**反而上升了——这不是过拟合（过拟合只影响测试误差），而是**优化困难**。直觉上，56 层网络至少应该不比 20 层差（多出来的 36 层可以学成恒等映射），但实际上优化器找不到这个解。

**残差连接（Skip Connection）的核心思想**：

与其让网络学习目标映射 $H(x)$，不如让它学习**残差** $F(x) = H(x) - x$。原始映射变成 $H(x) = F(x) + x$。

$$\text{输出} = F(x) + x$$

```
        ┌──────────────────────────┐
        │                          │  恒等捷径（Identity Shortcut）
        │                          │
x ──→ [Conv] → [BN] → [ReLU] → [Conv] → [BN] → (+) → [ReLU] → 输出
        │                                         ↑
        └─────────────────────────────────────────┘
                      x 直接跳过两层加到输出上

如果最优解是恒等映射 H(x) = x，网络只需让 F(x) = 0
把权重推向零比学习恒等映射容易得多！
```

**为什么残差连接有效？从梯度角度理解**：

在反向传播中，残差连接提供了**梯度直通路径**。设 $y = F(x) + x$，那么：

$$\frac{\partial y}{\partial x} = \frac{\partial F}{\partial x} + \mathbf{I}$$

其中 $\mathbf{I}$ 是单位矩阵。即使 $\frac{\partial F}{\partial x}$ 很小（梯度消失），梯度至少还有 $\mathbf{I}$ 这条路径可以传播。这就是为什么 ResNet 能训练到 152 层甚至 1000 层的原因。

对比 VGG（无残差连接），梯度需要穿过每一层的 $\frac{\partial F}{\partial x}$，这些连乘很容易衰减到接近零：

$$\frac{\partial \text{loss}}{\partial x_0} = \prod_{l=1}^{L} \frac{\partial F_l}{\partial x_{l-1}} \quad \text{(VGG: 连乘，容易消失)}$$

$$\frac{\partial \text{loss}}{\partial x_0} = \prod_{l=1}^{L} \left(\frac{\partial F_l}{\partial x_{l-1}} + \mathbf{I}\right) \quad \text{(ResNet: 每项含 I，不会归零)}$$

**ResNet 的架构要点**：

| 变体 | 层数 | 参数量 | Top-5 错误率 (ImageNet) |
|------|:----:|:------:|:----------------------:|
| ResNet-18 | 18 | 11.7M | 10.9% |
| ResNet-34 | 34 | 21.8M | 9.5% |
| ResNet-50 | 50 | 25.6M | 7.7% |
| ResNet-101 | 101 | 44.5M | 7.1% |
| ResNet-152 | 152 | 60.2M | 6.7% |

注意 ResNet-50 只有 2560 万参数（VGG-16 有 1.38 亿），但效果好得多。这得益于残差连接和**Bottleneck 结构**（$1 \times 1$ → $3 \times 3$ → $1 \times 1$ 的瓶颈块，下一节会讲）。

**残差连接的深远影响**：残差连接不仅限于 CNN，它成为了深度学习的**通用设计模式**。Transformer 的每个子层也使用了残差连接：$\text{output} = \text{SubLayer}(x) + x$。可以说，没有残差连接，就没有 GPT。

### 1x1 卷积

$1 \times 1$ 的卷积核看起来很奇怪——它不看空间邻域，有什么用？答案是：**它在通道维度上做线性组合**。

$$Y_{i,j} = \sum_{c=0}^{C_{in}-1} K_c \cdot X_{c,i,j} + b$$

这等价于在每个空间位置上做一次全连接层（权重在所有位置共享）。

```
输入 (256通道, HxW)     1x1 卷积 (64个核)     输出 (64通道, HxW)
  ┌──┐                    ┌──┐                  ┌──┐
  │  │  每个位置的          │  │                  │  │  每个位置的
  │  │  256维向量    ──→   │  │  线性变换  ──→    │  │  64维向量
  │  │                    │  │                  │  │
  └──┘                    └──┘                  └──┘
```

**三大用途**：

1. **降维/升维**：ResNet 的 Bottleneck 块用 $1 \times 1$ 卷积把 256 通道降到 64，做完 $3 \times 3$ 卷积后再用 $1 \times 1$ 升回 256。这大幅减少了 $3 \times 3$ 卷积的计算量。

```
Bottleneck 块:
x (256通道) → [1x1 Conv, 64] → [3x3 Conv, 64] → [1x1 Conv, 256] → (+x) → 输出
              降维               核心卷积          升维               残差连接
```

2. **跨通道信息融合**：不同通道的特征图可以通过 $1 \times 1$ 卷积组合，创造新的特征表示。

3. **增加非线性**：$1 \times 1$ 卷积后接 ReLU，等于在不改变空间分辨率的情况下增加了一层非线性变换。这是 GoogLeNet（Inception）的核心思想之一。

---

## Python 代码示例

### 示例 1：numpy 手写 2D 卷积（单通道）

从零实现最基础的 2D 卷积操作，对照公式理解每一步。

```python
import numpy as np

# ============================================================
# 用 numpy 手写 2D 卷积（单通道，支持 stride 和 padding）
# 对应公式：Y[i,j] = sum_{m,n} X[i*s+m, j*s+n] * K[m,n] + b
# ============================================================

def conv2d(X, K, stride=1, padding=0, bias=0.0):
    """
    2D 卷积（互相关）运算。

    参数:
        X:       输入矩阵, shape (H, W)
        K:       卷积核, shape (Kh, Kw)
        stride:  步长
        padding: 零填充圈数
        bias:    偏置

    返回:
        输出矩阵, shape (Oh, Ow)
    """
    # --- 填充 ---
    if padding > 0:
        X = np.pad(X, pad_width=padding, mode='constant', constant_values=0)

    H, W = X.shape
    Kh, Kw = K.shape

    # --- 计算输出尺寸 ---
    Oh = (H - Kh) // stride + 1
    Ow = (W - Kw) // stride + 1

    # --- 卷积计算 ---
    Y = np.zeros((Oh, Ow))
    for i in range(Oh):
        for j in range(Ow):
            # 取出卷积核覆盖的区域
            region = X[i * stride : i * stride + Kh,
                       j * stride : j * stride + Kw]
            # 逐元素相乘再求和
            Y[i, j] = np.sum(region * K) + bias

    return Y


# ============================================================
# 测试: 基本卷积 + same padding + stride
# ============================================================
X = np.array([[1, 2, 3, 0], [0, 1, 2, 3],
              [3, 0, 1, 2], [2, 3, 0, 1]], dtype=float)
K = np.array([[0, 1, 2], [2, 2, 0], [0, 1, 2]], dtype=float)

Y = conv2d(X, K, stride=1, padding=0)
print(f"基本卷积: 输入(4x4) * 核(3x3) → 输出{Y.shape}")
print(f"Y[0,0] = {Y[0,0]}  (手算: 0+2+6+0+2+0+0+0+2 = 12)")

# same padding: padding=1 保持尺寸不变
X2 = np.arange(1, 26, dtype=float).reshape(5, 5)
K2 = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=float)
Y2 = conv2d(X2, K2, stride=1, padding=1)
print(f"Same padding: 输入{X2.shape} → 输出{Y2.shape} (尺寸不变)")

# stride=2 下采样
Y3 = conv2d(np.random.randn(8, 8), np.ones((3, 3)) / 9, stride=2)
print(f"Stride=2: 输入(8,8) → 输出{Y3.shape}  公式: (8-3)//2+1={3}")

# 输出尺寸公式验证
for W, Ks, S, P, name in [(28,5,1,0,"LeNet"), (224,7,2,3,"ResNet"), (56,3,1,1,"VGG")]:
    O = (W - Ks + 2*P)//S + 1
    Y_t = conv2d(np.random.randn(W, W), np.random.randn(Ks, Ks), S, P)
    print(f"  {name}: W={W},K={Ks},S={S},P={P} → O={O} (实际={Y_t.shape[0]}) OK")
```

### 示例 2：numpy 手写最大池化

```python
import numpy as np

# ============================================================
# 用 numpy 手写最大池化和平均池化
# ============================================================

def max_pool2d(X, pool_size=2, stride=2):
    """
    最大池化。

    参数:
        X:          输入矩阵, shape (H, W)
        pool_size:  池化窗口大小
        stride:     步长

    返回:
        输出矩阵, shape (Oh, Ow)
    """
    H, W = X.shape
    Oh = (H - pool_size) // stride + 1
    Ow = (W - pool_size) // stride + 1

    Y = np.zeros((Oh, Ow))
    for i in range(Oh):
        for j in range(Ow):
            region = X[i * stride : i * stride + pool_size,
                       j * stride : j * stride + pool_size]
            Y[i, j] = np.max(region)

    return Y


def avg_pool2d(X, pool_size=2, stride=2):
    """平均池化。"""
    H, W = X.shape
    Oh = (H - pool_size) // stride + 1
    Ow = (W - pool_size) // stride + 1

    Y = np.zeros((Oh, Ow))
    for i in range(Oh):
        for j in range(Ow):
            region = X[i * stride : i * stride + pool_size,
                       j * stride : j * stride + pool_size]
            Y[i, j] = np.mean(region)

    return Y


# ============================================================
# 测试: 最大池化 vs 平均池化 + 平移不变性演示
# ============================================================
X = np.array([[1, 3, 2, 4], [5, 6, 7, 8],
              [3, 2, 1, 0], [1, 2, 3, 4]], dtype=float)

Y_max = max_pool2d(X, pool_size=2, stride=2)
Y_avg = avg_pool2d(X, pool_size=2, stride=2)

print(f"输入 (4x4):\n{X}\n")
print(f"Max Pool: {Y_max.flatten()}  (窗口取最大值)")
print(f"Avg Pool: {Y_avg.flatten()}  (窗口取平均值)\n")

# 平移不变性: 激活位置变了，池化结果不变
feat1 = np.array([[9,1,0,0],[1,0,0,0],[0,0,0,0],[0,0,0,0]], dtype=float)
feat2 = np.array([[1,9,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]], dtype=float)
print(f"激活右移1px, 池化结果相同: {np.array_equal(max_pool2d(feat1,2,2), max_pool2d(feat2,2,2))}")

# 全局平均池化: 每个通道压缩为一个标量
feat_map = np.random.randn(3, 7, 7)
gap = feat_map.mean(axis=(1, 2))  # shape: (3,)
print(f"全局平均池化: (3, 7, 7) → {gap.shape}  (替代全连接层的经典做法)")
```

### 示例 3：PyTorch 简单 CNN 分类示例

完整可运行的端到端 CNN 训练流程：在合成数据上训练一个简单的 CNN 分类器。

```python
import torch
import torch.nn as nn
import numpy as np

# ============================================================
# 完整示例：用 PyTorch CNN 做图像分类（合成数据）
# 任务：区分"水平条纹"和"竖直条纹"图像
# ============================================================

torch.manual_seed(42)
np.random.seed(42)

# --- 生成合成数据 ---
def generate_stripe_data(n_samples, img_size=16):
    """
    生成带有水平或竖直条纹的图像。
    水平条纹 → 标签 0
    竖直条纹 → 标签 1
    CNN 需要学会检测条纹方向。
    """
    images = []
    labels = []

    for i in range(n_samples):
        img = np.random.randn(1, img_size, img_size) * 0.1  # 噪声背景

        if i < n_samples // 2:
            # 水平条纹
            for row in range(0, img_size, 4):
                img[0, row:row+2, :] += 1.0
            labels.append(0)
        else:
            # 竖直条纹
            for col in range(0, img_size, 4):
                img[0, :, col:col+2] += 1.0
            labels.append(1)

        images.append(img)

    X = torch.tensor(np.array(images), dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    # 打乱顺序
    perm = torch.randperm(n_samples)
    return X[perm], y[perm]


# 生成数据
X_train, y_train = generate_stripe_data(400, img_size=16)
X_test, y_test = generate_stripe_data(100, img_size=16)

print("=" * 60)
print("PyTorch CNN 图像分类")
print("=" * 60)
print(f"训练集: {X_train.shape} (N, C, H, W)")
print(f"测试集: {X_test.shape}")
print(f"任务: 区分水平条纹(0) vs 竖直条纹(1)")

# --- 定义 CNN 模型 ---
class SimpleCNN(nn.Module):
    """
    简单 CNN:
    Conv1 → ReLU → MaxPool → Conv2 → ReLU → MaxPool → FC1 → FC2
    """
    def __init__(self):
        super().__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(
            in_channels=1,    # 输入通道数（灰度图）
            out_channels=8,   # 输出通道数（8个卷积核）
            kernel_size=3,    # 卷积核大小 3x3
            padding=1,        # same padding
        )
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            padding=1,
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        # 经过两次 MaxPool (16x16 → 8x8 → 4x4)，通道数 16
        # 展平后维度: 16 * 4 * 4 = 256
        self.fc1 = nn.Linear(16 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        # x: (batch, 1, 16, 16)
        x = self.pool(self.relu(self.conv1(x)))  # → (batch, 8, 8, 8)
        x = self.pool(self.relu(self.conv2(x)))  # → (batch, 16, 4, 4)
        x = x.view(x.size(0), -1)                # → (batch, 256)
        x = self.relu(self.fc1(x))                # → (batch, 32)
        x = self.fc2(x)                           # → (batch, 2)
        return x


model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(f"\n模型结构:\n{model}")
print(f"总参数量: {sum(p.numel() for p in model.parameters()):,}\n")

# --- 训练循环 ---
batch_size = 32
n_epochs = 20

print(f"{'Epoch':>6s} | {'Loss':>8s} | {'Train Acc':>10s} | {'Test Acc':>9s}")
print("-" * 45)

for epoch in range(1, n_epochs + 1):
    model.train()
    total_loss, correct, total = 0, 0, 0
    indices = torch.randperm(len(X_train))

    for start in range(0, len(X_train), batch_size):
        idx = indices[start:start + batch_size]
        logits = model(X_train[idx])
        loss = criterion(logits, y_train[idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(idx)
        correct += (logits.argmax(1) == y_train[idx]).sum().item()
        total += len(idx)

    model.eval()
    with torch.no_grad():
        test_acc = (model(X_test).argmax(1) == y_test).float().mean().item()

    if epoch % 5 == 0 or epoch == 1:
        print(f"{epoch:>6d} | {total_loss/total:>8.4f} | {correct/total:>9.1%} | {test_acc:>8.1%}")

print(f"\n最终测试准确率: {test_acc:.1%}")
```

---

## 工程师视角

### nn.Conv2d 参数详解

```python
torch.nn.Conv2d(
    in_channels,       # 输入通道数 (RGB=3, 灰度=1, 或上一层的 out_channels)
    out_channels,      # 输出通道数 = 卷积核个数
    kernel_size,       # 卷积核大小, int 或 (h, w)
    stride=1,          # 步长, int 或 (h, w)
    padding=0,         # 零填充, int 或 (h, w)
    dilation=1,        # 空洞卷积的扩张率 (不常用，默认 1)
    groups=1,          # 分组卷积 (=1 普通卷积, =in_channels 深度可分离卷积)
    bias=True,         # 是否加偏置
    padding_mode='zeros',  # 填充模式: 'zeros', 'reflect', 'replicate', 'circular'
)
```

**关键参数选择指南**：

| 参数 | 典型值 | 建议 |
|------|--------|------|
| `kernel_size` | 3 | 几乎所有现代 CNN 都用 $3 \times 3$（VGG 以来的共识） |
| `stride` | 1 或 2 | 1 保持尺寸（配合 padding=1），2 做下采样 |
| `padding` | `kernel_size // 2` | 保持 same padding 的通用公式 |
| `out_channels` | 逐层翻倍 | 典型: 64→128→256→512（VGG/ResNet 的模式） |
| `bias` | `False` | 如果后面接 BatchNorm，偏置可以省略（BN 会吸收偏置） |
| `groups` | 1 | 除非做深度可分离卷积（MobileNet），否则用默认值 |

**参数量与计算量速算**：

$$\text{参数量} = C_{out} \times C_{in} \times K^2 + C_{out}$$

$$\text{FLOPs} = C_{out} \times O_h \times O_w \times C_{in} \times K^2$$

例：`Conv2d(64, 128, 3, padding=1)`，输入 $56 \times 56$：参数量 $= 128 \times 64 \times 9 + 128 = 73{,}856$，FLOPs $\approx 2.31$ 亿。

### 感受野（Receptive Field）

**感受野**是指输出特征图上一个像素"看到"的输入图像区域大小。这是理解 CNN 层级特征提取能力的关键概念。

#### 计算公式

对于连续的 $L$ 个卷积层，每层核大小 $k_l$、步长 $s_l$，感受野递推公式为：

$$\text{RF}_L = \text{RF}_{L-1} + (k_L - 1) \times \prod_{i=1}^{L-1} s_i$$

简单情况（所有层步长为 1）：$L$ 个 $k \times k$ 卷积层的感受野 = $1 + L \times (k - 1)$。

**示例**：

| 架构 | 感受野 |
|------|:------:|
| 1 层 $3 \times 3$，stride=1 | $3 \times 3$ |
| 2 层 $3 \times 3$，stride=1 | $5 \times 5$ |
| 3 层 $3 \times 3$，stride=1 | $7 \times 7$ |
| VGG-16（13 层 $3 \times 3$ + 池化） | $212 \times 212$ |
| ResNet-50 最后一层 | 约 $483 \times 483$ |

**工程含义**：感受野太小则网络只能看到局部纹理；池化和步长 > 1 能快速扩大感受野；空洞卷积（dilated convolution）可在不增加参数的情况下扩大感受野。

### CNN 与 Vision Transformer 的关系

2020 年后，Vision Transformer（ViT）在图像任务上追平甚至超过了 CNN。它们的核心区别：

| 对比维度 | CNN | Vision Transformer (ViT) |
|----------|-----|--------------------------|
| **基本操作** | 局部卷积（$3 \times 3$ 窗口） | 全局自注意力（所有 patch 互相看） |
| **归纳偏置** | 强（局部性 + 平移不变性内建） | 弱（需要从数据中学习空间关系） |
| **数据效率** | 高（小数据集表现好） | 低（需要大量数据才能超过 CNN） |
| **感受野** | 逐层扩大，深层才看全图 | 第一层就能看到全图（全局注意力） |
| **计算复杂度** | $O(K^2 \cdot C^2 \cdot H \cdot W)$ | $O(N^2 \cdot D)$，$N$ = patch 数 |
| **位置信息** | 隐式（由卷积结构保证） | 显式（需要位置编码） |

**当代趋势**：CNN 和 Transformer 正在融合。ConvNeXt 用纯卷积追平 Transformer；Swin Transformer 引入了 CNN 的局部窗口思想。现代高性能视觉模型往往混合使用两者。

**为什么还要学 CNN？** ViT 的 patch 投影本质上就是卷积；边缘设备和实时推理仍依赖 CNN 的效率优势；CNN 的核心概念（局部感受野、层级特征、权值共享）是理解所有视觉模型的基础。

### 1D 卷积在 NLP 中的应用

卷积不仅能处理 2D 图像，也能处理 1D 序列。在 Transformer 之前，1D CNN 曾是 NLP 中重要的特征提取器。

**核心思想**：把一句话看成 1D 序列，词嵌入维度作为通道。`kernel_size=3` 的 1D 卷积核覆盖 3 个连续词，相当于 trigram 特征提取。PyTorch 用 `nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)`。

**TextCNN（2014）** 是经典的 NLP 卷积模型：用多种大小的卷积核（$K=3,4,5$）并行提取不同长度的 n-gram 特征，全局最大池化后拼接分类。

**1D 卷积 vs RNN vs Transformer 在 NLP 中的对比**：

| 特性 | 1D CNN | RNN | Transformer |
|------|--------|-----|-------------|
| 长距离依赖 | 受限于核大小 | 受限于梯度消失 | 无限（自注意力） |
| 并行性 | 高 | 低（串行） | 高 |
| 参数效率 | 高 | 中 | 低（但效果好） |
| 现状 | 轻量场景仍有用 | 基本被取代 | 主流 |

如今 NLP 由 Transformer 主导，但 1D 卷积仍用于语音前端特征提取（Wav2Vec）、时间序列预测（TCN）和 Transformer 内部的局部增强模块。

---

## 本章小结

| 概念 | 一句话总结 |
|------|-----------|
| **CNN 核心思想** | 局部连接 + 权值共享 + 层级特征提取，参数极少却能捕获空间结构 |
| **卷积运算** | 卷积核在输入上滑动，逐位置做加权求和（实际是互相关） |
| **输出尺寸公式** | $O = \lfloor(W - K + 2P) / S\rfloor + 1$ |
| **多通道卷积** | 核通道数 = 输入通道数，多个核 → 多个输出特征图 |
| **池化层** | 最大/平均池化降低分辨率，增强平移不变性，无可学习参数 |
| **LeNet-5** | CNN 奠基之作，确立"卷积→池化→全连接"流水线 |
| **AlexNet** | GPU 训练 + ReLU + Dropout，引爆深度学习 |
| **VGGNet** | 小核堆叠（$3 \times 3$），证明深度的重要性 |
| **ResNet** | 残差连接 $y = F(x) + x$，解决深层网络优化困难，影响至今（含 Transformer） |
| **$1 \times 1$ 卷积** | 通道维度的线性组合，用于降维/升维/跨通道融合 |
| **感受野** | 输出像素"看到"的输入区域大小，逐层扩大 |

**下一章**：[RNN 循环神经网络](2_RNN循环神经网络.md)——如何处理变长序列？隐藏状态如何编码时序信息？为什么梯度会消失？
