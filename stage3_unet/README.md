# Stage 3: U-Net 架构与 MNIST 图像生成

## 1. 直觉先行

Stage 2 我们用 MLP 在 2D 玩具数据上验证了扩散模型的训练逻辑。现在要处理真正的图像——28×28 的 MNIST 手写数字。

**为什么不能继续用 MLP？**

MLP 把输入展平为一维向量，完全丢失了空间结构。一个 28×28 的图像有 784 个像素，MLP 需要学习每个像素之间的关系——这既低效又难以扩展。

**为什么选 U-Net？**

扩散模型的核心任务是：**给定带噪图像 $x_t$ 和时间步 $t$，预测噪声 $\epsilon_\theta(x_t, t)$**。这本质上是一个图像到图像的转换任务——输入一张带噪图，输出一张同样大小的噪声预测图。U-Net 正是为这类任务设计的：

1. **编码器-解码器结构**：先提取多尺度特征（编码器），再逐步恢复空间分辨率（解码器）
2. **Skip connections**：把编码器的高分辨率特征直接传给解码器，保留细节信息
3. **多尺度感受野**：通过下采样-上采样，网络在不同分辨率上处理信息

---

## 2. 架构设计

### 2.1 整体结构

```
输入 x_t (B, C, H, W)
  │
  ├─ Time embedding (t → vector)
  │
  ▼
┌─────────────────────────────────────┐
│  Conv in → ResBlock → ResBlock      │  ← encoder block 1 (H×W)
│    │ (skip)                         │
│    ▼                                │
│  Downsample                         │
│  ResBlock → ResBlock                │  ← encoder block 2 (H/2×W/2)
│    │ (skip)                         │
│    ▼                                │
│  Downsample                         │
│  ResBlock → ResBlock → Attn         │  ← encoder block 3 (H/4×W/4)
│    │ (skip)                         │
│    ▼                                │
│  Downsample                         │
│  ResBlock → ResBlock → Attn         │  ← bottleneck (H/8×W/8)
│                                     │
│  Upsample                           │
│  ResBlock → ResBlock → Attn         │  ← decoder block 3
│    │ ← skip from encoder 3          │
│    ▼                                │
│  Upsample                           │
│  ResBlock → ResBlock                │  ← decoder block 2
│    │ ← skip from encoder 2          │
│    ▼                                │
│  Upsample                           │
│  ResBlock → ResBlock                │  ← decoder block 1
│    │ ← skip from encoder 1          │
│    ▼                                │
│  Conv out → ε_θ (B, C, H, W)        │
└─────────────────────────────────────┘
```

### 2.2 时间嵌入：Sinusoidal Positional Encoding

**问题**：时间步 $t$ 是一个整数（0 到 999），如何让网络理解"现在是第几步"？

**方案**：用正弦/余弦位置编码（来自 Transformer）将整数 $t$ 映射为连续向量：

$$\text{PE}(t, 2i) = \sin\left(\frac{t}{10000^{2i/d}}\right), \quad \text{PE}(t, 2i+1) = \cos\left(\frac{t}{10000^{2i/d}}\right)$$

其中 $d$ 是嵌入维度，$i$ 是维度索引。

**设计哲学**：
- **连续性**：相近的 $t$ 产生相近的嵌入向量，让网络能平滑地处理不同时间步
- **多频率**：不同维度对应不同频率的正弦波，低频维度捕捉粗粒度时间信息（早期 vs 晚期），高频维度捕捉细粒度信息（相邻步的差异）
- **无需学习**：编码本身是确定的，不需要额外参数

嵌入向量再通过一个 MLP 投影到各层所需的维度，然后通过加法注入 ResBlock。

### 2.3 Residual Block (ResBlock)

每个 ResBlock 的结构：

```
x_t feature (B, C, H, W)
  │
  ├─ GroupNorm → SiLU → Conv3×3
  │       │
  │       └─ + time embedding (经过 SiLU + Linear)
  │
  ├─ GroupNorm → SiLU → Conv3×3
  │
  └─ + residual (如果输入输出通道不同, 用1×1 Conv投影)
```

**为什么用 GroupNorm 而不是 BatchNorm？**
- BatchNorm 在小 batch size 时统计不稳定，扩散模型训练常用较小 batch
- GroupNorm 把通道分组归一化，不依赖 batch 统计，更稳定

**为什么用 SiLU (Swish) 而不是 ReLU？**
- SiLU = $x \cdot \text{sigmoid}(x)$，平滑且处处可微
- ReLU 在负半轴梯度为0，可能"杀死"神经元
- SiLU 在扩散模型中被广泛验证效果更好

### 2.4 Self-Attention

在较低分辨率层（H/4, H/8）加入自注意力：

```
x (B, C, H, W) → reshape (B, HW, C)
  │
  ├─ Q = Linear(C, C)
  ├─ K = Linear(C, C)
  ├─ V = Linear(C, C)
  │
  ├─ Attention = softmax(QK^T / √C) · V
  │
  └─ + residual → reshape back (B, C, H, W)
```

**为什么只在低分辨率层加注意力？**
- 注意力的计算复杂度是 $O(H^2W^2)$，在高分辨率层太昂贵
- 低分辨率层特征图小（如 4×4=16 个 token），注意力开销可接受
- 低分辨率层需要全局信息整合（理解"整体在生成什么"），注意力正好提供这个能力

### 2.5 Downsample / Upsample

- **Downsample**：用 stride=2 的卷积，同时降分辨率和增加通道数
- **Upsample**：先用 `F.interpolate` 放大2倍，再用卷积调整通道数

### 2.6 Skip Connection

编码器每层的输出直接拼接到解码器对应层：

$$x_{\text{decoder}} = \text{Cat}(x_{\text{upsampled}}, x_{\text{skip}})$$

**为什么需要 skip connection？**
- 扩散模型需要保留输入图像的细节（像素级精度）
- 编码器在降采样过程中丢失了高频细节，skip connection 把这些细节"抄"给解码器
- 这让网络可以专注于在低分辨率层学习语义信息，在高分辨率层只做细节调整

---

## 3. 代码对应

| 组件 | 代码位置 | 类/函数 |
|------|----------|---------|
| 正弦时间嵌入 | `sinusoidal_embedding.py` | `SinusoidalTimeEmbedding` |
| ResBlock | `unet.py` | `ResidualBlock` |
| Self-Attention | `unet.py` | `SelfAttention` |
| Downsample | `unet.py` | `Downsample` |
| Upsample | `unet.py` | `Upsample` |
| U-Net 整体 | `unet.py` | `UNet` |
| MNIST 训练 | `train_mnist.py` | `train()` |
| 生成采样 | `sample.py` | `sample()` |

---

## 4. 实验验证

```bash
python stage3_unet/train_mnist.py    # 训练 (约10-30分钟)
python stage3_unet/sample.py         # 生成手写数字
```

**验证要点**：
- 训练 loss 应持续下降（目标 < 0.05）
- 生成的数字应可辨认为人手写的 0-9
- 不同随机种子应生成不同的数字（说明模型学到了分布而非记忆）