# Stage 3 — U-Net 架构

## 核心思想

Stage 2 用 MLP 处理2D玩具数据，但图像生成需要处理 $(B, C, H, W)$ 的空间数据。扩散模型的标准架构是 **U-Net**：编码器-解码器结构 + skip connection，在低分辨率层使用自注意力整合全局信息。

U-Net 的任务是 $\epsilon_\theta(x_t, t)$：输入带噪图像 $x_t$ 和时间步 $t$，输出预测噪声 $\epsilon_\theta$。

## 架构设计

### 时间嵌入

时间步 $t$ 是一个整数，需要编码为连续向量才能输入网络。使用**正弦位置编码**（Transformer positional encoding 的变体）：

$$PE(t, 2i) = \sin\left(\frac{t}{10000^{2i/d}}\right), \quad PE(t, 2i+1) = \cos\left(\frac{t}{10000^{2i/d}}\right)$$

设计哲学：

- **连续性**：相近的 $t$ 产生相近的嵌入
- **多频率**：低频维度捕捉粗粒度时间信息，高频维度捕捉细粒度
- **无需学习**：编码本身确定，不需额外参数

然后通过 MLP 投影到各层所需的维度。

### ResidualBlock — 时间条件注入

```
GroupNorm → SiLU → Conv → +time_emb → GroupNorm → SiLU → Conv → +residual
```

时间嵌入通过**加法**注入：把 `time_emb` 投影到通道维度后加到中间特征上。

### SelfAttention — 全局信息整合

在低分辨率层（$H, W \leq 8$）使用自注意力，整合全局信息。计算复杂度 $O(H^2W^2)$，所以只在小分辨率使用。

### 编码器-解码器 + Skip Connection

```
输入 → Conv → [ResBlock × 2 → Downsample] × 3 → Bottleneck → [Upsample → ResBlock × 3] × 3 → Conv → 输出
                                    ↓ skip connection ↑
```

3层下采样 + bottleneck + 3层上采样，每层通过 skip connection 连接对应分辨率的特征。

## 代码实现

### `sinusoidal_embedding.py`

```python
class TimeEmbeddingMLP(nn.Module):
    """t → sinusoidal emb → MLP → projected emb"""
    def __init__(self, dim, time_emb_dim):
        self.sinusoidal = SinusoidalTimeEmbedding(time_emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(time_emb_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
    def forward(self, t):
        emb = self.sinusoidal(t)
        return self.mlp(emb)
```

### `unet.py` — 完整 U-Net

```python
class UNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, time_emb_dim=256,
                 channel_mults=(1, 2, 4), num_res_blocks=2, attn_resolutions=(8,)):
        # 时间嵌入
        self.time_embedding = TimeEmbeddingMLP(base_channels * 4, time_emb_dim)
        # 输入卷积
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        # 编码器 (3层下采样)
        # Bottleneck (ResBlock + SelfAttention + ResBlock)
        # 解码器 (3层上采样, skip connection)
        # 输出卷积
```

通道倍增 `(1, 2, 4)` 意味着：64 → 128 → 256 通道，分辨率 28 → 14 → 7。

## MNIST 训练

```bash
python stage3_unet/train_mnist.py
```

在 MNIST 上训练 U-Net，学习从噪声生成手写数字。

## 关键文件

| 文件 | 说明 |
|------|------|
| `stage3_unet/sinusoidal_embedding.py` | `SinusoidalTimeEmbedding`, `TimeEmbeddingMLP` |
| `stage3_unet/unet.py` | `ResidualBlock`, `SelfAttention`, `UNet` |
| `stage3_unet/train_mnist.py` | MNIST 训练脚本 |
| `stage3_unet/sample.py` | 生成采样脚本 |