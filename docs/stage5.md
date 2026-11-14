# Stage 5 — 条件生成

## 核心思想

前面的阶段都是**无条件生成**——模型从噪声随机生成数据，无法控制生成内容。条件生成让模型接收类别标签 $y$，生成指定类别的数据：$\epsilon_\theta(x_t, t, y)$。

**Classifier-free guidance (CFG)** 是当前最流行的条件生成技术，核心思想是同时训练条件和无条件模型，推理时用两者的差值作为"引导方向"。

## 条件 U-Net

在 Stage 3 的 U-Net 基础上增加类别条件注入：

### ClassEmbedding — 类别嵌入

```python
class ClassEmbedding(nn.Module):
    def __init__(self, num_classes, embed_dim):
        self.null_class = num_classes  # null 类别的索引
        self.embedding = nn.Embedding(num_classes + 1, embed_dim)

    def forward(self, y, drop_prob=0.0):
        if drop_prob > 0:
            drop_mask = torch.rand(y.shape[0]) < drop_prob
            y = torch.where(drop_mask, torch.full_like(y, self.null_class), y)
        return self.embedding(y)
```

关键设计：包含一个额外的 **null 类别**（索引 = `num_classes`），用于 CFG 的无条件生成。训练时以概率 `p_uncond` 将 $y$ 替换为 null 类别。

### ConditionalResidualBlock — 双条件注入

时间嵌入和类别嵌入**相加后一起注入**：

```python
class ConditionalResidualBlock(nn.Module):
    def forward(self, x, time_emb, class_emb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        # 时间 + 类别 双条件注入
        t = self.time_proj(time_emb)
        c = self.class_proj(class_emb)
        h = h + t[:, :, None, None] + c[:, :, None, None]
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return h + self.shortcut(x)
```

## Classifier-Free Guidance

### 训练

训练时，以概率 $p_{uncond}$（通常10%-20%）将类别标签替换为 null：

$$\epsilon_\theta(x_t, t, y) = \begin{cases} \epsilon_\theta(x_t, t, \text{null}) & \text{with prob } p_{uncond} \\ \epsilon_\theta(x_t, t, y) & \text{with prob } 1-p_{uncond} \end{cases}$$

同一个模型同时学习条件和无条件生成。

### 推理

推理时，同时计算条件和无条件预测，用**引导强度** $w$ 放大条件方向：

$$\tilde{\epsilon}_\theta = \epsilon_\theta(x_t, t, \text{null}) + w \cdot (\epsilon_\theta(x_t, t, y) - \epsilon_\theta(x_t, t, \text{null}))$$

- $w = 0$：纯无条件生成
- $w = 1$：纯条件生成
- $w > 1$：**放大条件方向**，生成更符合类别但多样性降低

### 直觉

CFG 的直觉类似于"放大信号"：无条件预测是"基准"，条件预测是"方向"，两者的差值就是"类别信号"。放大这个差值 = 更强的类别引导。

## 代码实现

### `conditional_unet.py`

```python
class ConditionalUNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, base_channels=64,
                 time_emb_dim=256, class_emb_dim=128, ...):
        self.time_embedding = TimeEmbeddingMLP(base_channels * 4, time_emb_dim)
        self.class_embedding = ClassEmbedding(num_classes, class_emb_dim)
        # 编码器/解码器使用 ConditionalResidualBlock

    def forward(self, x, t, y, drop_prob=0.0):
        time_emb = self.time_embedding(t)
        class_emb = self.class_embedding(y, drop_prob=drop_prob)
        # ... 编码器 → bottleneck → 解码器
```

### `cfg_train.py` / `cfg_sample.py`

训练和采样脚本，实现 CFG 的完整流程。

## 运行

```bash
# 训练条件 U-Net
python stage5_advanced/cfg_train.py

# 条件生成
python stage5_advanced/cfg_sample.py
```

## 关键文件

| 文件 | 说明 |
|------|------|
| `stage5_advanced/conditional_unet.py` | `ClassEmbedding`, `ConditionalResidualBlock`, `ConditionalUNet` |
| `stage5_advanced/cfg_train.py` | CFG 训练脚本 |
| `stage5_advanced/cfg_sample.py` | CFG 条件采样脚本 |