# Stage 5: 条件生成与进阶 — Classifier-free Guidance & Latent Diffusion

## 1. 直觉先行

前4个阶段我们做的是**无条件生成**——模型随机生成数字，无法控制生成哪个数字。但实际应用中，我们需要**条件生成**：告诉模型"生成一个7"，它就生成7。

**两种主流的条件生成方案**：

1. **Classifier Guidance**（2021）：训练一个独立的分类器 $p(y|x_t)$，在采样时用分类器的梯度引导生成方向。缺点：需要额外训练分类器，且分类器需要对带噪图像鲁棒。

2. **Classifier-free Guidance (CFG)**（2022）：**不需要分类器**。同一个扩散模型，既学习条件生成 $\epsilon_\theta(x_t, t, y)$，又学习无条件生成 $\epsilon_\theta(x_t, t, \varnothing)$。采样时用两者的差值作为引导。这是目前最主流的方案。

**Latent Diffusion**：在像素空间做扩散对高分辨率图像太昂贵。Stable Diffusion 的核心创新是先压缩到 latent space，在低维空间做扩散，再解码回像素空间。

---

## 2. Classifier-free Guidance 推导

### 2.1 条件扩散模型

把 U-Net 从 $\epsilon_\theta(x_t, t)$ 扩展为 $\epsilon_\theta(x_t, t, y)$，其中 $y$ 是条件（类别标签、文本嵌入等）。

条件注入方式：把 $y$ 编码为向量，与时间嵌入一起注入 ResBlock。

训练目标变为：

$$L = \mathbb{E}_{t, x_0, \epsilon, y} \left[ \| \epsilon - \epsilon_\theta(x_t, t, y) \|^2 \right]$$

### 2.2 问题：只有条件生成不够

如果模型只学习 $\epsilon_\theta(x_t, t, y)$，采样时必须有条件 $y$。但我们还需要：
- **无条件生成能力**：用于 CFG 的引导差值计算
- **鲁棒性**：模型不应该过度依赖条件

### 2.3 CFG 训练：随机丢弃条件

**关键技巧**：训练时以概率 $p_{\text{uncond}}$（通常10-20%）将条件 $y$ 替换为空标记 $\varnothing$。

$$L = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t, y') \|^2 \right]$$

其中 $y' = y$ 以概率 $1 - p_{\text{uncond}}$，$y' = \varnothing$ 以概率 $p_{\text{uncond}}$。

这样**同一个模型**既学会了条件生成，又学会了无条件生成。

### 2.4 CFG 采样：引导公式

采样时，同时计算条件预测和无条件预测，然后用差值引导：

$$\epsilon_{\text{guided}} = \epsilon_\theta(x_t, t, \varnothing) + w \cdot \left[ \epsilon_\theta(x_t, t, y) - \epsilon_\theta(x_t, t, \varnothing) \right]$$

其中 $w$ 是**引导强度**（guidance scale）：
- $w = 0$：纯无条件生成
- $w = 1$：纯条件生成（无引导增强）
- $w > 1$：条件方向被放大（增强条件匹配，但减少多样性）
- $w < 0$：条件方向被反转（生成与条件相反的内容）

**直觉理解**：$\epsilon_\theta(x_t, t, y) - \epsilon_\theta(x_t, t, \varnothing)$ 是"条件 $y$ 带来的额外信息方向"。把这个方向放大 $w$ 倍，就像对生成过程说"更往 $y$ 的方向走"。

### 2.5 CFG 的数学基础

从 score function 的角度理解：

$$\nabla_{x_t} \log p(x_t | y) = \nabla_{x_t} \log p(x_t) + \nabla_{x_t} \log p(y | x_t)$$

即：条件 score = 无条件 score + 条件带来的修正。

CFG 的公式等价于：

$$\tilde{\epsilon} = (1-w) \cdot \epsilon_{\text{uncond}} + w \cdot \epsilon_{\text{cond}}$$

当 $w > 1$ 时，相当于放大了 $\nabla_{x_t} \log p(y|x_t)$ 的贡献——更强烈地往条件方向走。

### 2.6 引导强度 $w$ 的影响

| $w$ | 效果 | 质量 | 多样性 |
|-----|------|------|--------|
| 0 | 无条件生成 | 低（无目标） | 最高 |
| 1 | 条件生成 | 中 | 高 |
| 3-5 | 典型CFG | 高 | 中 |
| 10+ | 强引导 | 很高（但可能过饱和） | 低 |
| 50+ | 极端引导 | 伪影 | 极低 |

**设计哲学**：$w$ 是质量-多样性的权衡旋钮。Stable Diffusion 默认 $w=7.5$。

---

## 3. Latent Diffusion

### 3.1 问题：像素空间扩散太昂贵

一张 256×256×3 的图像有 196608 个维度。在像素空间做扩散：
- U-Net 需要在 256×256 分辨率上操作，计算量大
- 训练需要大量 GPU 内存
- 采样需要 1000 步前向传播，每步都在高分辨率上

### 3.2 方案：压缩到 Latent Space

**两阶段方案**：

1. **感知压缩**：用 VAE（Variational Autoencoder）把图像压缩到低维 latent space
   - 编码器 $E$：$x \in \mathbb{R}^{H \times W \times 3} \rightarrow z \in \mathbb{R}^{h \times w \times c}$
   - 解码器 $D$：$z \rightarrow \hat{x}$
   - 通常 $h = H/8, w = W/8$，维度降低 64 倍

2. **latent 扩散**：在 $z$ 空间做扩散，而不是 $x$ 空间
   - 前向：$z_t = \sqrt{\bar\alpha_t} z_0 + \sqrt{1-\bar\alpha_t} \epsilon$
   - 反向：$\epsilon_\theta(z_t, t)$ 在低维 latent 上操作
   - 采样后用解码器恢复：$\hat{x} = D(z_0)$

### 3.3 为什么能这样做？

VAE 的 latent space 保留了图像的**语义信息**（物体类别、布局、风格），丢弃了**高频细节**（纹理、噪声）。扩散模型需要学习的是语义层面的生成，高频细节由 VAE 解码器负责恢复。

**分工**：
- VAE：负责高频细节的压缩与恢复
- 扩散模型：负责语义层面的生成与组合

### 3.4 Latent Diffusion 的优势

| 维度 | 像素空间 | Latent 空间 |
|------|----------|-------------|
| 维度 | 256×256×3 = 196608 | 32×32×4 = 4096 |
| 压缩比 | 1× | 48× |
| 训练成本 | 高 | 低 |
| 采样速度 | 慢 | 快 |
| 生成质量 | 好 | 好（VAE补充细节） |

---

## 4. 代码对应

| 概念 | 代码位置 | 类/函数 |
|------|----------|---------|
| 条件 U-Net | `conditional_unet.py` | `ConditionalUNet` |
| 类别嵌入 | `conditional_unet.py` | `ClassEmbedding` |
| CFG 训练 | `cfg_train.py` | `train_step_cfg()` |
| CFG 采样 | `cfg_sample.py` | `cfg_sample()` |
| 引导强度控制 | `cfg_sample.py` | `guidance_scale` 参数 |

---

## 5. 实验验证

```bash
python stage5_advanced/cfg_train.py     # 训练条件扩散模型
python stage5_advanced/cfg_sample.py    # 条件生成 + 引导强度对比
```

**验证要点**：
- 条件生成：指定数字"7"，应生成7
- 引导强度对比：$w=0$（随机）→ $w=1$（条件）→ $w=5$（强引导）
- 无条件生成：$y=\varnothing$ 时应生成随机数字