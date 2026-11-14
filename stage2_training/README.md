# Stage 2: 训练目标 — 从 ELBO 到 L_simple

## 1. 直觉先行

Stage 1 我们学会了如何**摧毁**数据（前向加噪）。现在要训练一个模型来**恢复**数据（反向去噪）。

核心问题：**怎么训练这个模型？损失函数是什么？**

最自然的想法：让模型预测每一步怎么去噪，跟真实去噪方向对比。但"真实去噪方向"是什么？前向过程是确定的，反向过程我们不知道——这正是要学习的。

扩散模型的精妙之处：**通过变分推断（Variational Inference），把"学习反向过程"转化为"预测前向加进去的噪声"**。这是一个看似简单但极其深刻的转化。

---

## 2. 数学推导

### 2.1 扩散模型作为潜变量模型

扩散模型定义了一个潜变量模型：

$$p_\theta(x_0) = \int p_\theta(x_{0:T}) \, dx_{1:T}$$

其中联合分布为：

$$p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1} | x_t)$$

- $p(x_T) = \mathcal{N}(x_T; 0, \mathbf{I})$：先验，纯噪声
- $p_\theta(x_{t-1} | x_t)$：反向过程，由神经网络参数化

训练目标是最大化数据的对数似然 $\log p_\theta(x_0)$。但由于积分不可解析，我们优化其下界（ELBO）。

### 2.2 ELBO 推导

利用 Jensen 不等式和变分推断，引入前向过程 $q(x_{1:T}|x_0)$ 作为变分后验：

$$\log p_\theta(x_0) \geq \mathbb{E}_{q(x_{1:T}|x_0)} \left[ \log \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)} \right] = \text{ELBO}$$

将联合分布展开：

$$\text{ELBO} = \mathbb{E}_q \left[ \log p(x_T) + \sum_{t=1}^{T} \log p_\theta(x_{t-1}|x_t) - \sum_{t=1}^{T} \log q(x_t|x_{t-1}) \right]$$

利用贝叶斯公式重写 $q(x_t|x_{t-1})$：

$$q(x_t|x_{t-1}) = \frac{q(x_t|x_0) \, q(x_{t-1}|x_0)}{q(x_{t-1}|x_t, x_0)}$$

代入并重新组织（标准ELBO分解技巧）：

$$\text{ELBO} = \underbrace{\mathbb{E}_q[\log p_\theta(x_0|x_1)]}_{L_0} - \underbrace{D_{KL}(q(x_T|x_0) \| p(x_T))}_{L_T} - \sum_{t=2}^{T} \underbrace{D_{KL}(q(x_{t-1}|x_t,x_0) \| p_\theta(x_{t-1}|x_t))}_{L_t}$$

**三项含义**：
- $L_0$：重建项，从 $x_1$ 预测 $x_0$
- $L_T$：先验匹配项，$x_T$ 应接近纯噪声（此项无需训练，因为前向过程保证 $x_T \approx \mathcal{N}(0, \mathbf{I})$）
- $L_t$：去噪匹配项，**这是核心训练目标**

### 2.3 后验分布 $q(x_{t-1}|x_t, x_0)$

这是整个推导的关键。给定 $x_0$ 和 $x_t$，$x_{t-1}$ 的后验分布是高斯的：

$$q(x_{t-1}|x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t \mathbf{I})$$

**推导均值 $\tilde{\mu}_t$**：

利用贝叶斯公式和 Stage 1 的闭式解：

$$q(x_{t-1}|x_t, x_0) \propto q(x_t|x_{t-1}) \, q(x_{t-1}|x_0)$$

两个高斯相乘仍是高斯。展开指数项并配方：

$$q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{\alpha_t} x_{t-1}, (1-\alpha_t)\mathbf{I})$$

$$q(x_{t-1}|x_0) = \mathcal{N}(x_{t-1}; \sqrt{\bar{\alpha}_{t-1}} x_0, (1-\bar{\alpha}_{t-1})\mathbf{I})$$

将两个高斯的指数部分相加，对 $x_{t-1}$ 配方，得到：

$$\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}(1-\alpha_t)}{1-\bar{\alpha}_t} x_0$$

**推导方差 $\tilde{\beta}_t$**：

$$\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t} \cdot (1-\alpha_t) = \beta_t \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}$$

### 2.4 参数化反向过程 $p_\theta(x_{t-1}|x_t)$

将反向过程也设为高斯：

$$p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

DDPM 固定方差 $\Sigma_\theta = \tilde{\beta}_t \mathbf{I}$（不学习方差，只学习均值），因此 $L_t$ 变为：

$$L_t = \mathbb{E}_q \left[ \frac{1}{2\tilde{\beta}_t} \| \tilde{\mu}_t(x_t, x_0) - \mu_\theta(x_t, t) \|^2 \right]$$

### 2.5 关键转化：从预测均值到预测噪声

将 $\tilde{\mu}_t$ 中的 $x_0$ 用 Stage 1 的闭式解替换：

$$x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}} (x_t - \sqrt{1-\bar{\alpha}_t} \, \epsilon)$$

代入 $\tilde{\mu}_t$ 并化简（代数过程略，核心是 $\sqrt{\alpha_t} \cdot \frac{1}{\sqrt{\bar\alpha_t}} = \frac{1}{\sqrt{\bar\alpha_{t-1}}}$ 等恒等式）：

$$\tilde{\mu}_t = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon \right)$$

**设计决策**：让模型也以同样形式参数化均值：

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)$$

其中 $\epsilon_\theta(x_t, t)$ 是神经网络，**预测前向过程加入的噪声 $\epsilon$**。

代入 $L_t$：

$$\| \tilde{\mu}_t - \mu_\theta \|^2 = \frac{\beta_t^2}{\alpha_t (1-\bar{\alpha}_t)} \| \epsilon - \epsilon_\theta(x_t, t) \|^2$$

因此：

$$L_t = \mathbb{E}_q \left[ \frac{\beta_t^2}{2\alpha_t \tilde{\beta}_t (1-\bar{\alpha}_t)} \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$$

### 2.6 简化：L_simple

DDPM 论文的关键简化——**去掉权重系数**，对所有时间步等权处理：

$$\boxed{L_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon, \, t) \|^2 \right]}$$

其中 $t \sim \text{Uniform}(\{1, ..., T\})$，$\epsilon \sim \mathcal{N}(0, \mathbf{I})$。

**为什么去掉权重系数反而更好？**

1. **梯度均衡**：权重系数 $\frac{\beta_t^2}{2\alpha_t \tilde{\beta}_t (1-\bar\alpha_t)}$ 在不同 $t$ 下差异巨大，小 $t$ 时系数很小（信号多，容易预测），大 $t$ 时系数很大（噪声多，难预测）。去掉后所有时间步贡献均衡的梯度。

2. **实践验证**：DDPM 论文实验表明，L_simple 比带权重的完整 ELBO 生成质量更好、训练更稳定。

3. **计算简洁**：不需要计算复杂的系数，只需采样 $(x_0, t, \epsilon)$，计算 $\| \epsilon - \epsilon_\theta(x_t, t) \|^2$。

### 2.7 为什么预测 $\epsilon$ 而不是 $x_0$ 或 score？

三种等价参数化方式：

| 预测目标 | 关系 | 训练损失 |
|----------|------|----------|
| $\epsilon$（噪声） | $x_0 = \frac{1}{\sqrt{\bar\alpha_t}}(x_t - \sqrt{1-\bar\alpha_t}\epsilon)$ | $\|\epsilon - \epsilon_\theta\|^2$ |
| $x_0$（原始数据） | 直接预测 | $\|x_0 - x_{\theta}(x_t,t)\|^2$（加权后等价） |
| $s$（score function） | $s = -\frac{\epsilon}{\sqrt{1-\bar\alpha_t}}$ | score matching loss |

**预测 $\epsilon$ 最优的原因**：

1. **数值稳定性**：$\epsilon$ 始终是标准正态分布，无论 $t$ 多大，目标分布不变。预测 $x_0$ 时，不同 $t$ 下 $x_0$ 的"信号"在 $x_t$ 中占比变化剧烈，梯度尺度不稳定。

2. **与去噪 score matching 的等价性**：预测 $\epsilon$ 等价于预测 score function $\nabla_{x_t} \log q(x_t|x_0) = -\frac{\epsilon}{\sqrt{1-\bar\alpha_t}}$，有深厚的理论基础。

3. **训练效率**：$\epsilon$ 的方差恒为1，不需要对不同 $t$ 做梯度归一化。

---

## 3. 代码对应

| 公式 | 代码位置 | 函数 |
|------|----------|------|
| $L_{\text{simple}}$ 训练目标 | `train.py` | `train_step()` |
| $\epsilon_\theta(x_t, t)$ 模型 | `simple_model.py` | `DiffusionMLP` |
| $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon$ | `forward_diffusion.py` | `q_sample()` |
| 2D玩具数据分布 | `toy_dataset.py` | `get_toy_data()` |
| 反向采样 $p_\theta(x_{t-1}\|x_t)$ | `train.py` | `p_sample()` |
| 可视化2D扩散 | `visualize.py` | `main()` |

---

## 4. 实验验证

```bash
python stage2_training/train.py         # 训练2D玩具模型
python stage2_training/visualize.py     # 可视化反向去噪过程
```

**验证要点**：
- 训练损失应持续下降并收敛
- 从纯噪声开始反向采样，应逐步收敛到玩具数据的原始分布
- 不同时间步的 $\epsilon_\theta$ 输出应接近标准正态分布