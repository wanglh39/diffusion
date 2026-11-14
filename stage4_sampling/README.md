# Stage 4: 采样算法 — DDPM 与 DDIM

## 1. 直觉先行

Stage 3 我们实现了 DDPM 采样：从纯噪声开始，**逐步**去噪 1000 步才能生成一张图。这太慢了——生成一张图需要 1000 次神经网络前向传播。

**DDIM (Denoising Diffusion Implicit Models)** 解决了这个问题：用 **50 步**就能达到接近 1000 步 DDPM 的质量。50 倍加速。

**核心洞察**：DDPM 的训练目标和采样过程是**解耦的**。训练时学的是 $\epsilon_\theta(x_t, t)$——预测噪声的能力。这个能力一旦学会，怎么用它来采样是自由的。DDPM 选择遵循马尔可夫链一步步走，DDIM 则"跳着走"。

**类比**：DDPM 像走楼梯，从1000楼一层层走到1楼。DDIM 像坐电梯，可以选择在哪些楼层停靠，50层就够。

---

## 2. 数学推导

### 2.1 回顾：DDPM 采样的马尔可夫约束

DDPM 的反向过程定义为马尔可夫链：

$$p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 \mathbf{I})$$

其中：

$$\mu_\theta = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)$$

$$\sigma_t^2 = \tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t} \beta_t$$

**关键限制**：因为是马尔可夫链，$x_{t-1}$ 只能从 $x_t$ 一步步推导，不能跳步。要得到 $x_0$，必须依次计算 $x_{T-1}, x_{T-2}, \ldots, x_1, x_0$。

### 2.2 DDIM 的关键思路：非马尔可夫前向过程

DDIM (Song et al., 2021) 的核心发现：**存在一族非马尔可夫的前向过程，它们的边际分布 $q(x_t|x_0)$ 与 DDPM 完全相同，但允许跳步采样**。

定义非马尔可夫前向过程：

$$q_\sigma(x_t | x_{t-1}, x_0) = \mathcal{N}\left(x_t; \sqrt{\alpha_t} \, x_{t-1}, \sqrt{1-\alpha_t - \sigma_t^2} \, \mathbf{I}\right)$$

注意与 DDPM 的区别：
- DDPM：$q(x_t|x_{t-1}) = \mathcal{N}(\sqrt{\alpha_t} x_{t-1}, \sqrt{\beta_t} \mathbf{I})$，**不依赖 $x_0$**
- DDIM：$q_\sigma(x_t|x_{t-1}, x_0)$ **依赖 $x_0$**，因此不是马尔可夫的

**为什么边际分布相同？** 可以验证，对这个非马尔可夫过程递推：

$$q_\sigma(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} \, x_0, (1-\bar{\alpha}_t) \mathbf{I})$$

与 Stage 1 推导的 DDPM 闭式解完全一致！因此**训练目标 $L_{\text{simple}}$ 不变**——同一个训练好的模型，既可以用 DDPM 采样，也可以用 DDIM 采样。

### 2.3 DDIM 反向采样公式

给定 $x_t$，DDIM 的反向步骤：

**Step 1**：用模型预测 $\epsilon_\theta(x_t, t)$，然后估计 $x_0$：

$$\hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t} \, \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}$$

**Step 2**：从 $x_t$ 跳到 $x_{t-1}$（可以是任意子序列中的前一步）：

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \, \hat{x}_0 + \sqrt{1-\bar{\alpha}_{t-1} - \sigma_t^2} \, \epsilon_\theta(x_t, t) + \sigma_t \, z$$

其中 $z \sim \mathcal{N}(0, \mathbf{I})$。

**$\sigma_t$ 的选择控制了随机性**：

| $\sigma_t$ 的值 | 采样类型 | 特点 |
|----------------|----------|------|
| $\sigma_t = 0$ | DDIM (确定性) | 完全确定，同一种子永远生成同一张图 |
| $\sigma_t = \sqrt{\tilde{\beta}_t}$ | DDPM (随机) | 等价于 DDPM 采样 |
| $0 < \sigma_t < \sqrt{\tilde{\beta}_t}$ | 插值 | 介于确定性和随机性之间 |

### 2.4 为什么 DDIM 能跳步？

DDPM 必须逐步走的原因是马尔可夫假设：$x_{t-1}$ 只依赖 $x_t$，而 $\mu_\theta$ 的公式中包含了 $\beta_t, \alpha_t, \bar{\alpha}_t$ 等当前步的特定参数。

DDIM 的公式中，$x_{t-1}$ 直接由 $\hat{x}_0$ 和 $\epsilon_\theta(x_t, t)$ 构造，**不需要中间步骤的信息**。因此我们可以选择任意子序列 $\{\tau_1, \tau_2, \ldots, \tau_S\} \subset \{1, 2, \ldots, T\}$，只在这些时间步上采样。

**子序列选择**：均匀选取 $S$ 个时间步：

$$\tau_i = \left\lfloor \frac{i \cdot T}{S} \right\rfloor, \quad i = 0, 1, \ldots, S-1$$

例如 $T=1000, S=50$：在 $t = 0, 20, 40, \ldots, 980$ 处采样，跳过中间步骤。

### 2.5 DDIM 的设计哲学

1. **训练与采样解耦**：这是 DDIM 最重要的贡献。一旦训练好 $\epsilon_\theta$，采样策略可以自由选择——步数、随机性、甚至完全不同的采样算法。这启发了后续大量工作（DPM-Solver, PNDM, Euler 等）。

2. **确定性采样的价值**：$\sigma_t = 0$ 时 DDIM 是确定性的。这意味着：
   - 同一种子 + 同一噪声 → 永远生成同一张图（可复现）
   - 可以做"插值"：在两个噪声向量之间线性插值，生成连续变化的图像
   - 可以做"编辑"：对 $x_0$ 做轻微修改，再前向加噪到 $x_t$，用 DDIM 反向得到修改后的图

3. **质量-速度权衡**：步数越多质量越好，但边际收益递减。50步通常已接近1000步的质量，10步也能生成可辨认的图像。

### 2.6 数值稳定性对比

在 Stage 2 中我们发现，cosine schedule 在大 $t$ 时 $\alpha_t \to 0$，导致 DDPM 采样公式中 $1/\sqrt{\alpha_t}$ 爆炸。

DDIM 的公式**没有 $1/\sqrt{\alpha_t}$ 项**：

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \, \hat{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}} \, \epsilon_\theta$$

因此 DDIM 在数值上更稳定，尤其适合 cosine schedule。

---

## 3. 代码对应

| 概念 | 代码位置 | 函数/类 |
|------|----------|---------|
| DDPM 逐步采样 | `ddpm_sampler.py` | `DDPMSampler` |
| DDIM 跳步采样 | `ddim_sampler.py` | `DDIMSampler` |
| $\hat{x}_0$ 估计 | `ddim_sampler.py` | `predict_x0()` |
| 子序列选择 | `ddim_sampler.py` | `get_timestep_subsequence()` |
| 对比可视化 | `compare.py` | `main()` |

---

## 4. 实验验证

```bash
python stage4_sampling/compare.py
```

**验证要点**：
- DDPM 1000步 vs DDIM 50步：生成质量对比
- DDIM 确定性：同一种子生成完全相同的图
- DDIM 数值稳定性：cosine schedule 下 DDIM 不爆炸
- 步数扫描：DDIM 10/20/50/100步的质量变化