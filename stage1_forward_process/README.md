# Stage 1: 前向扩散过程

## 1. 直觉先行

想象你有一张清晰的照片。你往上面撒一点沙子（噪声），照片变得稍微模糊。再撒一点，更模糊。不断重复，直到照片完全变成一片均匀的沙地——纯噪声。

这就是**前向扩散过程**（Forward Diffusion Process）：一个把清晰数据逐步摧毁成噪声的过程。

为什么我们要研究"摧毁"？因为如果我们能精确理解噪声是如何一步步加上去的，就能训练一个模型**逆向**这个过程——从噪声中一步步恢复出清晰图像。这就是扩散模型的核心思想。

**关键洞察**：前向过程不需要学习，它是完全确定的数学操作。我们只需要定义好"怎么加噪声"，反向过程才是需要神经网络学习的部分。

---

## 2. 数学推导

### 2.1 逐步加噪：马尔可夫链

定义数据 $x_0 \sim q(x_0)$（真实数据分布），前向过程是一个马尔可夫链：

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} \, x_{t-1}, \beta_t \mathbf{I})$$

其中 $\beta_t \in (0, 1)$ 是第 $t$ 步的**噪声调度值**（noise schedule），控制每一步加多少噪声。

**解读**：
- 均值 $\sqrt{1-\beta_t} \, x_{t-1}$：对前一步数据做轻微缩放（保留大部分信号）
- 方差 $\beta_t \mathbf{I}$：加上一个小的高斯噪声
- $\beta_t$ 越大，加的噪声越多，信号保留越少

定义 $\alpha_t = 1 - \beta_t$，则：

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{\alpha_t} \, x_{t-1}, (1 - \alpha_t) \mathbf{I})$$

### 2.2 一步到位：闭式解 $q(x_t | x_0)$

逐步采样 $T$ 次才能得到 $x_T$ 太慢了。幸运的是，由于高斯分布的可加性，我们可以**直接从 $x_0$ 跳到任意 $x_t$**。

**推导过程**：

从 $x_{t-1}$ 的条件分布出发，利用重参数化技巧递推：

$$x_t = \sqrt{\alpha_t} \, x_{t-1} + \sqrt{1 - \alpha_t} \, \epsilon_{t-1}, \quad \epsilon_{t-1} \sim \mathcal{N}(0, \mathbf{I})$$

将 $x_{t-1}$ 展开：

$$x_t = \sqrt{\alpha_t} \left( \sqrt{\alpha_{t-1}} \, x_{t-2} + \sqrt{1-\alpha_{t-1}} \, \epsilon_{t-2} \right) + \sqrt{1-\alpha_t} \, \epsilon_{t-1}$$

$$= \sqrt{\alpha_t \alpha_{t-1}} \, x_{t-2} + \sqrt{\alpha_t(1-\alpha_{t-1})} \, \epsilon_{t-2} + \sqrt{1-\alpha_t} \, \epsilon_{t-1}$$

关键步骤：两个独立高斯噪声的合并。若 $\epsilon_1 \sim \mathcal{N}(0, \sigma_1^2 \mathbf{I})$，$\epsilon_2 \sim \mathcal{N}(0, \sigma_2^2 \mathbf{I})$ 独立，则：

$$\sigma_1 \epsilon_1 + \sigma_2 \epsilon_2 \sim \mathcal{N}(0, (\sigma_1^2 + \sigma_2^2) \mathbf{I})$$

因此：

$$\sqrt{\alpha_t(1-\alpha_{t-1})} \, \epsilon_{t-2} + \sqrt{1-\alpha_t} \, \epsilon_{t-1} \sim \mathcal{N}\left(0, \left[\alpha_t(1-\alpha_{t-1}) + (1-\alpha_t)\right] \mathbf{I}\right)$$

化简方差：

$$\alpha_t(1-\alpha_{t-1}) + (1-\alpha_t) = \alpha_t - \alpha_t\alpha_{t-1} + 1 - \alpha_t = 1 - \alpha_t\alpha_{t-1}$$

所以：

$$x_t = \sqrt{\alpha_t\alpha_{t-1}} \, x_{t-2} + \sqrt{1 - \alpha_t\alpha_{t-1}} \, \bar{\epsilon}$$

其中 $\bar{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$。

**递推到 $x_0$**：定义累积乘积 $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$，重复上述递推：

$$\boxed{q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} \, x_0, (1 - \bar{\alpha}_t) \mathbf{I})}$$

即：

$$x_t = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{1 - \bar{\alpha}_t} \, \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

**这就是扩散模型最重要的公式之一**。它意味着：
- 给定原始数据 $x_0$，我们可以一步生成任意时间步 $t$ 的带噪版本
- 训练时不需要逐步模拟前向过程，直接采样即可
- 当 $\bar{\alpha}_t \to 0$（即 $t \to T$），$x_t \to \mathcal{N}(0, \mathbf{I})$，纯噪声

### 2.3 噪声调度表

$\beta_t$ 的选择决定了前向过程的"节奏"——加噪太快还是太慢。

**Linear Schedule**（DDPM原论文）：

$$\beta_t = \beta_{\text{start}} + \frac{t}{T}(\beta_{\text{end}} - \beta_{\text{start}})$$

典型值：$\beta_{\text{start}} = 10^{-4}$，$\beta_{\text{end}} = 0.02$，$T = 1000$。

**问题**：线性调度在最后几步 $\bar{\alpha}_t$ 已经接近0，信息几乎完全摧毁，但中间步骤的信号衰减不均匀。

**Cosine Schedule**（Improved DDPM, Nichol & Dhariwal 2021）：

$$\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\left(\frac{t/T + s}{1+s} \cdot \frac{\pi}{2}\right)^2$$

其中 $s = 0.008$ 是偏移量。然后反推：

$$\beta_t = 1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}$$

**优势**：余弦调度让 $\bar{\alpha}_t$ 的衰减更平滑、更均匀，在中间步骤保留更多信号，训练效果更好。

### 2.4 设计哲学：为什么是高斯噪声？

1. **数学便利性**：高斯分布的线性变换仍是高斯分布，这使得闭式解 $q(x_t|x_0)$ 成为可能。如果是其他分布（如均匀分布），无法得到如此简洁的递推。

2. **信息论最优**：在给定方差约束下，高斯分布是熵最大的分布。加高斯噪声等价于以最"公平"的方式摧毁信息——不偏袒任何特征。

3. **逆向可行**：高斯分布的逆向也是高斯分布，这让我们可以用神经网络参数化逆向过程 $p_\theta(x_{t-1}|x_t)$，且只需预测均值和方差。

4. **与score matching的联系**：高斯噪声的score function $\nabla_x \log q(x_t|x_0)$ 有解析形式，这连接了扩散模型与score-based generative models。

---

## 3. 代码对应

| 公式 | 代码位置 | 函数 |
|------|----------|------|
| $\beta_t$ 线性调度 | `noise_schedule.py` | `get_linear_schedule()` |
| $\beta_t$ 余弦调度 | `noise_schedule.py` | `get_cosine_schedule()` |
| $\alpha_t = 1 - \beta_t$ | `noise_schedule.py` | `get_alphas()` |
| $\bar{\alpha}_t = \prod \alpha_s$ | `noise_schedule.py` | `get_alphas_cumprod()` |
| $q(x_t\|x_0)$ 闭式采样 | `forward_diffusion.py` | `q_sample()` |
| 可视化加噪轨迹 | `visualize.py` | `main()` |

---

## 4. 实验验证

运行可视化脚本：

```bash
python stage1_forward_process/visualize.py
```

你将看到：
1. **噪声调度表曲线**：$\beta_t$, $\alpha_t$, $\bar{\alpha}_t$ 随时间步的变化
2. **加噪轨迹**：一张MNIST数字从清晰到纯噪声的逐步演变
3. **一步采样验证**：直接用闭式解 $q(x_t|x_0)$ 跳到不同时间步，与逐步加噪对比

**验证要点**：
- $\bar{\alpha}_t$ 应从1（$t=0$，无噪声）单调递减到接近0（$t=T$，纯噪声）
- 闭式解一步采样的结果应与逐步加噪的结果统计一致
- 在 $t$ 较小时，图像应仍可辨认；$t$ 较大时，应完全不可辨认