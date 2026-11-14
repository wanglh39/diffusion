# Stage 1 — 前向扩散过程

## 核心思想

扩散模型的前向过程（Forward Process）是一个**逐步加噪**的马尔可夫链：从原始数据 $x_0$ 出发，经过 $T$ 步逐渐添加高斯噪声，最终变成纯噪声 $x_T \sim \mathcal{N}(0, I)$。

这个过程是**确定性的**（给定 $x_0$ 和噪声序列，每一步的输出都确定），它的目的是为反向过程提供一个训练目标——如果模型能学会"去噪"，就能从噪声生成数据。

## 数学框架

### 噪声调度表 (Noise Schedule)

每一步添加的噪声量由 $\beta_t$ 控制：

$$\beta_t \in (0, 1), \quad t = 1, 2, \ldots, T$$

由此派生出两个关键量：

$$\alpha_t = 1 - \beta_t \quad \text{(每步保留的信号比例)}$$

$$\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s \quad \text{(累积保留的信号比例)}$$

### 闭式解 $q(x_t | x_0)$

最关键的公式——不需要逐步加噪，可以**一步**从 $x_0$ 跳到任意 $x_t$：

$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} \cdot x_0, (1-\bar{\alpha}_t) \cdot I)$$

即：

$$x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1-\bar{\alpha}_t} \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

### 逐步加噪 $q(x_t | x_{t-1})$

单步转移公式：

$$x_t = \sqrt{\alpha_t} \cdot x_{t-1} + \sqrt{1-\alpha_t} \cdot \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I)$$

## 两种调度策略

### Linear Schedule (DDPM原论文)

$$\beta_t = \beta_{start} + (\beta_{end} - \beta_{start}) \cdot \frac{t}{T}$$

- $\beta_{start} = 10^{-4}$, $\beta_{end} = 0.02$
- 简单直观，但末端 $\bar{\alpha}_t$ 下降太快

### Cosine Schedule (Improved DDPM)

$$\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\left(\frac{t/T + s}{1+s} \cdot \frac{\pi}{2}\right)^2$$

- $s = 0.008$，防止 $t=0$ 时 $\bar{\alpha}_t$ 变化太快
- 信号衰减更平缓，训练更稳定

## 代码实现

### `noise_schedule.py` — 噪声调度表

```python
class NoiseSchedule:
    def __init__(self, timesteps=1000, schedule_type='linear',
                 beta_start=1e-4, beta_end=2e-2, cosine_s=0.008):
        # 计算 β_t
        if schedule_type == 'linear':
            self.betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        elif schedule_type == 'cosine':
            self.betas = cosine_beta_schedule(timesteps, cosine_s)

        # α_t = 1 - β_t
        self.alphas = 1.0 - self.betas

        # ᾱ_t = ∏ α_s (累积乘积)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # √ᾱ_t 和 √(1-ᾱ_t): 用于闭式采样
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
```

### `forward_diffusion.py` — 前向采样

```python
def q_sample(schedule, x_0, t, noise=None):
    """闭式采样: x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε"""
    if noise is None:
        noise = torch.randn_like(x_0)
    sqrt_alpha = schedule.get(schedule.sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alpha = schedule.get(schedule.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
    return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
```

## 可视化

运行 `python stage1_forward_process/visualize.py` 生成：

1. **调度表对比** — Linear vs Cosine 的 $\beta_t$, $\bar{\alpha}_t$, $\sqrt{1-\bar{\alpha}_t}$ 曲线
2. **前向加噪轨迹** — MNIST数字从 $t=0$ 到 $t=999$ 的逐步加噪过程
3. **闭式解 vs 逐步加噪验证** — 统计一致性验证

## 关键文件

| 文件 | 说明 |
|------|------|
| `stage1_forward_process/noise_schedule.py` | `NoiseSchedule` 类，管理 $\beta_t, \alpha_t, \bar{\alpha}_t$ |
| `stage1_forward_process/forward_diffusion.py` | `q_sample()`, `q_sample_step()`, `q_sample_chain()` |
| `stage1_forward_process/visualize.py` | 3组可视化 |
| `utils/math_utils.py` | `linear_beta_schedule()`, `cosine_beta_schedule()` |