# Stage 4 — 采样算法

## 核心思想

训练好的模型 $\epsilon_\theta$ 怎么用来生成数据？这就是**采样算法**。DDPM 和 DDIM 是两种核心采样方法，质量与速度的权衡是关键。

## DDPM 采样 — 逐步马尔可夫链

### 算法

从 $x_T \sim \mathcal{N}(0, I)$ 出发，**逐步**去噪到 $x_0$，共需 $T$ 步（通常1000步）：

$$\mu_\theta = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t)\right)$$

$$x_{t-1} = \mu_\theta + \sqrt{\tilde{\beta}_t} \cdot z, \quad z \sim \mathcal{N}(0, I)$$

其中后验方差：

$$\tilde{\beta}_t = \frac{\beta_t (1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}$$

### 特点

- **1000步逐步去噪**，质量最高但最慢
- 每步都有**随机噪声注入**（$\sigma_t = \sqrt{\tilde{\beta}_t}$）
- 数值上依赖 $1/\sqrt{\alpha_t}$，在 cosine schedule 大 $t$ 时可能不稳定

## DDIM 采样 — 非马尔可夫跳步

### 核心公式

DDIM 的关键洞察：不需要逐步走，可以**跳步**。先估计 $x_0$，再直接跳到任意时间步：

$$\hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t} \cdot \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}$$

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \cdot \hat{x}_0 + \sqrt{1-\bar{\alpha}_{t-1} - \sigma_t^2} \cdot \epsilon_\theta + \sigma_t \cdot z$$

### $\eta$ 参数 — 控制随机性

$$\sigma_t = \eta \cdot \sqrt{\tilde{\beta}_t}$$

- $\eta = 0$：**确定性采样**，同一种子永远生成同一张图（DDIM默认）
- $\eta = 1$：等价于 DDPM（随机采样）

### 特点

- **可跳步采样**，50步即可接近1000步DDPM的质量
- $\eta=0$ 时**确定性采样**，同种子可复现
- **数值稳定**，无 $1/\sqrt{\alpha_t}$ 项，适合 cosine schedule

## 对比分析

| 特性 | DDPM | DDIM |
|------|------|------|
| 步数 | 1000 | 50（可调） |
| 随机性 | 每步注入噪声 | $\eta$ 控制 |
| 确定性 | 不可复现 | $\eta=0$ 可复现 |
| 数值稳定性 | $1/\sqrt{\alpha_t}$ 可能不稳定 | 稳定 |
| 质量 | 最高 | 接近DDPM |

## 代码实现

### `ddpm_sampler.py`

```python
class DDPMSampler:
    @torch.no_grad()
    def step(self, model, x_t, t_index):
        """单步反向采样: x_t → x_{t-1}"""
        eps_theta = model(x_t, t)
        # β̃_t = β_t · (1-ᾱ_{t-1}) / (1-ᾱ_t)
        posterior_var = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
        # μ_θ = 1/√α_t · (x_t - β_t/√(1-ᾱ_t) · ε_θ)
        mean = (1.0 / torch.sqrt(alpha_t)) * (
            x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * eps_theta
        )
        if t_index > 0:
            sigma = torch.sqrt(posterior_var)
            z = torch.randn_like(x_t)
            return mean + sigma * z
        return mean
```

### `ddim_sampler.py`

```python
class DDIMSampler:
    @torch.no_grad()
    def step(self, model, x_t, t_index, prev_t_index, eta=0.0):
        """DDIM 跳步采样: x_t → x_{prev_t_index}"""
        x_0_pred, eps_theta = self.predict_x0(model, x_t, t_index)
        # σ_t = η · √(β̃_t)
        sigma = eta * torch.sqrt(...)
        # x_{prev} = √ᾱ_{prev} · x̂_0 + √(1-ᾱ_{prev} - σ_t²) · ε_θ + σ_t · z
        mean = torch.sqrt(alpha_bar_prev) * x_0_pred
        dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma**2) * eps_theta
        if eta > 0:
            return mean + dir_xt + sigma * z
        return mean + dir_xt
```

## 运行

```bash
python stage4_sampling/compare.py
```

对比 DDPM 1000步 vs DDIM 50步的生成质量和速度。

## 关键文件

| 文件 | 说明 |
|------|------|
| `stage4_sampling/ddpm_sampler.py` | `DDPMSampler` — 逐步马尔可夫链采样 |
| `stage4_sampling/ddim_sampler.py` | `DDIMSampler` — 跳步非马尔可夫采样 |
| `stage4_sampling/compare.py` | DDPM vs DDIM 对比实验 |