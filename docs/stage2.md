# Stage 2 — 训练目标

## 核心思想

扩散模型的训练目标是学习**反向过程** $p_\theta(x_{t-1} | x_t)$——给定带噪数据 $x_t$，预测上一步 $x_{t-1}$。但直接建模 $p_\theta(x_{t-1} | x_t)$ 的分布很难，DDPM 的关键洞察是：**模型不需要预测分布，只需要预测噪声**。

## 数学推导

### ELBO 分解

扩散模型的变分下界（ELBO）可以分解为：

$$L = L_0 + \sum_{t=1}^{T} L_t + L_T$$

其中：

- $L_T$：KL散度项，$x_T$ 是纯噪声，与模型无关
- $L_0$：重建项，$-\log p_\theta(x_0 | x_1)$
- $L_t$：去噪匹配项，KL散度 $D_{KL}(q(x_{t-1}|x_t,x_0) \| p_\theta(x_{t-1}|x_t))$

### $L_t$ 的简化

$q(x_{t-1}|x_t,x_0)$ 是已知的高斯分布（后验）：

$$q(x_{t-1}|x_t,x_0) = \mathcal{N}(\tilde{\mu}_t, \tilde{\beta}_t I)$$

$$\tilde{\mu}_t = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon\right)$$

$$\tilde{\beta}_t = \frac{\beta_t (1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}$$

模型 $p_\theta$ 也参数化为高斯：

$$p_\theta(x_{t-1}|x_t) = \mathcal{N}(\mu_\theta(x_t, t), \sigma_t^2 I)$$

两个高斯的 KL 散度只取决于均值差：

$$L_t = \frac{1}{2\sigma_t^2} \|\tilde{\mu}_t - \mu_\theta\|^2$$

### $L_{simple}$ — DDPM 的最终目标

将 $\mu_\theta$ 参数化为预测噪声 $\epsilon_\theta$：

$$\mu_\theta = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t)\right)$$

代入 $L_t$ 后，系数项被丢弃，得到简化目标：

$$L_{simple} = \mathbb{E}_{t, x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon, t)\|^2\right]$$

**这就是训练的全部目标——让网络从 $x_t$ 中预测出加入的噪声 $\epsilon$。**

## 反向采样 $p_\theta(x_{t-1}|x_t)$

训练好后，用 $\epsilon_\theta$ 计算 $\mu_\theta$，然后采样：

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t)\right) + \sigma_t z, \quad z \sim \mathcal{N}(0, I)$$

## 代码实现

### `simple_model.py` — MLP 模型

```python
class SinusoidalTimeEmbedding(nn.Module):
    """正弦时间嵌入: 把整数 t 映射到连续向量"""
    def forward(self, t):
        half_dim = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half_dim) / half_dim)
        args = t[:, None].float() * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb

class DiffusionMLP(nn.Module):
    """ε_θ(x_t, t): 输入 (x_t, t), 输出预测噪声"""
    def __init__(self, data_dim=2, hidden_dim=128, time_emb_dim=64):
        self.time_embedding = SinusoidalTimeEmbedding(time_emb_dim)
        self.net = nn.Sequential(
            nn.Linear(data_dim + time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, data_dim),
        )

    def forward(self, x_t, t):
        t_emb = self.time_embedding(t)
        x = torch.cat([x_t, t_emb], dim=-1)
        return self.net(x)
```

### `train.py` — 训练循环

```python
def train_step(model, schedule, x_0, optimizer):
    """L_simple = E[||ε - ε_θ(√ᾱ_t·x_0 + √(1-ᾱ_t)·ε, t)||²]"""
    batch_size = x_0.shape[0]
    t = torch.randint(0, schedule.timesteps, (batch_size,))
    noise = torch.randn_like(x_0)
    x_t = q_sample(schedule, x_0, t, noise)
    predicted_noise = model(x_t, t)
    loss = nn.functional.mse_loss(predicted_noise, noise)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def p_sample(model, schedule, x_t, t_index):
    """反向采样一步: μ_θ + σ_t · z"""
    eps_theta = model(x_t, t)
    mean = (1.0 / torch.sqrt(alpha_t)) * (
        x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * eps_theta
    )
    if t_index > 0:
        sigma = torch.sqrt(beta_t)
        z = torch.randn_like(x_t)
        return mean + sigma * z
    return mean
```

### `toy_dataset.py` — 2D玩具数据

三种2D分布用于验证训练逻辑：

- **Two Moons** — 两个交叠月牙，扩散模型论文常用
- **Swiss Roll** — 螺旋形分布
- **Gaussian Mixture** — 圆上均匀分布的8个高斯簇

## 运行

```bash
python stage2_training/train.py
```

训练5000轮后，模型能从2D高斯噪声恢复出目标分布的形状。

## 关键文件

| 文件 | 说明 |
|------|------|
| `stage2_training/simple_model.py` | `DiffusionMLP`, `SinusoidalTimeEmbedding` |
| `stage2_training/train.py` | `train_step()`, `p_sample()`, `p_sample_loop()` |
| `stage2_training/toy_dataset.py` | 2D玩具数据集生成 |
| `stage2_training/visualize.py` | 训练结果可视化 |