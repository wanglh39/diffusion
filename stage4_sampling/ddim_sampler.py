"""
DDIM 采样器: 非马尔可夫跳步采样

从 x_T ~ N(0,I) 跳步去噪到 x_0, 只需 S << T 步.
核心公式:
  x̂_0 = (x_t - √(1-ᾱ_t) · ε_θ(x_t, t)) / √ᾱ_t
  x_{t-1} = √ᾱ_{t-1} · x̂_0 + √(1-ᾱ_{t-1} - σ_t²) · ε_θ(x_t, t) + σ_t · z

当 σ_t = 0 时为确定性采样 (DDIM默认), 同一种子永远生成同一张图.
当 σ_t = √(β̃_t) 时退化为 DDPM.
"""

import torch
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def get_timestep_subsequence(timesteps, num_steps):
    """
    均匀选取 num_steps 个时间步作为子序列.

    例如 timesteps=1000, num_steps=50:
      返回 [0, 20, 40, ..., 980], 共50个时间步
    """
    step_ratio = timesteps / num_steps
    timesteps_seq = (np.arange(0, num_steps) * step_ratio).round().astype(int)
    timesteps_seq = np.clip(timesteps_seq, 0, timesteps - 1)
    return timesteps_seq


class DDIMSampler:
    """
    DDIM 采样器: 非马尔可夫跳步采样.

    特点:
      - 可跳步采样, 50步即可接近1000步DDPM的质量
      - σ_t=0 时确定性采样, 同种子可复现
      - 数值稳定, 无 1/√α_t 项, 适合 cosine schedule
    """
    def __init__(self, schedule):
        self.schedule = schedule
        self.timesteps = schedule.timesteps

    @torch.no_grad()
    def predict_x0(self, model, x_t, t_index):
        """
        从 x_t 估计 x̂_0:
          x̂_0 = (x_t - √(1-ᾱ_t) · ε_θ(x_t, t)) / √ᾱ_t
        """
        t = torch.full((x_t.shape[0],), t_index, dtype=torch.long, device=x_t.device)
        eps_theta = model(x_t, t)

        alpha_bar_t = self.schedule.alphas_cumprod[t_index]
        sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar_t)

        x_0_pred = (x_t - sqrt_one_minus_alpha_bar * eps_theta) / sqrt_alpha_bar
        return x_0_pred, eps_theta

    @torch.no_grad()
    def step(self, model, x_t, t_index, prev_t_index, eta=0.0):
        """
        DDIM 单步反向采样: x_t → x_{prev_t_index}

        Args:
            t_index: 当前时间步
            prev_t_index: 下一个(更小的)时间步, 可以跳步
            eta: 控制随机性, 0=确定性DDIM, 1=DDPM

        公式:
          x_{prev} = √ᾱ_{prev} · x̂_0 + √(1-ᾱ_{prev} - σ_t²) · ε_θ + σ_t · z
        """
        x_0_pred, eps_theta = self.predict_x0(model, x_t, t_index)

        alpha_bar_t = self.schedule.alphas_cumprod[t_index]
        alpha_bar_prev = self.schedule.alphas_cumprod[prev_t_index] if prev_t_index >= 0 else torch.tensor(1.0)

        # σ_t = η · √(β̃_t), 其中 β̃_t = (1-ᾱ_{prev})/(1-ᾱ_t) · (1-ᾱ_t/ᾱ_{prev})
        # 当 η=0: σ_t=0, 确定性采样
        # 当 η=1: σ_t=√β̃_t, 等价于DDPM
        sigma = eta * torch.sqrt(
            (1 - alpha_bar_prev) / (1 - alpha_bar_t) *
            (1 - alpha_bar_t / alpha_bar_prev)
        ) if prev_t_index >= 0 else torch.tensor(0.0)

        # 方向项: √(1-ᾱ_{prev} - σ_t²) · ε_θ
        dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps_theta

        # 均值项: √ᾱ_{prev} · x̂_0
        mean = torch.sqrt(alpha_bar_prev) * x_0_pred

        if eta > 0 and prev_t_index >= 0:
            z = torch.randn_like(x_t)
            return mean + dir_xt + sigma * z
        else:
            return mean + dir_xt

    @torch.no_grad()
    def sample(self, model, shape, num_steps=50, eta=0.0, device='cpu',
               save_trajectory=False):
        """
        DDIM 跳步采样: x_T → x_0, 共 num_steps 步.

        Args:
            num_steps: 采样步数 (远小于 timesteps)
            eta: 随机性参数, 0=确定性, 1=DDPM
        """
        timesteps_seq = get_timestep_subsequence(self.timesteps, num_steps)

        x = torch.randn(shape, device=device)
        trajectory = {}

        for i in reversed(range(len(timesteps_seq))):
            t_index = int(timesteps_seq[i])
            prev_t_index = int(timesteps_seq[i - 1]) if i > 0 else -1

            x = self.step(model, x, t_index, prev_t_index, eta=eta)

            if save_trajectory:
                trajectory[t_index] = x.clone()

        if save_trajectory:
            return x, trajectory
        return x


if __name__ == '__main__':
    from stage1_forward_process.noise_schedule import NoiseSchedule
    from stage2_training.simple_model import DiffusionMLP

    schedule = NoiseSchedule(timesteps=1000, schedule_type='linear')
    model = DiffusionMLP(data_dim=2, hidden_dim=64, time_emb_dim=32)
    sampler = DDIMSampler(schedule)

    # 50步 DDIM 采样
    x = sampler.sample(model, (4, 2), num_steps=50, eta=0.0)
    print(f"DDIM sample shape: {x.shape}")
    print(f"DDIM steps: 50 (vs DDPM 1000)")

    # 确定性验证: eta=0 时, 相同初始噪声 → 结果完全相同
    init_noise = torch.randn(4, 2)
    x1 = sampler.sample(model, (4, 2), num_steps=50, eta=0.0)
    x2 = sampler.sample(model, (4, 2), num_steps=50, eta=0.0)
    print(f"DDIM sample shape: {x.shape}")
    print(f"DDIM steps: 50 (vs DDPM 1000)")
    print(f"Deterministic (eta=0): reproducible = {torch.allclose(x1, x2)}")

    # 随机性验证: eta=1 时, 不同种子 → 结果不同
    torch.manual_seed(42)
    x3 = sampler.sample(model, (4, 2), num_steps=50, eta=1.0)
    torch.manual_seed(999)
    x4 = sampler.sample(model, (4, 2), num_steps=50, eta=1.0)
    print(f"Stochastic (eta=1): different = {not torch.allclose(x3, x4)}")