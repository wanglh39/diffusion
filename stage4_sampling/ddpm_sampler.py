"""
DDPM 采样器: 逐步马尔可夫链采样

从 x_T ~ N(0,I) 逐步去噪到 x_0, 每一步都走.
共需 T 步神经网络前向传播.

公式:
  μ_θ = 1/√α_t · (x_t - β_t/√(1-ᾱ_t) · ε_θ(x_t, t))
  x_{t-1} = μ_θ + √(β̃_t) · z,  z ~ N(0, I)
  β̃_t = β_t · (1-ᾱ_{t-1}) / (1-ᾱ_t)
"""

import torch
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class DDPMSampler:
    """
    DDPM 采样器: 严格遵循马尔可夫链, 必须逐步采样.

    特点:
      - 1000步逐步去噪, 质量最高但最慢
      - 每步都有随机噪声注入 (σ_t = √β̃_t)
      - 数值上依赖 1/√α_t, 在 cosine schedule 大t时可能不稳定
    """
    def __init__(self, schedule):
        self.schedule = schedule
        self.timesteps = schedule.timesteps

    @torch.no_grad()
    def step(self, model, x_t, t_index):
        """单步反向采样: x_t → x_{t-1}"""
        t = torch.full((x_t.shape[0],), t_index, dtype=torch.long, device=x_t.device)
        eps_theta = model(x_t, t)

        beta_t = self.schedule.betas[t_index]
        alpha_t = self.schedule.alphas[t_index]
        alpha_bar_t = self.schedule.alphas_cumprod[t_index]
        alpha_bar_prev = self.schedule.alphas_cumprod_prev[t_index]

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
        else:
            return mean

    @torch.no_grad()
    def sample(self, model, shape, device='cpu', save_trajectory=False,
               trajectory_steps=None):
        """
        完整采样: x_T → x_0, 共 T 步.

        Args:
            model: 训练好的 ε_θ 模型
            shape: 生成样本的形状, 如 (64, 1, 28, 28)
            save_trajectory: 是否保存中间快照
            trajectory_steps: 保存快照的时间步集合
        """
        x = torch.randn(shape, device=device)
        trajectory = {}

        for t_index in reversed(range(self.timesteps)):
            x = self.step(model, x, t_index)
            if save_trajectory and trajectory_steps and t_index in trajectory_steps:
                trajectory[t_index] = x.clone()

        if save_trajectory:
            trajectory[0] = x.clone()
            return x, trajectory
        return x


if __name__ == '__main__':
    from stage1_forward_process.noise_schedule import NoiseSchedule
    from stage2_training.simple_model import DiffusionMLP

    schedule = NoiseSchedule(timesteps=100, schedule_type='linear')
    model = DiffusionMLP(data_dim=2, hidden_dim=64, time_emb_dim=32)
    sampler = DDPMSampler(schedule)

    x = sampler.sample(model, (4, 2))
    print(f"DDPM sample shape: {x.shape}")
    print(f"Total steps: {schedule.timesteps}")