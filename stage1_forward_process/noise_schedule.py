"""
噪声调度表 (Noise Schedule)

实现两种调度策略:
1. Linear schedule (DDPM原论文)
2. Cosine schedule (Improved DDPM)

核心概念:
  β_t:   每一步添加的噪声量
  α_t:   = 1 - β_t, 每一步保留的信号比例
  ᾱ_t:  = ∏_{s=1}^{t} α_s, 累积保留的信号比例

关键公式:
  q(x_t | x_0) = N(x_t; √ᾱ_t · x_0, (1-ᾱ_t)·I)
  即: x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε,  ε ~ N(0, I)
"""

import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.math_utils import linear_beta_schedule, cosine_beta_schedule


class NoiseSchedule:
    """
    噪声调度表: 管理 β_t, α_t, ᾱ_t 及其衍生量.

    所有张量都预计算并缓存, 训练时只需按时间步索引提取.
    """

    def __init__(self, timesteps=1000, schedule_type='linear',
                 beta_start=1e-4, beta_end=2e-2, cosine_s=0.008):
        self.timesteps = timesteps
        self.schedule_type = schedule_type

        if schedule_type == 'linear':
            self.betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        elif schedule_type == 'cosine':
            self.betas = cosine_beta_schedule(timesteps, cosine_s)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        # α_t = 1 - β_t
        self.alphas = 1.0 - self.betas

        # ᾱ_t = ∏_{s=1}^{t} α_s  (累积乘积)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # ᾱ_{t-1}: 用于逆向过程, 在t=0处补1.0
        self.alphas_cumprod_prev = torch.cat([
            torch.tensor([1.0]),
            self.alphas_cumprod[:-1]
        ])

        # √ᾱ_t 和 √(1-ᾱ_t): 用于 q(x_t|x_0) 闭式采样
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # 逆向过程所需量 (Stage 3/4 会用到, 这里一并预计算)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # posterior q(x_{t-1} | x_t, x_0) 的方差 (Stage 2 会推导)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def get(self, tensor, t, x_shape):
        """
        按时间步 t 从预计算张量中提取值, 并reshape到与 x 兼容.

        例如: sqrt_alphas_cumprod[t] 提取出来后 reshape 为 (B, 1, 1, 1) 以便与 (B, C, H, W) 的 x 广播.
        """
        b = t.shape[0]
        out = tensor.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    def __len__(self):
        return self.timesteps


def get_schedule(timesteps=1000, schedule_type='linear'):
    """便捷构造函数."""
    return NoiseSchedule(timesteps=timesteps, schedule_type=schedule_type)


if __name__ == '__main__':
    print("=== Linear Schedule ===")
    sched = NoiseSchedule(timesteps=1000, schedule_type='linear')
    print(f"β_0 = {sched.betas[0]:.6f}, β_T = {sched.betas[-1]:.6f}")
    print(f"α_0 = {sched.alphas[0]:.6f}, α_T = {sched.alphas[-1]:.6f}")
    print(f"ᾱ_0 = {sched.alphas_cumprod[0]:.6f}, ᾱ_T = {sched.alphas_cumprod[-1]:.6f}")
    print(f"√ᾱ_0 = {sched.sqrt_alphas_cumprod[0]:.6f}, √ᾱ_T = {sched.sqrt_alphas_cumprod[-1]:.6f}")

    print("\n=== Cosine Schedule ===")
    sched_c = NoiseSchedule(timesteps=1000, schedule_type='cosine')
    print(f"β_0 = {sched_c.betas[0]:.6f}, β_T = {sched_c.betas[-1]:.6f}")
    print(f"ᾱ_0 = {sched_c.alphas_cumprod[0]:.6f}, ᾱ_T = {sched_c.alphas_cumprod[-1]:.6f}")