import torch
import numpy as np


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def extract(a, t, x_shape):
    """
    从张量 a 中按时间步索引 t 提取值，并reshape到与 x 兼容的形状。

    对应公式: 在不同时间步 t 下提取 α_t, β_t 等调度值
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=2e-2):
    """
    Linear noise schedule (DDPM原论文使用).

    β_t 从 beta_start 线性增长到 beta_end.
    对应公式: β_t = β_start + (β_end - β_start) * t / T
    """
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine noise schedule (Improved DDPM, Nichol & Dhariwal 2021).

    通过累积余弦函数构造 α̅_t，再反推 β_t.
    对应公式:
        α̅_t = f(t) / f(0),  f(t) = cos((t/T + s) / (1+s) * π/2)²
        β_t = 1 - α̅_t / α̅_{t-1}
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)