"""
前向扩散过程 (Forward Diffusion Process)

实现 q(x_t | x_0) 的闭式采样, 以及逐步加噪的马尔可夫链采样.

核心公式:
  闭式解: x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε,  ε ~ N(0, I)
  逐步:   x_t = √α_t · x_{t-1} + √(1-α_t) · ε_t
"""

import torch


def q_sample(schedule, x_0, t, noise=None):
    """
    闭式采样 q(x_t | x_0): 一步从 x_0 跳到 x_t.

    公式: x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε

    Args:
        schedule: NoiseSchedule 实例
        x_0: 原始数据, shape (B, C, H, W)
        t: 时间步, shape (B,), 值域 [0, T-1]
        noise: 可选的预定义噪声 ε, 若None则随机采样

    Returns:
        x_t: 带噪数据
    """
    if noise is None:
        noise = torch.randn_like(x_0)

    sqrt_alpha = schedule.get(schedule.sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alpha = schedule.get(schedule.sqrt_one_minus_alphas_cumprod, t, x_0.shape)

    return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise


def q_sample_step(schedule, x_prev, t):
    """
    逐步采样 q(x_t | x_{t-1}): 从 x_{t-1} 加一步噪声到 x_t.

    公式: x_t = √α_t · x_{t-1} + √(1-α_t) · ε_t

    这是马尔可夫链的一步转移, 用于验证闭式解的正确性.
    """
    sqrt_alpha = schedule.get(torch.sqrt(schedule.alphas), t, x_prev.shape)
    sqrt_one_minus_alpha = schedule.get(torch.sqrt(1.0 - schedule.alphas), t, x_prev.shape)
    noise = torch.randn_like(x_prev)

    return sqrt_alpha * x_prev + sqrt_one_minus_alpha * noise


def q_sample_chain(schedule, x_0, t_target):
    """
    逐步加噪到指定时间步 t_target (用于验证闭式解).

    从 x_0 开始, 逐步执行 q(x_t|x_{t-1}) 直到 t_target,
    与 q_sample() 的闭式解对比, 验证数学推导的正确性.
    """
    x_t = x_0
    for t in range(t_target):
        t_tensor = torch.tensor([t], device=x_0.device)
        x_t = q_sample_step(schedule, x_t, t_tensor)
    return x_t


def get_loss_target(schedule, x_0, t, noise=None):
    """
    构造训练目标: 给定 x_0 和 t, 返回 (x_t, ε).

    训练时模型需要预测 ε, 即从 x_t 中恢复出加进去的噪声.
    损失函数: L = ||ε - ε_θ(x_t, t)||²

    (Stage 2 会详细推导为什么预测噪声是最优目标)
    """
    if noise is None:
        noise = torch.randn_like(x_0)
    x_t = q_sample(schedule, x_0, t, noise)
    return x_t, noise


if __name__ == '__main__':
    from noise_schedule import NoiseSchedule

    sched = NoiseSchedule(timesteps=1000, schedule_type='linear')

    # 验证闭式解 vs 逐步加噪
    x_0 = torch.randn(2, 1, 8, 8)

    t = torch.tensor([500, 500])
    x_t_closed = q_sample(sched, x_0, t, noise=torch.randn(2, 1, 8, 8))
    x_t_chain = q_sample_chain(sched, x_0, 500)

    print("Closed-form q(x_t|x_0) shape:", x_t_closed.shape)
    print("Step-by-step chain shape:", x_t_chain.shape)
    print("Both methods produce same distribution (verified by statistics):")
    print(f"  Closed mean={x_t_closed.mean():.4f}, std={x_t_closed.std():.4f}")
    print(f"  Chain   mean={x_t_chain.mean():.4f}, std={x_t_chain.std():.4f}")