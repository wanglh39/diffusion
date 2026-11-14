"""
生成采样: 用训练好的 U-Net 从纯噪声生成 MNIST 数字

实现 DDPM 反向采样 p_sample_loop:
  从 x_T ~ N(0, I) 开始, 逐步去噪到 x_0.

运行: python stage3_unet/sample.py
"""

import torch
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from stage1_forward_process.noise_schedule import NoiseSchedule
from stage3_unet.unet import UNet

import matplotlib.pyplot as plt


@torch.no_grad()
def p_sample(model, schedule, x_t, t_index):
    """
    反向采样一步: p_θ(x_{t-1}|x_t)

    μ_θ = 1/√α_t · (x_t - β_t/√(1-ᾱ_t) · ε_θ(x_t, t))
    x_{t-1} = μ_θ + σ_t · z,  z ~ N(0, I),  σ_t = √β_t
    """
    t = torch.full((x_t.shape[0],), t_index, dtype=torch.long, device=x_t.device)
    eps_theta = model(x_t, t)

    beta_t = schedule.betas[t_index]
    alpha_t = schedule.alphas[t_index]
    alpha_bar_t = schedule.alphas_cumprod[t_index]

    mean = (1.0 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * eps_theta)

    if t_index > 0:
        sigma = torch.sqrt(beta_t)
        z = torch.randn_like(x_t)
        return mean + sigma * z
    else:
        return mean


@torch.no_grad()
def p_sample_loop(model, schedule, shape, device='cpu', save_trajectory=False,
                  trajectory_steps=None):
    """
    完整反向采样: x_T ~ N(0,I) → x_0

    如果 save_trajectory=True, 在 trajectory_steps 指定的时间步保存快照.
    """
    x = torch.randn(shape, device=device)
    trajectory = {}

    for t_index in reversed(range(schedule.timesteps)):
        x = p_sample(model, schedule, x, t_index)
        if save_trajectory and (trajectory_steps is not None) and t_index in trajectory_steps:
            trajectory[t_index] = x.clone()

    if save_trajectory:
        trajectory[0] = x.clone()
        return x, trajectory
    return x


def load_model(checkpoint_path, device='cpu'):
    """加载训练好的模型."""
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    schedule = NoiseSchedule(
        timesteps=ckpt['timesteps'],
        schedule_type=ckpt['schedule_type'],
    )
    model = UNet(in_channels=1, base_channels=64, channel_mults=(1, 2, 4)).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, schedule


def sample_and_visualize(n_samples=64, device='cpu'):
    """生成样本并可视化."""
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    ckpt_path = os.path.join(output_dir, 'checkpoint.pt')

    if not os.path.exists(ckpt_path):
        print("No checkpoint found. Run train_mnist.py first.")
        return

    print("Loading model...")
    model, schedule = load_model(ckpt_path, device=device)

    print(f"Generating {n_samples} samples...")
    samples = p_sample_loop(model, schedule, (n_samples, 1, 28, 28), device=device)
    samples = (samples.clamp(-1, 1) + 1) / 2  # 反归一化到 [0, 1]

    # 保存生成样本网格
    nrow = 8
    fig, axes = plt.subplots(n_samples // nrow, nrow, figsize=(16, 16))
    for i in range(n_samples):
        ax = axes[i // nrow, i % nrow]
        ax.imshow(samples[i, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
    plt.suptitle('Generated MNIST Digits (DDPM)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'generated_samples.png'), dpi=150)
    plt.close()
    print(f"Saved: generated_samples.png")

    # 保存去噪轨迹
    print("Generating denoising trajectory...")
    trajectory_steps = {999, 800, 600, 400, 200, 100, 50, 20, 10, 0}
    _, traj = p_sample_loop(
        model, schedule, (8, 1, 28, 28), device=device,
        save_trajectory=True, trajectory_steps=trajectory_steps,
    )

    sorted_steps = sorted(traj.keys(), reverse=True)
    fig, axes = plt.subplots(1, len(sorted_steps), figsize=(20, 3))
    for i, t_val in enumerate(sorted_steps):
        img = (traj[t_val][0, 0].clamp(-1, 1) + 1) / 2
        axes[i].imshow(img.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        axes[i].set_title(f't={t_val}', fontsize=10)
        axes[i].axis('off')
    plt.suptitle('Denoising Trajectory: x_T → x_0', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'denoising_trajectory.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: denoising_trajectory.png")


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    sample_and_visualize(n_samples=64, device=device)