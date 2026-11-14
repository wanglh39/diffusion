"""
Stage 1 可视化: 噪声调度表 + 前向加噪轨迹

运行: python stage1_forward_process/visualize.py
输出: 3张图保存到 stage1_forward_process/output/
  1. schedule_comparison.png  — Linear vs Cosine 调度表对比
  2. forward_trajectory.png   — MNIST数字逐步加噪轨迹
  3. closed_vs_chain.png      — 闭式解 vs 逐步加噪验证
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from stage1_forward_process.noise_schedule import NoiseSchedule
from stage1_forward_process.forward_diffusion import q_sample, q_sample_chain
from utils.viz_utils import show_images, plot_schedule
from torchvision import datasets, transforms


def visualize_schedule_comparison():
    """对比 Linear 和 Cosine 两种噪声调度表."""
    T = 1000
    sched_lin = NoiseSchedule(timesteps=T, schedule_type='linear')
    sched_cos = NoiseSchedule(timesteps=T, schedule_type='cosine')

    t = np.arange(T)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # β_t 对比
    axes[0].plot(t, sched_lin.betas.numpy(), label='Linear', color='red')
    axes[0].plot(t, sched_cos.betas.numpy(), label='Cosine', color='blue')
    axes[0].set_title('β_t: noise added per step')
    axes[0].set_xlabel('t')
    axes[0].legend()

    # ᾱ_t 对比 (最重要的图)
    axes[1].plot(t, sched_lin.alphas_cumprod.numpy(), label='Linear', color='red')
    axes[1].plot(t, sched_cos.alphas_cumprod.numpy(), label='Cosine', color='blue')
    axes[1].set_title('ᾱ_t: cumulative signal retained (KEY)')
    axes[1].set_xlabel('t')
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1].legend()

    # √(1-ᾱ_t) 对比 (噪声占比)
    axes[2].plot(t, sched_lin.sqrt_one_minus_alphas_cumprod.numpy(), label='Linear', color='red')
    axes[2].plot(t, sched_cos.sqrt_one_minus_alphas_cumprod.numpy(), label='Cosine', color='blue')
    axes[2].set_title('√(1-ᾱ_t): noise magnitude')
    axes[2].set_xlabel('t')
    axes[2].legend()

    plt.tight_layout()
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'schedule_comparison.png'), dpi=150)
    plt.close()
    print("Saved: schedule_comparison.png")


def create_synthetic_digit(size=32):
    """用合成图像代替MNIST下载, 验证前向加噪逻辑."""
    img = torch.zeros(1, 1, size, size)
    cx, cy = size // 2, size // 2
    r = size // 4
    for i in range(size):
        for j in range(size):
            if (i - cx) ** 2 + (j - cy) ** 2 < r ** 2:
                img[0, 0, i, j] = 0.9
    for i in range(cx - 2, cx + 3):
        for j in range(cy - r + 4, cy + r - 4):
            img[0, 0, i, j] = 0.1
    return img


def visualize_forward_trajectory():
    """可视化一张图像的逐步加噪轨迹 (使用合成图像或MNIST)."""
    T = 1000
    sched = NoiseSchedule(timesteps=T, schedule_type='cosine')

    try:
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
        ])
        dataset = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
        x_0 = dataset[0][0].unsqueeze(0)
        data_source = "MNIST"
    except Exception:
        x_0 = create_synthetic_digit(32)
        data_source = "Synthetic"

    # 选择若干时间步展示加噪过程
    timesteps_to_show = [0, 50, 100, 200, 400, 600, 800, 999]
    images = []
    titles = []

    for t_val in timesteps_to_show:
        t = torch.tensor([t_val])
        x_t = q_sample(sched, x_0, t)
        images.append(x_t[0, 0].numpy())
        alpha_bar = sched.alphas_cumprod[t_val].item()
        titles.append(f't={t_val}\nᾱ={alpha_bar:.4f}')

    fig, axes = plt.subplots(1, len(timesteps_to_show), figsize=(20, 2.5))
    for i, (img, title) in enumerate(zip(images, titles)):
        axes[i].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[i].set_title(title, fontsize=9)
        axes[i].axis('off')

    plt.suptitle(f'Forward Diffusion: x_0 → x_T ({data_source}, Cosine Schedule)', fontsize=12, y=1.05)
    plt.tight_layout()
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'forward_trajectory.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: forward_trajectory.png")


def visualize_closed_vs_chain():
    """验证闭式解 q(x_t|x_0) 与逐步加噪 q(x_t|x_{t-1})...q(x_1|x_0) 统计一致性."""
    T = 1000
    sched = NoiseSchedule(timesteps=T, schedule_type='linear')

    # 用一个简单的2D点作为 x_0, 方便统计对比
    torch.manual_seed(42)
    N = 5000
    x_0 = torch.randn(N, 2) * 0.5 + torch.tensor([1.0, -1.0])

    t_target = 500
    t = torch.full((N,), t_target, dtype=torch.long)

    # 方法1: 闭式解一步采样
    x_t_closed = q_sample(sched, x_0, t)

    # 方法2: 逐步加噪 (取少量样本验证, 因为逐步加噪慢)
    N_chain = 200
    x_t_chain = q_sample_chain(sched, x_0[:N_chain], t_target)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 原始数据
    axes[0].scatter(x_0[:, 0].numpy(), x_0[:, 1].numpy(), s=2, alpha=0.3, c='blue')
    axes[0].set_title(f'x_0 (original data)\nmean=({x_0.mean(0)[0]:.2f}, {x_0.mean(0)[1]:.2f})')
    axes[0].set_xlim(-5, 5)
    axes[0].set_ylim(-5, 5)
    axes[0].set_aspect('equal')

    # 闭式解
    axes[1].scatter(x_t_closed[:, 0].numpy(), x_t_closed[:, 1].numpy(), s=2, alpha=0.3, c='red')
    alpha_bar = sched.alphas_cumprod[t_target].item()
    expected_mean_scale = np.sqrt(alpha_bar)
    axes[1].set_title(
        f'Closed-form q(x_t|x_0)\n'
        f'√ᾱ={expected_mean_scale:.4f}, mean=({x_t_closed.mean(0)[0]:.2f}, {x_t_closed.mean(0)[1]:.2f})'
    )
    axes[1].set_xlim(-5, 5)
    axes[1].set_ylim(-5, 5)
    axes[1].set_aspect('equal')

    # 逐步加噪
    axes[2].scatter(x_t_chain[:, 0].numpy(), x_t_chain[:, 1].numpy(), s=2, alpha=0.3, c='green')
    axes[2].set_title(
        f'Step-by-step chain\n'
        f'mean=({x_t_chain.mean(0)[0]:.2f}, {x_t_chain.mean(0)[1]:.2f})'
    )
    axes[2].set_xlim(-5, 5)
    axes[2].set_ylim(-5, 5)
    axes[2].set_aspect('equal')

    plt.suptitle(
        f'Verification: q(x_t|x_0) closed-form vs step-by-step chain at t={t_target}\n'
        f'Both should have mean ≈ √ᾱ_t · mean(x_0) = {expected_mean_scale:.4f} × ({x_0.mean(0)[0]:.2f}, {x_0.mean(0)[1]:.2f})',
        fontsize=11
    )
    plt.tight_layout()
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'closed_vs_chain.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: closed_vs_chain.png")


def main():
    print("=" * 60)
    print("Stage 1: 前向扩散过程 可视化")
    print("=" * 60)

    print("\n[1/3] 噪声调度表对比 (Linear vs Cosine)...")
    visualize_schedule_comparison()

    print("\n[2/3] MNIST 前向加噪轨迹...")
    visualize_forward_trajectory()

    print("\n[3/3] 闭式解 vs 逐步加噪 验证...")
    visualize_closed_vs_chain()

    print("\n" + "=" * 60)
    print("完成! 输出文件在 stage1_forward_process/output/")
    print("=" * 60)


if __name__ == '__main__':
    main()