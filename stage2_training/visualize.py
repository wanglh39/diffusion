"""
Stage 2 可视化: 训练损失 + 反向去噪过程

运行: python stage2_training/visualize.py
输出:
  1. training_loss.png     — 训练损失曲线
  2. reverse_trajectory.png — 反向去噪轨迹 (从噪声到数据)
  3. data_vs_samples.png   — 原始数据 vs 生成样本对比
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from stage1_forward_process.noise_schedule import NoiseSchedule
from stage2_training.simple_model import DiffusionMLP
from stage2_training.toy_dataset import get_toy_data
from stage2_training.train import p_sample, p_sample_loop


def visualize_training_loss(losses, output_dir):
    """训练损失曲线."""
    plt.figure(figsize=(8, 4))
    plt.plot(losses, alpha=0.7, linewidth=0.5)
    plt.plot(np.convolve(losses, np.ones(100)/100, mode='valid'), color='red', linewidth=2, label='smoothed')
    plt.xlabel('Epoch')
    plt.ylabel('L_simple = ||ε - ε_θ||²')
    plt.title('Training Loss')
    plt.legend()
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_loss.png'), dpi=150)
    plt.close()
    print("Saved: training_loss.png")


def visualize_reverse_trajectory(model, schedule, data, output_dir, device='cpu'):
    """反向去噪轨迹: 从纯噪声逐步恢复到数据分布."""
    n_steps_to_show = 10
    timesteps_to_show = np.linspace(schedule.timesteps - 1, 0, n_steps_to_show, dtype=int)

    x = torch.randn(2000, 2, device=device)
    snapshots = {}

    for t_index in reversed(range(schedule.timesteps)):
        if t_index in timesteps_to_show:
            snapshots[t_index] = x.clone()
        x = p_sample(model, schedule, x, t_index)
    snapshots[0] = x.clone()

    fig, axes = plt.subplots(1, n_steps_to_show, figsize=(25, 3))
    for i, t_val in enumerate(sorted(timesteps_to_show, reverse=True)):
        pts = snapshots[t_val].cpu().numpy()
        axes[i].scatter(pts[:, 0], pts[:, 1], s=2, alpha=0.3, c='blue')
        axes[i].set_title(f't={t_val}', fontsize=9)
        axes[i].set_xlim(-3, 3)
        axes[i].set_ylim(-3, 3)
        axes[i].set_aspect('equal')
        axes[i].axis('off')

    plt.suptitle('Reverse Diffusion: x_T (noise) → x_0 (data)', fontsize=12, y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reverse_trajectory.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: reverse_trajectory.png")


def visualize_data_vs_samples(data, samples, output_dir):
    """原始数据 vs 生成样本对比."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].scatter(data[:, 0].numpy(), data[:, 1].numpy(), s=2, alpha=0.3, c='blue')
    axes[0].set_title('Original Data')
    axes[0].set_xlim(-3, 3)
    axes[0].set_ylim(-3, 3)
    axes[0].set_aspect('equal')

    axes[1].scatter(samples[:, 0].numpy(), samples[:, 1].numpy(), s=2, alpha=0.3, c='red')
    axes[1].set_title('Generated Samples')
    axes[1].set_xlim(-3, 3)
    axes[1].set_ylim(-3, 3)
    axes[1].set_aspect('equal')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'data_vs_samples.png'), dpi=150)
    plt.close()
    print("Saved: data_vs_samples.png")


def main():
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)

    ckpt_path = os.path.join(output_dir, 'checkpoint.pt')

    if os.path.exists(ckpt_path):
        print("Loading trained model...")
        ckpt = torch.load(ckpt_path, weights_only=False)
        schedule = NoiseSchedule(
            timesteps=ckpt['timesteps'],
            schedule_type=ckpt['schedule_type'],
        )
        model = DiffusionMLP(data_dim=2, hidden_dim=128, time_emb_dim=64)
        model.load_state_dict(ckpt['model_state'])
        model.eval()
        losses = ckpt['losses']
        dataset_type = ckpt['dataset_type']
        samples = torch.load(os.path.join(output_dir, 'samples.pt'), weights_only=False)
    else:
        print("No checkpoint found. Training from scratch...")
        from stage2_training.train import train
        model, schedule, losses, samples = train(
            dataset_type='two_moons', timesteps=1000, epochs=2000,
        )
        dataset_type = 'two_moons'

    data = get_toy_data(dataset_type, n_samples=5000)

    print("\n[1/3] Training loss curve...")
    visualize_training_loss(losses, output_dir)

    print("\n[2/3] Reverse diffusion trajectory...")
    visualize_reverse_trajectory(model, schedule, data, output_dir)

    print("\n[3/3] Data vs samples comparison...")
    visualize_data_vs_samples(data, samples, output_dir)

    print("\nDone! Output in stage2_training/output/")


if __name__ == '__main__':
    main()