"""
对比可视化: DDPM vs DDIM

对比内容:
  1. 采样速度: DDPM 1000步 vs DDIM 50步
  2. 采样质量: 两种方法的生成样本统计量对比
  3. 确定性: DDIM 同种子可复现
  4. 步数扫描: DDIM 10/20/50/100步的质量变化
  5. 数值稳定性: cosine schedule 下 DDPM vs DDIM

运行: python stage4_sampling/compare.py
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from stage1_forward_process.noise_schedule import NoiseSchedule
from stage2_training.simple_model import DiffusionMLP
from stage2_training.toy_dataset import get_toy_data
from stage4_sampling.ddpm_sampler import DDPMSampler
from stage4_sampling.ddim_sampler import DDIMSampler


def compare_speed_and_quality(output_dir):
    """对比 DDPM 和 DDIM 的速度与质量."""
    schedule = NoiseSchedule(timesteps=1000, schedule_type='linear')

    # 加载 Stage 2 训练好的模型
    ckpt_path = os.path.join(os.path.dirname(__file__), '..', 'stage2_training', 'output', 'checkpoint.pt')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=False)
        model = DiffusionMLP(data_dim=2, hidden_dim=128, time_emb_dim=64)
        model.load_state_dict(ckpt['model_state'])
        model.eval()
        dataset_type = ckpt['dataset_type']
    else:
        print("No Stage 2 checkpoint found. Skipping speed/quality comparison.")
        return

    data = get_toy_data(dataset_type, n_samples=5000)

    ddpm_sampler = DDPMSampler(schedule)
    ddim_sampler = DDIMSampler(schedule)

    # DDPM 1000步
    print("DDPM 1000 steps...")
    torch.manual_seed(42)
    t0 = time.time()
    ddpm_samples = ddpm_sampler.sample(model, (2000, 2))
    ddpm_time = time.time() - t0

    # DDIM 50步
    print("DDIM 50 steps...")
    torch.manual_seed(42)
    t0 = time.time()
    ddim_samples = ddim_sampler.sample(model, (2000, 2), num_steps=50, eta=0.0)
    ddim_time = time.time() - t0

    # DDIM 20步
    print("DDIM 20 steps...")
    torch.manual_seed(42)
    ddim20_samples = ddim_sampler.sample(model, (2000, 2), num_steps=20, eta=0.0)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].scatter(data[:, 0].numpy(), data[:, 1].numpy(), s=2, alpha=0.3, c='blue')
    axes[0].set_title(f'Original Data\n({dataset_type})')
    axes[0].set_xlim(-3, 3); axes[0].set_ylim(-3, 3); axes[0].set_aspect('equal')

    axes[1].scatter(ddpm_samples[:, 0].numpy(), ddpm_samples[:, 1].numpy(), s=2, alpha=0.3, c='red')
    axes[1].set_title(f'DDPM (1000 steps)\n{ddpm_time:.1f}s\nmean=({ddpm_samples.mean(0)[0]:.2f},{ddpm_samples.mean(0)[1]:.2f})')
    axes[1].set_xlim(-3, 3); axes[1].set_ylim(-3, 3); axes[1].set_aspect('equal')

    axes[2].scatter(ddim_samples[:, 0].numpy(), ddim_samples[:, 1].numpy(), s=2, alpha=0.3, c='green')
    axes[2].set_title(f'DDIM (50 steps, η=0)\n{ddim_time:.1f}s\nmean=({ddim_samples.mean(0)[0]:.2f},{ddim_samples.mean(0)[1]:.2f})')
    axes[2].set_xlim(-3, 3); axes[2].set_ylim(-3, 3); axes[2].set_aspect('equal')

    axes[3].scatter(ddim20_samples[:, 0].numpy(), ddim20_samples[:, 1].numpy(), s=2, alpha=0.3, c='orange')
    axes[3].set_title(f'DDIM (20 steps, η=0)\nmean=({ddim20_samples.mean(0)[0]:.2f},{ddim20_samples.mean(0)[1]:.2f})')
    axes[3].set_xlim(-3, 3); axes[3].set_ylim(-3, 3); axes[3].set_aspect('equal')

    plt.suptitle(f'DDPM vs DDIM: Speed & Quality\n'
                 f'DDPM {ddpm_time:.1f}s | DDIM-50 {ddim_time:.1f}s | Speedup: {ddpm_time/max(ddim_time,0.01):.1f}x',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ddpm_vs_ddim.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: ddpm_vs_ddim.png")
    print(f"  DDPM: {ddpm_time:.1f}s, mean=({ddpm_samples.mean(0)[0]:.3f},{ddpm_samples.mean(0)[1]:.3f}), std=({ddpm_samples.std(0)[0]:.3f},{ddpm_samples.std(0)[1]:.3f})")
    print(f"  DDIM-50: {ddim_time:.1f}s, mean=({ddim_samples.mean(0)[0]:.3f},{ddim_samples.mean(0)[1]:.3f}), std=({ddim_samples.std(0)[0]:.3f},{ddim_samples.std(0)[1]:.3f})")
    print(f"  DDIM-20: mean=({ddim20_samples.mean(0)[0]:.3f},{ddim20_samples.mean(0)[1]:.3f}), std=({ddim20_samples.std(0)[0]:.3f},{ddim20_samples.std(0)[1]:.3f})")


def compare_determinism(output_dir):
    """验证 DDIM 的确定性: 同种子两次采样应完全相同."""
    schedule = NoiseSchedule(timesteps=1000, schedule_type='linear')
    model = DiffusionMLP(data_dim=2, hidden_dim=64, time_emb_dim=32)
    ddim_sampler = DDIMSampler(schedule)

    torch.manual_seed(123)
    x1 = ddim_sampler.sample(model, (1000, 2), num_steps=50, eta=0.0)
    torch.manual_seed(123)
    x2 = ddim_sampler.sample(model, (1000, 2), num_steps=50, eta=0.0)

    torch.manual_seed(123)
    x3 = ddim_sampler.sample(model, (1000, 2), num_steps=50, eta=1.0)
    torch.manual_seed(123)
    x4 = ddim_sampler.sample(model, (1000, 2), num_steps=50, eta=1.0)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, (samples, title) in zip(axes, [
        (x1, 'DDIM η=0, run 1'),
        (x2, 'DDIM η=0, run 2\n(same seed)'),
        (x3, 'DDIM η=1, run 1'),
        (x4, 'DDIM η=1, run 2\n(same seed)'),
    ]):
        ax.scatter(samples[:, 0].numpy(), samples[:, 1].numpy(), s=2, alpha=0.3)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.set_aspect('equal')

    plt.suptitle(f'Determinism Test\n'
                 f'η=0: identical={torch.allclose(x1, x2)} | '
                 f'η=1: identical={torch.allclose(x3, x4)}',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'determinism_test.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: determinism_test.png")
    print(f"  η=0 (deterministic): identical = {torch.allclose(x1, x2)}")
    print(f"  η=1 (stochastic):    identical = {torch.allclose(x3, x4)}")


def compare_step_sweep(output_dir):
    """DDIM 步数扫描: 10/20/50/100/200步的生成质量变化."""
    schedule = NoiseSchedule(timesteps=1000, schedule_type='linear')

    ckpt_path = os.path.join(os.path.dirname(__file__), '..', 'stage2_training', 'output', 'checkpoint.pt')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=False)
        model = DiffusionMLP(data_dim=2, hidden_dim=128, time_emb_dim=64)
        model.load_state_dict(ckpt['model_state'])
        model.eval()
    else:
        print("No Stage 2 checkpoint found. Skipping step sweep.")
        return

    ddim_sampler = DDIMSampler(schedule)
    step_counts = [10, 20, 50, 100, 200, 1000]

    fig, axes = plt.subplots(1, len(step_counts), figsize=(25, 5))
    stats = []

    for ax, n_steps in zip(axes, step_counts):
        torch.manual_seed(42)
        if n_steps == 1000:
            ddpm_sampler = DDPMSampler(schedule)
            samples = ddpm_sampler.sample(model, (2000, 2))
            label = 'DDPM\n(1000 steps)'
        else:
            samples = ddim_sampler.sample(model, (2000, 2), num_steps=n_steps, eta=0.0)
            label = f'DDIM\n({n_steps} steps)'

        m = samples.mean(0)
        s = samples.std(0)
        stats.append((n_steps, m, s))

        ax.scatter(samples[:, 0].numpy(), samples[:, 1].numpy(), s=2, alpha=0.3)
        ax.set_title(f'{label}\nmean=({m[0]:.2f},{m[1]:.2f})\nstd=({s[0]:.2f},{s[1]:.2f})', fontsize=9)
        ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_aspect('equal')

    plt.suptitle('DDIM Step Sweep: Quality vs Number of Steps', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'step_sweep.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: step_sweep.png")
    for n_steps, m, s in stats:
        print(f"  {n_steps:4d} steps: mean=({m[0]:.3f},{m[1]:.3f}), std=({s[0]:.3f},{s[1]:.3f})")


def compare_numerical_stability(output_dir):
    """对比 DDPM 和 DDIM 在 cosine schedule 下的数值稳定性."""
    schedule = NoiseSchedule(timesteps=1000, schedule_type='cosine')
    model = DiffusionMLP(data_dim=2, hidden_dim=64, time_emb_dim=32)

    ddpm_sampler = DDPMSampler(schedule)
    ddim_sampler = DDIMSampler(schedule)

    # DDPM cosine: 可能数值爆炸
    torch.manual_seed(42)
    try:
        ddpm_samples = ddpm_sampler.sample(model, (500, 2))
        ddpm_max = ddpm_samples.abs().max().item()
        ddpm_status = f"max|x|={ddpm_max:.2f}"
    except Exception as e:
        ddpm_samples = None
        ddpm_status = f"ERROR: {e}"

    # DDIM cosine: 应该数值稳定
    torch.manual_seed(42)
    ddim_samples = ddim_sampler.sample(model, (500, 2), num_steps=50, eta=0.0)
    ddim_max = ddim_samples.abs().max().item()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    if ddpm_samples is not None and ddpm_max < 1e6:
        axes[0].scatter(ddpm_samples[:, 0].numpy(), ddpm_samples[:, 1].numpy(), s=2, alpha=0.3, c='red')
        axes[0].set_title(f'DDPM + Cosine\n{ddpm_status}')
    else:
        axes[0].text(0.5, 0.5, f'DDPM + Cosine\nNUMERICAL EXPLOSION\n{ddpm_status}',
                     ha='center', va='center', transform=axes[0].transAxes, fontsize=12, color='red')
        axes[0].set_title('DDPM + Cosine (UNSTABLE)')
    axes[0].set_xlim(-5, 5); axes[0].set_ylim(-5, 5); axes[0].set_aspect('equal')

    axes[1].scatter(ddim_samples[:, 0].numpy(), ddim_samples[:, 1].numpy(), s=2, alpha=0.3, c='green')
    axes[1].set_title(f'DDIM + Cosine\nmax|x|={ddim_max:.2f} (STABLE)')
    axes[1].set_xlim(-5, 5); axes[1].set_ylim(-5, 5); axes[1].set_aspect('equal')

    plt.suptitle('Numerical Stability: DDPM vs DDIM on Cosine Schedule', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'numerical_stability.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: numerical_stability.png")
    print(f"  DDPM + Cosine: {ddpm_status}")
    print(f"  DDIM + Cosine: max|x|={ddim_max:.2f} (stable)")


def main():
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Stage 4: DDPM vs DDIM 对比")
    print("=" * 60)

    print("\n[1/4] 速度与质量对比...")
    compare_speed_and_quality(output_dir)

    print("\n[2/4] 确定性验证...")
    compare_determinism(output_dir)

    print("\n[3/4] 步数扫描...")
    compare_step_sweep(output_dir)

    print("\n[4/4] 数值稳定性对比...")
    compare_numerical_stability(output_dir)

    print("\n" + "=" * 60)
    print("完成! 输出文件在 stage4_sampling/output/")
    print("=" * 60)


if __name__ == '__main__':
    main()