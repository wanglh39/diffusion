"""
CFG 采样: Classifier-free Guidance 条件生成

核心公式:
  ε_guided = ε_uncond + w · (ε_cond - ε_uncond)
  = (1-w) · ε_uncond + w · ε_cond

w=0: 无条件生成
w=1: 条件生成 (无引导增强)
w>1: 强引导 (增强条件匹配, 减少多样性)

支持 DDPM 和 DDIM 两种采样方式.
"""

import torch
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from stage1_forward_process.noise_schedule import NoiseSchedule
from stage5_advanced.conditional_unet import ConditionalUNet
from stage4_sampling.ddim_sampler import DDIMSampler, get_timestep_subsequence

import matplotlib.pyplot as plt


@torch.no_grad()
def cfg_predict_eps(model, x_t, t_index, y, guidance_scale=3.0, num_classes=10):
    """
    CFG 预测噪声: 同时计算条件和无条件预测, 用差值引导.

    ε_guided = ε_uncond + w · (ε_cond - ε_uncond)

    需要2次模型前向传播: 一次条件, 一次无条件.
    """
    t = torch.full((x_t.shape[0],), t_index, dtype=torch.long, device=x_t.device)

    # 无条件预测: y 全部替换为 null
    y_uncond = torch.full_like(y, num_classes)  # null class index
    eps_uncond = model(x_t, t, y_uncond, drop_prob=0.0)

    # 条件预测
    eps_cond = model(x_t, t, y, drop_prob=0.0)

    # 引导公式
    eps_guided = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
    return eps_guided


@torch.no_grad()
def cfg_sample_ddim(model, schedule, y, guidance_scale=3.0, num_steps=50,
                    device='cpu', num_classes=10):
    """
    CFG + DDIM 采样: 条件生成.

    Args:
        y: (B,) 目标类别标签
        guidance_scale: 引导强度 w
        num_steps: DDIM 采样步数
    """
    batch_size = y.shape[0]
    shape = (batch_size, 1, 28, 28)
    timesteps_seq = get_timestep_subsequence(schedule.timesteps, num_steps)

    x = torch.randn(shape, device=device)

    for i in reversed(range(len(timesteps_seq))):
        t_index = int(timesteps_seq[i])
        prev_t_index = int(timesteps_seq[i - 1]) if i > 0 else -1

        # CFG 预测噪声
        eps = cfg_predict_eps(model, x, t_index, y, guidance_scale, num_classes)

        # DDIM step (与 ddim_sampler.py 相同的逻辑)
        alpha_bar_t = schedule.alphas_cumprod[t_index]
        alpha_bar_prev = schedule.alphas_cumprod[prev_t_index] if prev_t_index >= 0 else torch.tensor(1.0)

        x_0_pred = (x - torch.sqrt(1 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)

        sigma = torch.tensor(0.0)  # eta=0, 确定性DDIM
        dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        mean = torch.sqrt(alpha_bar_prev) * x_0_pred

        x = mean + dir_xt

    return x


def load_model(checkpoint_path, device='cpu'):
    """加载 CFG 训练好的模型."""
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    schedule = NoiseSchedule(
        timesteps=ckpt['timesteps'],
        schedule_type=ckpt['schedule_type'],
    )
    model = ConditionalUNet(
        in_channels=1, num_classes=ckpt['num_classes'],
        base_channels=64, channel_mults=(1, 2, 4),
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, schedule, ckpt['num_classes']


def sample_and_visualize(device='cpu'):
    """条件生成 + 引导强度对比可视化."""
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    ckpt_path = os.path.join(output_dir, 'checkpoint.pt')

    if not os.path.exists(ckpt_path):
        print("No checkpoint found. Run cfg_train.py first.")
        return

    print("Loading model...")
    model, schedule, num_classes = load_model(ckpt_path, device=device)

    # 为每个数字 0-9 生成样本, 对比不同引导强度
    guidance_scales = [0.0, 1.0, 3.0, 5.0]
    labels = list(range(10))

    fig, axes = plt.subplots(len(guidance_scales), len(labels),
                             figsize=(20, 8))

    for row, w in enumerate(guidance_scales):
        y = torch.tensor(labels, device=device)
        print(f"Generating with guidance_scale={w}...")
        samples = cfg_sample_ddim(
            model, schedule, y, guidance_scale=w,
            num_steps=50, device=device, num_classes=num_classes,
        )
        samples = (samples.clamp(-1, 1) + 1) / 2

        for col in range(len(labels)):
            ax = axes[row, col]
            ax.imshow(samples[col, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            ax.axis('off')
            if row == 0:
                ax.set_title(f'y={col}', fontsize=10)

        axes[row, 0].set_ylabel(f'w={w}', fontsize=12, rotation=0, labelpad=30)

    plt.suptitle('Classifier-free Guidance: Condition vs Guidance Scale', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cfg_guidance_comparison.png'), dpi=150)
    plt.close()
    print(f"Saved: cfg_guidance_comparison.png")


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    sample_and_visualize(device=device)