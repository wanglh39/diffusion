"""
训练循环: 在2D玩具数据上训练扩散模型

实现 L_simple = E[||ε - ε_θ(x_t, t)||²] 的完整训练流程,
以及反向采样 p_sample() 用于生成.

运行: python stage2_training/train.py
"""

import torch
import torch.nn as nn
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from stage1_forward_process.noise_schedule import NoiseSchedule
from stage1_forward_process.forward_diffusion import q_sample
from stage2_training.simple_model import DiffusionMLP
from stage2_training.toy_dataset import get_toy_data


def train_step(model, schedule, x_0, optimizer):
    """
    一步训练: 采样 (x_0, t, ε), 计算 L_simple.

    L_simple = E[||ε - ε_θ(√ᾱ_t·x_0 + √(1-ᾱ_t)·ε, t)||²]
    """
    batch_size = x_0.shape[0]
    t = torch.randint(0, schedule.timesteps, (batch_size,))
    noise = torch.randn_like(x_0)
    x_t = q_sample(schedule, x_0, t, noise)
    predicted_noise = model(x_t, t)
    loss = nn.functional.mse_loss(predicted_noise, noise)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def p_sample(model, schedule, x_t, t_index):
    """
    反向采样一步: p_θ(x_{t-1}|x_t).

    μ_θ = 1/√α_t · (x_t - β_t/√(1-ᾱ_t) · ε_θ(x_t, t))
    x_{t-1} = μ_θ + σ_t · z,  z ~ N(0, I)
    """
    t = torch.full((x_t.shape[0],), t_index, dtype=torch.long)
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
def p_sample_loop(model, schedule, shape, device='cpu'):
    """
    完整反向采样: 从 x_T ~ N(0,I) 逐步去噪到 x_0.
    """
    x = torch.randn(shape, device=device)
    for t_index in reversed(range(schedule.timesteps)):
        x = p_sample(model, schedule, x, t_index)
    return x


def train(dataset_type='two_moons', timesteps=1000, epochs=5000,
          batch_size=256, lr=1e-3, device='cpu'):
    """完整训练流程."""
    print(f"Training diffusion model on {dataset_type} ({timesteps} timesteps)")

    data = get_toy_data(dataset_type, n_samples=10000)
    data = data.to(device)

    schedule = NoiseSchedule(timesteps=timesteps, schedule_type='cosine')
    model = DiffusionMLP(data_dim=2, hidden_dim=128, time_emb_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_batches = len(data) // batch_size
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        perm = torch.randperm(len(data))
        for i in range(n_batches):
            idx = perm[i * batch_size:(i + 1) * batch_size]
            x_0 = data[idx]
            loss = train_step(model, schedule, x_0, optimizer)
            epoch_loss += loss
        epoch_loss /= n_batches
        losses.append(epoch_loss)

        if (epoch + 1) % 500 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, loss={epoch_loss:.6f}")

    # 保存模型和损失曲线
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    torch.save({
        'model_state': model.state_dict(),
        'schedule_type': 'cosine',
        'timesteps': timesteps,
        'dataset_type': dataset_type,
        'losses': losses,
    }, os.path.join(output_dir, 'checkpoint.pt'))
    print(f"Model saved to {output_dir}/checkpoint.pt")

    # 生成样本
    samples = p_sample_loop(model, schedule, (2000, 2), device=device)
    torch.save(samples, os.path.join(output_dir, 'samples.pt'))
    print(f"Generated {samples.shape[0]} samples")

    return model, schedule, losses, samples


if __name__ == '__main__':
    model, schedule, losses, samples = train(
        dataset_type='two_moons',
        timesteps=1000,
        epochs=5000,
        batch_size=256,
        lr=1e-3,
    )