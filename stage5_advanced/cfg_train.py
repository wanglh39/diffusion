"""
CFG 训练: Classifier-free Guidance 的训练流程

核心: 训练时以概率 p_uncond 将类别标签替换为 null,
让同一个模型同时学习条件生成和无条件生成.

L = E[||ε - ε_θ(x_t, t, y')||²]
  y' = y 以概率 (1-p_uncond)
  y' = null 以概率 p_uncond

运行: python stage5_advanced/cfg_train.py
"""

import torch
import torch.nn as nn
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from stage1_forward_process.noise_schedule import NoiseSchedule
from stage1_forward_process.forward_diffusion import q_sample
from stage5_advanced.conditional_unet import ConditionalUNet

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def train_step_cfg(model, schedule, x_0, y, optimizer, p_uncond=0.1):
    """
    CFG 训练一步: 随机丢弃条件, 计算 L_simple.

    模型内部以概率 p_uncond 将 y 替换为 null 类别.
    """
    batch_size = x_0.shape[0]
    t = torch.randint(0, schedule.timesteps, (batch_size,))
    noise = torch.randn_like(x_0)
    x_t = q_sample(schedule, x_0, t, noise)

    # drop_prob 在模型内部处理, 将部分 y 替换为 null
    predicted_noise = model(x_t, t, y, drop_prob=p_uncond)
    loss = nn.functional.mse_loss(predicted_noise, noise)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def get_mnist_loader(batch_size=128, data_dir='./data'):
    transform = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    dataset = datasets.MNIST(root=data_dir, train=True, download=False, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)


def train(epochs=10, batch_size=128, lr=2e-4, timesteps=1000,
          p_uncond=0.1, device='cpu', save_every=2):
    print(f"Training Conditional U-Net on MNIST with CFG (p_uncond={p_uncond})")

    schedule = NoiseSchedule(timesteps=timesteps, schedule_type='linear')
    model = ConditionalUNet(
        in_channels=1, num_classes=10, base_channels=64,
        channel_mults=(1, 2, 4),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    loader = get_mnist_loader(batch_size=batch_size)
    losses = []

    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for x_0, y in loader:
            x_0 = x_0.to(device)
            y = y.to(device)

            loss = train_step_cfg(model, schedule, x_0, y, optimizer, p_uncond=p_uncond)

            epoch_loss += loss
            n_batches += 1

        epoch_loss /= n_batches
        losses.append(epoch_loss)
        print(f"  Epoch {epoch+1}/{epochs}, loss={epoch_loss:.6f}")

        if (epoch + 1) % save_every == 0 or epoch == epochs - 1:
            torch.save({
                'model_state': model.state_dict(),
                'schedule_type': 'linear',
                'timesteps': timesteps,
                'num_classes': 10,
                'p_uncond': p_uncond,
                'epoch': epoch + 1,
                'losses': losses,
            }, os.path.join(output_dir, 'checkpoint.pt'))

    print(f"Model saved to {output_dir}/checkpoint.pt")
    return model, schedule, losses


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    train(epochs=10, batch_size=128, lr=2e-4, device=device)