"""
MNIST 训练: 在手写数字数据集上训练 U-Net 扩散模型

实现 L_simple = E[||ε - ε_θ(x_t, t)||²] 的完整训练流程.

运行: python stage3_unet/train_mnist.py
"""

import torch
import torch.nn as nn
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from stage1_forward_process.noise_schedule import NoiseSchedule
from stage1_forward_process.forward_diffusion import q_sample
from stage3_unet.unet import UNet

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_mnist_loader(batch_size=128, data_dir='./data'):
    transform = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    dataset = datasets.MNIST(root=data_dir, train=True, download=False, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)


def train(epochs=10, batch_size=128, lr=2e-4, timesteps=1000, device='cpu',
          save_every=2):
    print(f"Training U-Net on MNIST ({timesteps} timesteps, {epochs} epochs)")

    schedule = NoiseSchedule(timesteps=timesteps, schedule_type='linear')
    model = UNet(in_channels=1, base_channels=64, channel_mults=(1, 2, 4)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    loader = get_mnist_loader(batch_size=batch_size)
    losses = []

    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for x_0, _ in loader:
            x_0 = x_0.to(device)
            batch_size_actual = x_0.shape[0]

            t = torch.randint(0, timesteps, (batch_size_actual,), device=device)
            noise = torch.randn_like(x_0)
            x_t = q_sample(schedule, x_0, t, noise)
            predicted_noise = model(x_t, t)
            loss = nn.functional.mse_loss(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        epoch_loss /= n_batches
        losses.append(epoch_loss)
        print(f"  Epoch {epoch+1}/{epochs}, loss={epoch_loss:.6f}")

        if (epoch + 1) % save_every == 0 or epoch == epochs - 1:
            torch.save({
                'model_state': model.state_dict(),
                'schedule_type': 'linear',
                'timesteps': timesteps,
                'epoch': epoch + 1,
                'losses': losses,
            }, os.path.join(output_dir, 'checkpoint.pt'))

    print(f"Model saved to {output_dir}/checkpoint.pt")
    return model, schedule, losses


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    train(epochs=10, batch_size=128, lr=2e-4, device=device)