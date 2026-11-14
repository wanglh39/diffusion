"""
简单MLP模型: ε_θ(x_t, t)

用于2D玩具数据的扩散模型. 输入 (x_t, t), 输出预测噪声 ε.
核心组件:
  1. Sinusoidal time embedding — 把整数时间步 t 编码为连续向量
  2. MLP with time conditioning — 通过拼接/加法将时间信息注入网络
"""

import torch
import torch.nn as nn
import math


class SinusoidalTimeEmbedding(nn.Module):
    """
    正弦时间嵌入 (Transformer positional encoding 的变体).

    把整数 t 映射到 dim 维向量, 使得相近的 t 有相近的嵌入.
    公式: PE(t, 2i) = sin(t / 10000^{2i/dim}), PE(t, 2i+1) = cos(t / 10000^{2i/dim})
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half_dim, device=t.device) / half_dim)
        args = t[:, None].float() * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(emb.shape[0], 1, device=t.device)], dim=-1)
        return emb


class DiffusionMLP(nn.Module):
    """
    ε_θ(x_t, t): 预测噪声的MLP.

    输入: x_t (B, 2), t (B,)
    输出: ε_θ (B, 2) — 预测的噪声

    结构: x_t 和 time embedding 拼接后通过4层MLP.
    """
    def __init__(self, data_dim=2, hidden_dim=128, time_emb_dim=64):
        super().__init__()
        self.time_embedding = SinusoidalTimeEmbedding(time_emb_dim)

        self.net = nn.Sequential(
            nn.Linear(data_dim + time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, data_dim),
        )

    def forward(self, x_t, t):
        t_emb = self.time_embedding(t)
        x = torch.cat([x_t, t_emb], dim=-1)
        return self.net(x)


if __name__ == '__main__':
    model = DiffusionMLP(data_dim=2, hidden_dim=128, time_emb_dim=64)
    x_t = torch.randn(4, 2)
    t = torch.tensor([0, 250, 500, 999])
    eps = model(x_t, t)
    print(f"Input x_t shape: {x_t.shape}")
    print(f"Input t shape: {t.shape}")
    print(f"Output ε_θ shape: {eps.shape}")
    print(f"Total params: {sum(p.numel() for p in model.parameters())}")