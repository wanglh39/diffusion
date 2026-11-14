"""
正弦时间嵌入 (Sinusoidal Time Embedding)

把整数时间步 t 映射为连续向量, 使网络能理解"现在是第几步".

公式:
  PE(t, 2i)   = sin(t / 10000^{2i/d})
  PE(t, 2i+1) = cos(t / 10000^{2i/d})

设计哲学:
  - 连续性: 相近的 t 产生相近的嵌入
  - 多频率: 低频维度捕捉粗粒度时间信息, 高频维度捕捉细粒度
  - 无需学习: 编码本身确定, 不需额外参数
"""

import torch
import torch.nn as nn
import math


class SinusoidalTimeEmbedding(nn.Module):
    """
    正弦时间嵌入: t (B,) → emb (B, dim)
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


class TimeEmbeddingMLP(nn.Module):
    """
    时间嵌入 + MLP投影: t → sinusoidal emb → MLP → projected emb

    正弦编码是确定的, MLP学习如何把时间信息投影到各层所需的维度.
    """
    def __init__(self, dim, time_emb_dim):
        super().__init__()
        self.sinusoidal = SinusoidalTimeEmbedding(time_emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(time_emb_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t):
        emb = self.sinusoidal(t)
        return self.mlp(emb)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    emb = SinusoidalTimeEmbedding(dim=64)
    t = torch.arange(1000)
    embeddings = emb(t)  # (1000, 64)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    axes[0].imshow(embeddings.numpy().T, aspect='auto', cmap='RdBu')
    axes[0].set_title('Sinusoidal Time Embedding (t=0..999, dim=0..63)')
    axes[0].set_xlabel('t')
    axes[0].set_ylabel('embedding dim')

    # 展示几个维度随t的变化
    for i in [0, 1, 16, 32, 48, 62]:
        axes[1].plot(embeddings[:, i].numpy(), label=f'dim {i}', alpha=0.7)
    axes[1].set_title('Individual embedding dimensions vs t')
    axes[1].set_xlabel('t')
    axes[1].legend()

    plt.tight_layout()
    import os
    os.makedirs('stage3_unet/output', exist_ok=True)
    plt.savefig('stage3_unet/output/time_embedding.png', dpi=150)
    print("Saved: time_embedding.png")