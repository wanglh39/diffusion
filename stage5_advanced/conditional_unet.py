"""
条件 U-Net: ε_θ(x_t, t, y)

在 Stage 3 的 U-Net 基础上增加类别条件注入.
核心改动:
  1. ClassEmbedding: 把类别标签 y 编码为向量 (含 null 类别用于 CFG)
  2. 条件注入: 类别嵌入与时间嵌入一起通过加法注入 ResBlock

CFG 训练时, y 有概率被替换为 null 类别 (类别数), 模型同时学习条件和无条件生成.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from stage3_unet.sinusoidal_embedding import TimeEmbeddingMLP
from stage3_unet.unet import ResidualBlock, SelfAttention, Downsample, Upsample


class ClassEmbedding(nn.Module):
    """
    类别嵌入: y (整数标签) → embedding vector.

    包含一个额外的 "null" 类别 (索引 = num_classes), 用于 CFG 的无条件生成.
    训练时以概率 p_uncond 将 y 替换为 null 类别.
    """
    def __init__(self, num_classes, embed_dim):
        super().__init__()
        self.num_classes = num_classes
        self.null_class = num_classes  # null 类别的索引
        self.embedding = nn.Embedding(num_classes + 1, embed_dim)

    def forward(self, y, drop_prob=0.0):
        """
        Args:
            y: (B,) 类别标签, 值域 [0, num_classes-1]
            drop_prob: 替换为 null 的概率 (训练时用)
        """
        if drop_prob > 0:
            drop_mask = torch.rand(y.shape[0], device=y.device) < drop_prob
            y = torch.where(drop_mask, torch.full_like(y, self.null_class), y)
        return self.embedding(y)


class ConditionalResidualBlock(nn.Module):
    """
    条件残差块: 在 ResidualBlock 基础上增加类别条件注入.

    时间嵌入和类别嵌入相加后一起注入.
    """
    def __init__(self, in_channels, out_channels, time_emb_dim, class_emb_dim, groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )
        self.class_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(class_emb_dim, out_channels),
        )
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, time_emb, class_emb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        t = self.time_proj(time_emb)
        c = self.class_proj(class_emb)
        h = h + t[:, :, None, None] + c[:, :, None, None]

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        return h + self.shortcut(x)


class ConditionalUNet(nn.Module):
    """
    条件 U-Net: ε_θ(x_t, t, y)

    与 Stage 3 的 UNet 结构相同, 但 ResBlock 换成 ConditionalResidualBlock,
    增加类别条件注入.

    输入: x_t (B, C, H, W), t (B,), y (B,)
    输出: ε_θ (B, C, H, W)
    """
    def __init__(self, in_channels=1, num_classes=10, base_channels=64,
                 time_emb_dim=256, class_emb_dim=128,
                 channel_mults=(1, 2, 4), num_res_blocks=2, attn_resolutions=(8,)):
        super().__init__()
        self.num_classes = num_classes
        self.time_embedding = TimeEmbeddingMLP(base_channels * 4, time_emb_dim)
        self.class_embedding = ClassEmbedding(num_classes, class_emb_dim)

        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # 编码器
        self.encoder = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        channels = base_channels
        skip_channels = [base_channels]

        for level, mult in enumerate(channel_mults):
            out_channels = base_channels * mult
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ConditionalResidualBlock(
                    channels, out_channels, base_channels * 4, class_emb_dim))
                channels = out_channels
                skip_channels.append(channels)
            self.encoder.append(blocks)

            if level < len(channel_mults) - 1:
                self.downsamplers.append(Downsample(channels))
                skip_channels.append(channels)

        # Bottleneck
        self.mid_block1 = ConditionalResidualBlock(channels, channels, base_channels * 4, class_emb_dim)
        self.mid_attn = SelfAttention(channels) if 8 in attn_resolutions else nn.Identity()
        self.mid_block2 = ConditionalResidualBlock(channels, channels, base_channels * 4, class_emb_dim)

        # 解码器
        self.decoder = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        rev_mults = list(reversed(channel_mults))

        for level, mult in enumerate(rev_mults):
            out_channels = base_channels * mult
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                skip_c = skip_channels.pop()
                blocks.append(ConditionalResidualBlock(
                    channels + skip_c, out_channels, base_channels * 4, class_emb_dim))
                channels = out_channels
            self.decoder.append(blocks)

            if level < len(rev_mults) - 1:
                self.upsamplers.append(Upsample(channels))

        self.norm_out = nn.GroupNorm(8, channels)
        self.conv_out = nn.Conv2d(channels, in_channels, 3, padding=1)

    def forward(self, x, t, y, drop_prob=0.0):
        time_emb = self.time_embedding(t)
        class_emb = self.class_embedding(y, drop_prob=drop_prob)

        x = self.conv_in(x)
        skips = [x]

        for level, blocks in enumerate(self.encoder):
            for block in blocks:
                x = block(x, time_emb, class_emb)
                skips.append(x)
            if level < len(self.downsamplers):
                x = self.downsamplers[level](x)
                skips.append(x)

        x = self.mid_block1(x, time_emb, class_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, time_emb, class_emb)

        for level, blocks in enumerate(self.decoder):
            for block in blocks:
                skip = skips.pop()
                x = torch.cat([x, skip], dim=1)
                x = block(x, time_emb, class_emb)
            if level < len(self.upsamplers):
                x = self.upsamplers[level](x)

        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)

        return x


if __name__ == '__main__':
    model = ConditionalUNet(in_channels=1, num_classes=10, base_channels=64)
    x = torch.randn(2, 1, 28, 28)
    t = torch.tensor([0, 500])
    y = torch.tensor([3, 7])

    # 条件前向
    eps_cond = model(x, t, y, drop_prob=0.0)
    # 无条件前向 (drop_prob=1.0, 全部替换为 null)
    eps_uncond = model(x, t, y, drop_prob=1.0)

    print(f"Input: x={x.shape}, t={t.shape}, y={y.shape}")
    print(f"Conditional output: {eps_cond.shape}")
    print(f"Unconditional output: {eps_uncond.shape}")
    print(f"Cond != Uncond: {not torch.allclose(eps_cond, eps_uncond)}")
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")