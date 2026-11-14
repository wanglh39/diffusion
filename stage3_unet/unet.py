"""
U-Net 架构: ε_θ(x_t, t) 的神经网络实现

核心组件:
  1. ResidualBlock — 卷积 + 时间嵌入注入 + 残差连接
  2. SelfAttention — 全局信息整合 (仅在低分辨率层使用)
  3. Downsample / Upsample — 分辨率调整
  4. UNet — 编码器-解码器 + skip connection 的完整架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from stage3_unet.sinusoidal_embedding import TimeEmbeddingMLP


class ResidualBlock(nn.Module):
    """
    残差块: GroupNorm → SiLU → Conv → +time_emb → GroupNorm → SiLU → Conv → +residual

    时间嵌入通过加法注入: 把 time_emb 投影到通道维度后加到中间特征上.
    """
    def __init__(self, in_channels, out_channels, time_emb_dim, groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, time_emb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        t = self.time_proj(time_emb)
        h = h + t[:, :, None, None]

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        return h + self.shortcut(x)


class SelfAttention(nn.Module):
    """
    自注意力: 在低分辨率层使用, 整合全局信息.

    计算复杂度 O(H²W²), 所以只在 H,W ≤ 8 时使用.
    """
    def __init__(self, channels, groups=8):
        super().__init__()
        self.norm = nn.GroupNorm(groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(B, C, H * W).permute(0, 2, 1)
        k = k.reshape(B, C, H * W)
        v = v.reshape(B, C, H * W).permute(0, 2, 1)

        attn = torch.bmm(q, k) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(attn, v)
        out = out.permute(0, 2, 1).reshape(B, C, H, W)
        out = self.proj(out)

        return x + out


class Downsample(nn.Module):
    """下采样: stride=2 卷积, 同时降分辨率."""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """上采样: 先 interpolate 放大2倍, 再用卷积调整通道."""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net for Diffusion: ε_θ(x_t, t)

    输入: x_t (B, C, H, W), t (B,)
    输出: ε_θ (B, C, H, W) — 预测的噪声

    架构: 3层下采样 + bottleneck + 3层上采样, skip connection连接对应层.
    在最低分辨率层(bottleneck和decoder底层)使用自注意力.
    """
    def __init__(self, in_channels=1, base_channels=64, time_emb_dim=256,
                 channel_mults=(1, 2, 4), num_res_blocks=2, attn_resolutions=(8,)):
        super().__init__()
        self.time_embedding = TimeEmbeddingMLP(base_channels * 4, time_emb_dim)

        # 输入卷积
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
                blocks.append(ResidualBlock(channels, out_channels, base_channels * 4))
                channels = out_channels
                skip_channels.append(channels)
            self.encoder.append(blocks)

            if level < len(channel_mults) - 1:
                self.downsamplers.append(Downsample(channels))
                skip_channels.append(channels)

        # Bottleneck
        self.mid_block1 = ResidualBlock(channels, channels, base_channels * 4)
        self.mid_attn = SelfAttention(channels) if 8 in attn_resolutions else nn.Identity()
        self.mid_block2 = ResidualBlock(channels, channels, base_channels * 4)

        # 解码器
        self.decoder = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        rev_mults = list(reversed(channel_mults))

        for level, mult in enumerate(rev_mults):
            out_channels = base_channels * mult
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                skip_c = skip_channels.pop()
                blocks.append(ResidualBlock(channels + skip_c, out_channels, base_channels * 4))
                channels = out_channels
            self.decoder.append(blocks)

            if level < len(rev_mults) - 1:
                self.upsamplers.append(Upsample(channels))

        # 输出卷积
        self.norm_out = nn.GroupNorm(8, channels)
        self.conv_out = nn.Conv2d(channels, in_channels, 3, padding=1)

    def forward(self, x, t):
        time_emb = self.time_embedding(t)

        x = self.conv_in(x)
        skips = [x]

        # 编码器
        for level, blocks in enumerate(self.encoder):
            for block in blocks:
                x = block(x, time_emb)
                skips.append(x)
            if level < len(self.downsamplers):
                x = self.downsamplers[level](x)
                skips.append(x)

        # Bottleneck
        x = self.mid_block1(x, time_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, time_emb)

        # 解码器
        for level, blocks in enumerate(self.decoder):
            for block in blocks:
                skip = skips.pop()
                x = torch.cat([x, skip], dim=1)
                x = block(x, time_emb)
            if level < len(self.upsamplers):
                x = self.upsamplers[level](x)

        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)

        return x


if __name__ == '__main__':
    model = UNet(in_channels=1, base_channels=64, channel_mults=(1, 2, 4))
    x = torch.randn(2, 1, 28, 28)
    t = torch.tensor([0, 500])
    eps = model(x, t)
    print(f"Input: {x.shape}")
    print(f"Output: {eps.shape}")
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")