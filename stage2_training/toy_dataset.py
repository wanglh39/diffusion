"""
2D玩具数据集

用于验证扩散模型训练逻辑, 不需要图像数据.
模型学习从2D高斯噪声恢复到目标分布.
"""

import torch
import numpy as np
import os


def get_two_moons(n_samples=10000, noise=0.05):
    """
    Two Moons 数据集: 两个交叠的月牙形分布.
    这是扩散模型论文中常用的2D测试分布.
    """
    t = np.linspace(0, np.pi, n_samples // 2)
    upper_moon = np.stack([np.cos(t), np.sin(t)], axis=1)
    lower_moon = np.stack([1 - np.cos(t), -np.sin(t)], axis=1)
    data = np.concatenate([upper_moon, lower_moon], axis=0)
    data += np.random.randn(*data.shape) * noise
    return torch.tensor(data, dtype=torch.float32)


def get_swiss_roll(n_samples=10000, noise=0.05):
    """Swiss Roll 数据集: 螺旋形分布."""
    t = 1.5 * np.pi * (1 + 2 * np.random.rand(n_samples))
    data = np.stack([t * np.cos(t), t * np.sin(t)], axis=1) / 10
    data += np.random.randn(*data.shape) * noise
    return torch.tensor(data, dtype=torch.float32)


def get_gaussian_mixture(n_samples=10000, n_components=8, scale=1.0):
    """高斯混合分布: 多个高斯簇均匀分布在圆上."""
    angles = np.linspace(0, 2 * np.pi, n_components, endpoint=False)
    centers = np.stack([np.cos(angles), np.sin(angles)], axis=1) * scale
    samples_per_component = n_samples // n_components
    data = []
    for i in range(n_components):
        pts = centers[i] + np.random.randn(samples_per_component, 2) * 0.15
        data.append(pts)
    data = np.concatenate(data, axis=0)
    np.random.shuffle(data)
    return torch.tensor(data, dtype=torch.float32)


def get_toy_data(dataset_type='two_moons', n_samples=10000):
    """便捷接口."""
    if dataset_type == 'two_moons':
        return get_two_moons(n_samples)
    elif dataset_type == 'swiss_roll':
        return get_swiss_roll(n_samples)
    elif dataset_type == 'gaussian_mixture':
        return get_gaussian_mixture(n_samples)
    else:
        raise ValueError(f"Unknown dataset: {dataset_type}")


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, name in zip(axes, ['two_moons', 'swiss_roll', 'gaussian_mixture']):
        data = get_toy_data(name, 5000)
        ax.scatter(data[:, 0].numpy(), data[:, 1].numpy(), s=2, alpha=0.3)
        ax.set_title(name)
        ax.set_aspect('equal')
    plt.tight_layout()
    os.makedirs('stage2_training/output', exist_ok=True)
    plt.savefig('stage2_training/output/toy_datasets.png', dpi=150)
    print("Saved: toy_datasets.png")