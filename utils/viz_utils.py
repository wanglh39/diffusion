import torch
import matplotlib.pyplot as plt
import numpy as np


def show_images(images, titles=None, figsize=(15, 3), nrow=None):
    """
    展示一组图像（用于可视化加噪轨迹等）.
    """
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()

    n = len(images)
    if nrow is None:
        nrow = n
    ncol = (n + nrow - 1) // nrow

    fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
    if nrow == 1 and ncol == 1:
        axes = np.array([[axes]])
    elif nrow == 1:
        axes = axes[np.newaxis, :]
    elif ncol == 1:
        axes = axes[:, np.newaxis]

    for i in range(nrow):
        for j in range(ncol):
            idx = i * ncol + j
            ax = axes[i][j]
            if idx < n:
                img = images[idx]
                if img.ndim == 3 and img.shape[0] in (1, 3):
                    img = img.transpose(1, 2, 0)
                ax.imshow(img, cmap='gray' if img.ndim == 2 else None, vmin=0, vmax=1)
                if titles is not None and idx < len(titles):
                    ax.set_title(titles[idx], fontsize=10)
            ax.axis('off')

    plt.tight_layout()
    return fig


def plot_schedule(betas, alphas, alphas_cumprod, figsize=(12, 4)):
    """
    可视化噪声调度表: β_t, α_t, α̅_t 随时间步变化.
    """
    t = np.arange(len(betas))
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].plot(t, betas, label='β_t', color='red')
    axes[0].set_title('β_t (noise added per step)')
    axes[0].set_xlabel('t')
    axes[0].legend()

    axes[1].plot(t, alphas, label='α_t = 1 - β_t', color='green')
    axes[1].set_title('α_t (signal retained per step)')
    axes[1].set_xlabel('t')
    axes[1].legend()

    axes[2].plot(t, alphas_cumprod, label='α̅_t = ∏α_t', color='blue')
    axes[2].set_title('α̅_t (cumulative signal retained)')
    axes[2].set_xlabel('t')
    axes[2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[2].legend()

    plt.tight_layout()
    return fig


def plot_2d_scatter(points, title='', ax=None, color='blue', alpha=0.3):
    """
    2D散点图（用于Stage 2的玩具数据可视化）.
    """
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(points[:, 0], points[:, 1], s=1, c=color, alpha=alpha)
    ax.set_title(title)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    return ax