# Differtion — 动手学习扩散模型

> 从零开始，用 PyTorch 一步步手写扩散模型（Diffusion Models），理解底层原理与设计哲学。

## 项目哲学

- **从零构建**：不依赖 `diffusers` 等高层库，核心算法全部手写
- **数学与代码对应**：每个公式都有对应的代码实现，每个函数都标注它对应哪个公式
- **逐步递进**：5个阶段，每个阶段都可独立运行、验证
- **直觉先行**：每个概念先用类比/可视化建立直觉，再深入数学推导

## 5个阶段

| Stage | 主题 | 核心内容 |
|-------|------|----------|
| 1 | 前向过程 | 噪声调度表、$q(x_t\|x_0)$闭式解、加噪可视化 |
| 2 | 训练目标 | ELBO推导、$L_{simple}$、2D玩具数据验证 |
| 3 | U-Net架构 | 时间嵌入、下/上采样、skip connection、MNIST生成 |
| 4 | 采样算法 | DDPM采样、DDIM加速采样、对比分析 |
| 5 | 条件生成 | Classifier-free guidance、条件U-Net |

## 目录结构

```
differtion/
├── stage1_forward_process/   # 前向加噪过程
├── stage2_training/          # 训练目标与2D玩具实验
├── stage3_unet/              # U-Net与MNIST图像生成
├── stage4_sampling/          # DDPM/DDIM采样算法
├── stage5_advanced/          # 条件生成与进阶
├── utils/                    # 通用工具函数
└── requirements.txt
```

## 快速开始

```bash
pip install -r requirements.txt

# Stage 1: 可视化前向加噪过程
python stage1_forward_process/visualize.py
```

## 技术栈

- **PyTorch** — 模型实现与张量运算
- **NumPy** — 数学计算
- **Matplotlib** — 可视化
- **MNIST** — 图像生成实验数据集