# Deep Interest Network (DIN) — PyTorch Implementation

[English](#english) | [中文](#中文)

---

## English

### Overview

A PyTorch implementation of **Deep Interest Network (DIN)** for click-through rate (CTR) prediction, based on the paper [Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/abs/1706.06978) (Zhou et al., KDD 2018).

DIN models user interest diversity by applying an attention mechanism to historical behavior sequences, assigning different weights to different historical items based on their relevance to the target item.

### Project Structure

```
├── model.py        # DIN model: ActivationUnit, MLP, DIN
├── data.py         # Data loading, train/test split, DataLoader collate
├── train.py        # Training loop and evaluation (AUC/GAUC)
├── main.py         # Entry point
├── config.py       # Hyperparameter configuration
├── gpu.py          # GPU selection utility
└── best.pth        # Pretrained model weights
```

### Model Architecture

- **Embedding Layer**: Item embedding + Category embedding (concatenated)
- **Activation Unit**: Attention mechanism with MLP (PReLU activation) to compute relevance scores between target item and historical items
- **Pooling**: Weighted sum of historical item embeddings using attention weights
- **FCN**: Fully connected network `[80, 40] → 1` with PReLU activation
- **Item Bias**: Per-item bias term

### Dataset

Amazon Electronics 5-core dataset:
- `reviews_Electronics_5.json`: User review records
- `meta_Electronics.json`: Item metadata (categories)

### Quick Start

```bash
# Install dependencies
pip install torch jsonlines scikit-learn

# Prepare data
# Place reviews_Electronics_5.json and meta_Electronics.json in data/ directory

# Train
python main.py
```

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 4096 | Training batch size |
| `lr` | 0.001 | Learning rate (Adam) |
| `epochs` | 100 | Number of training epochs |
| `seq_len` | 100 | Max history sequence length |
| `embedding_dim` | 64 | Embedding dimension |
| `max_category_len` | 1 | Max categories per item |
| `weight_decay` | 1e-4 | L2 regularization |

### Evaluation Metrics

- **AUC**: Global ranking accuracy across all test samples
- **GAUC**: Per-user pairwise ranking accuracy, averaged across all users

---

## 中文

### 概述

基于论文 [Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/abs/1706.06978)（Zhou et al., KDD 2018）的 **PyTorch 实现**。

DIN 通过注意力机制建模用户兴趣多样性，根据目标物品与历史行为的相关性，为不同的历史物品分配不同的权重。

### 项目结构

```
├── model.py        # DIN 模型：ActivationUnit、MLP、DIN
├── data.py         # 数据加载、训练/测试集划分、DataLoader collate
├── train.py        # 训练循环与评估（AUC/GAUC）
├── main.py         # 入口文件
├── config.py       # 超参数配置
├── gpu.py          # GPU 选择工具
└── best.pth        # 预训练模型权重
```

### 模型架构

- **Embedding 层**：物品 Embedding + 类别 Embedding（拼接）
- **Activation Unit**：注意力机制，使用 MLP（PReLU 激活）计算目标物品与历史物品的相关性分数
- **池化层**：使用注意力权重对历史物品 Embedding 进行加权求和
- **FCN**：全连接网络 `[80, 40] → 1`，PReLU 激活
- **Item Bias**：每个物品的偏置项

### 数据集

Amazon Electronics 5-core 数据集：
- `reviews_Electronics_5.json`：用户评论记录
- `meta_Electronics.json`：物品元数据（类别信息）

### 快速开始

```bash
# 安装依赖
pip install torch jsonlines scikit-learn

# 准备数据
# 将 reviews_Electronics_5.json 和 meta_Electronics.json 放入 data/ 目录

# 训练
python main.py
```

### 超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `batch_size` | 4096 | 训练 batch 大小 |
| `lr` | 0.001 | 学习率（Adam） |
| `epochs` | 100 | 训练轮数 |
| `seq_len` | 100 | 历史序列最大长度 |
| `embedding_dim` | 64 | Embedding 维度 |
| `max_category_len` | 1 | 每个物品最大类别数 |
| `weight_decay` | 1e-4 | L2 正则化系数 |

### 评估指标

- **AUC**：全局排序准确率
- **GAUC**：逐用户 Pairwise 排序准确率的均值
