# 数学推理

## 课件幻灯片

```pdf
chapter4/math.pdf
```

## 教程内容

> 导读：数学推理能力是大模型智能水平的重要体现。本章将介绍如何通过SFT（监督微调）和思维链（CoT）等技术增强大模型的数学解题能力。

## 实验目的
- 了解数学相关的数据集的清理和预处理
- 了解微调大模型的基本方法
- 了解使用大模型进行推理的基本方法
- 了解评测模型数学能力的基本方法

## 环境要求

- Python 3.8+
- PyTorch
- Transformers
- Datasets
- vllm
- 至少40GB显存的GPU

## 实验文档

1. 打开Jupyter notebook：
```bash
jupyter notebook sft_math.ipynb
```

2. notebook包含几个主要部分：
   - 数据集下载以及预处理
   - 模型加载、训练
   - 模型生成、评测

3. 具体操作和说明请参考notebook。

## 输出文件

训练过程会产生训练的checkpoint文件，生成完成后会保存生成结果。

## 注意事项

- 模型和数据需要从huggingface加载，国内可以通过https://hf-mirror.com 镜像进行下载。
- 确保有足够的磁盘空间保存原始模型文件、数据文件以及训练后的checkpoint(请预留至少50GB空间)
- 需要使用至少40GB显存的GPU显卡
