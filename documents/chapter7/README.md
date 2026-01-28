# 隐写术

## 课件幻灯片

```pdf
chapter7/stega.pdf
```

## 教程内容

> 导读：隐写术(Steganography)是一门将秘密信息隐藏在非秘密载体中的技术。在大模型时代，如何将隐写术与文本生成结合？本章将探讨基于LLM的语言隐写技术。

本项目实现了基于大型语言模型（LLM）的文本隐写术，使用GPT-2模型在生成的文本中隐藏信息。项目包含两种编码方法：霍夫曼编码（Huffman Coding）和固定长度编码（Fixed Length Coding, FLC）。

## 功能特点

- 在GPT-2文本生成中实现隐写术
- 支持两种编码方法：
  - 霍夫曼编码（Huffman Coding）
  - 固定长度编码（Fixed Length Coding, FLC）
- 同时生成隐写文本和普通文本用于对比

## 环境要求

- Python 3.8+
- PyTorch
- Transformers
- CUDA（可选，用于GPU加速）

## 安装步骤

安装依赖包：

```bash
pip install torch # 如果是GPU，注意指定torch是GPU版本
pip install transformers
pip install jupyter
```

## 使用方法

1. 打开Jupyter notebook：
```bash
jupyter notebook llm_stega.ipynb
```

2. notebook包含三个主要部分：
   - 霍夫曼编码实现
   - 固定长度编码（FLC）实现
   - 文本生成与隐写

3. 生成隐写文本：
   - 选择编码方法（霍夫曼编码或FLC）
   - 设置参数（编码的k值）
   - 使用自定义提示词运行主函数

示例代码：
```python
# 初始化隐写处理器
k = 2
handle = Huffman(k=2**k, bits=bits)
# 或者使用FLC
# handle = FLC(k=k, bits=bits)

# 生成带隐写的文本
prompt = "Hello，I'm"
stega_output, normal_output = generate_text_with_steganography(
    model, tokenizer, prompt, handle
)
```

## 输出文件

程序会生成两个文本文件：
- `outputs-gpt2-stega.txt`：隐写文本
- `outputs-gpt2-normal.txt`：没有隐写约束情况下原本正常生成的文本

## 注意事项

- GPT-2模型会在首次运行时自动下载，国内也可以先通过https://hf-mirror.com 镜像下载好GPT-2模型，也可以使用其他大模型。
- 确保有足够的磁盘空间（模型约500MB）
- 建议使用支持CUDA的GPU以获得更好的性能
- 隐写信息长度会影响生成的文本长度

## 工作原理

1. 隐写过程：
   - 将秘密信息转换为二进制序列
   - 使用GPT-2模型生成文本
   - 通过编码方法将信息位嵌入到生成的文本中

2. 解码过程：
   - 使用相同的GPT-2模型和上下文
   - 通过解码方法从文本中提取隐藏的信息位
   - 将二进制序列转换回原始信息
