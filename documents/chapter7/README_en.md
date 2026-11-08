# LLM Text Steganography

This project implements text steganography based on large language models (LLMs), using the GPT-2 model to hide information in generated text. The project includes two encoding methods: Huffman coding and Fixed Length Coding (FLC).

## Features

- Implements steganography in GPT-2 text generation
- Supports two encoding methods:
  - Huffman coding
  - Fixed Length Coding (FLC)
- Generates both steganographic text and normal text for comparison

## Environment Requirements

- Python 3.8+
- PyTorch
- Transformers
- CUDA (optional, for GPU acceleration)

## Installation

Install the dependencies:

```bash
pip install torch # If using a GPU, make sure to install the GPU version of torch
pip install transformers
pip install jupyter
```

## Usage

1. Open the Jupyter notebook:
```bash
jupyter notebook llm_stega.ipynb
```

2. The notebook contains three main parts:
   - Huffman coding implementation
   - Fixed Length Coding (FLC) implementation
   - Text generation and steganography

3. Generate steganographic text:
   - Choose the encoding method (Huffman coding or FLC)
   - Set the parameters (the encoding value k)
   - Run the main function with a custom prompt

Example code:
```python
# Initialize the steganography handler
k = 2
handle = Huffman(k=2**k, bits=bits)
# Or use FLC
# handle = FLC(k=k, bits=bits)

# Generate text with steganography
prompt = "Hello, I'm"
stega_output, normal_output = generate_text_with_steganography(
    model, tokenizer, prompt, handle
)
```

## Output Files

The program generates two text files:
- `outputs-gpt2-stega.txt`: steganographic text
- `outputs-gpt2-normal.txt`: the normal text generated without steganographic constraints

## Notes

- The GPT-2 model is downloaded automatically on the first run. In mainland China, you can also pre-download the GPT-2 model through the https://hf-mirror.com mirror, and you may also use other large models.
- Make sure you have enough disk space (the model is about 500MB)
- A CUDA-capable GPU is recommended for better performance
- The length of the hidden message affects the length of the generated text

## How It Works

1. Steganography process:
   - Convert the secret message into a binary sequence
   - Generate text with the GPT-2 model
   - Embed the information bits into the generated text using the encoding method

2. Decoding process:
   - Use the same GPT-2 model and context
   - Extract the hidden information bits from the text using the decoding method
   - Convert the binary sequence back to the original message
