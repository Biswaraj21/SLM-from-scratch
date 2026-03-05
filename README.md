# CoreGPT – GPT-Style Small Language Model from Scratch

CoreGPT is a **GPT-style transformer language model implemented completely from scratch in PyTorch**.  
The objective of this project is to understand how modern **Large Language Models (LLMs)** work internally by building each component step-by-step without relying on high-level NLP frameworks.

This project implements the full NLP pipeline including tokenization, dataset creation, transformer architecture, training loop, and text generation.

---

## Project Motivation

Modern models like **GPT, LLaMA, and Gemini** rely on the Transformer architecture.  
Most tutorials only show how to *use* these models, but not how they work internally.

This project focuses on implementing the core building blocks of LLMs from first principles to gain a deeper understanding of:

- Tokenization
- Transformer architecture
- Attention mechanisms
- Autoregressive language modeling
- Training dynamics of language models

---

## Features

- Character-level tokenizer
- Byte Pair Encoding (BPE) tokenizer
- Sliding window dataset generation
- Token embeddings
- Positional embeddings
- Multi-head self-attention
- Causal masking for autoregressive training
- Feed-forward transformer layers
- Residual connections and layer normalization
- Stacked transformer blocks
- GPT-style language model architecture
- Training pipeline using cross-entropy loss
- Autoregressive text generation

---

## Model Architecture

The model follows a simplified GPT architecture:

```
Input Tokens
     ↓
Token Embedding
     +
Positional Embedding
     ↓
Transformer Blocks (N layers)
     ↓
LayerNorm
     ↓
Linear Projection
     ↓
Vocabulary Logits
```

Each Transformer Block contains:

```
Multi-Head Self Attention
        ↓
Residual Connection
        ↓
LayerNorm
        ↓
Feed Forward Network
        ↓
Residual Connection
```

---

## Attention Mechanism

Self-attention computes relationships between tokens using **Query, Key, Value projections**.

```
Attention(Q,K,V) = softmax(QKᵀ / √d) V
```

Causal masking ensures that the model **cannot attend to future tokens** during training.

---

## Repository Structure

```
SLM-from-scratch
│
├── tokenizer
│   ├── simple_tokenizer.py
│   └── bpe_tokenizer.py
│
├── model
│   ├── embedding.py
│   ├── positional_embedding.py
│   ├── self_attention.py
│   ├── multi_head_attention.py
│   ├── feed_forward.py
│   ├── transformer_block.py
│   └── gpt_model.py
│
├── training
│   └── train.py
│
├── generate.py
└── README.md
```

---

## Training Objective

The model is trained using **next-token prediction**.

Example:

```
Input  : "The cat sat on the"
Target : "cat sat on the mat"
```

The training objective is to minimize **Cross Entropy Loss** between predicted tokens and the true next tokens.

---

## Example Training Output

```
Epoch 1, Loss: 0.0839
Epoch 2, Loss: 0.0496
Epoch 3, Loss: 0.0481
```

A decreasing loss indicates the model is learning language patterns.

---

## Text Generation

The model generates text using **autoregressive decoding**.

Steps:

1. Encode prompt
2. Predict next token
3. Append predicted token
4. Repeat

Example prompt:

```
Once upon
```

Example output:

```
Once upon a time there was a little boy who lived near a forest.
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/CoreGPT.git
cd CoreGPT
```

Install dependencies:

```bash
pip install torch numpy tqdm
```

---

## Training the Model

Run the training script:

```bash
python -m training.train
```

The trained model weights will be saved as:

```
model.pt
```

---

## Generate Text

Run the generation script:

```bash
python generate.py
```

Modify the prompt inside `generate.py` to generate custom text.

---

## Technologies Used

- Python
- PyTorch
- Transformer Architecture
- Multi-Head Self Attention
- Byte Pair Encoding (BPE)
- Natural Language Processing
- Deep Learning

---

## Future Improvements

Potential enhancements for the project:

- Training on larger datasets (WikiText, TinyStories)
- GPU training support
- Top-k and temperature sampling
- Attention visualization
- Perplexity evaluation
- HuggingFace tokenizer compatibility

---

## Learning Outcomes

This project demonstrates practical understanding of:

- Transformer internals
- Attention mechanisms
- Tokenization techniques
- Language model training
- Autoregressive text generation

---

## Author

**Biswaraj Bhattacharyya**  
M.Tech – Artificial Intelligence & Machine Learning