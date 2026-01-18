# HindiGPT 🚀  
A Decoder-Only Transformer Language Model Trained from Scratch for Hindi.

HindiGPT-v1 is a custom 57.7M parameter decoder-only GPT architecture developed completely from scratch by Ajay Kumar.  
The project includes a domain-specific SentencePiece tokenizer, a RoPE-based multi-head self-attention transformer, and large-scale pretraining on high-quality Hindi corpora (IndicCorp, OSCAR web crawl, and Wikipedia).  

<div style="display:flex; gap:20px;">
  <img src="https://github.com/user-attachments/assets/a3eaff1c-8def-4819-90b7-00f8babec582" width="45%"/>
  <img src="https://github.com/user-attachments/assets/5c73e49d-ea66-46de-8006-6ae9e38afbb2" width="45%"/>
</div>

The model is fine-tuned for Hindi question–answering and deployed with a custom Gradio frontend on Hugging Face Spaces.

<img width="1722" height="807" alt="hindi_gpt_web_app" src="https://github.com/user-attachments/assets/cadb461e-dfe8-4ee4-93db-ecc41db78251" />

## 🔑 Key Highlights

- Transformer (GPT-style) implemented fully from scratch in PyTorch
- Custom Hindi SentencePiece BPE tokenizer (32K vocab)
- Decoder-only architecture with:
  - RMSNorm
  - SwiGLU Feed Forward Network
  - RoPE (Rotary Positional Embeddings)
  - Masked Multi-Head Self Attention
- Trained on 15+ GB of Hindi text
- ~1.35 Billion tokens processed during training
- Deployed on Hugging Face Spaces with Gradio UI

## 🧠 Model Configuration

| Parameter | Value |
|---------|------|
| Model Name | HindiGPT-v1 |
| Architecture | Decoder-only Transformer (GPT-style) |
| Vocabulary Size | 32,768 (SentencePiece BPE – Hindi only) |
| Layers | 12 |
| Hidden Size (d_model) | 512 |
| Attention Heads | 8 |
| Feed Forward Dim (d_ff) | 2048 |
| Max Sequence Length | 512 |
| Dropout | 0.1 |
| Activation | SwiGLU (SiLU-based gated FFN) |
| Positional Encoding | Learned |
| Weight Tying | Yes (Embedding ↔ LM Head) |
| **Total Parameters** | **57,709,056 (≈57M)** |
| **Trainable Parameters** | **57,709,056 (100%)** |
| **Total Tokens Trained** | **~1.35B (1,352,264,099)** |

## 🖥️ Training Hardware & Environment

| Component | Details |
|---------|--------|
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU |
| GPU Memory | 6 GB |
| CUDA Version | 12.5 |
| CPU | 20 Cores (28 Logical) |
| RAM | 16 GB |
| OS | Windows 11 |
| Python | CPython 3.13 |
| Framework | PyTorch (from scratch implementation) |

## ⚙️ Training Hyperparameters

| Parameter | Value |
|---------|------|
| Batch Size | 4 |
| Gradient Accumulation | 4 |
| Effective Batch Size | 16 |
| Max Training Steps | 300,000 |
| Optimizer | AdamW |
| Learning Rate (Max) | 6e-4 |
| Learning Rate (Min) | 6e-5 |
| Gradient Clipping | 1.0 |
| Evaluation Interval | 5,000 steps |
| Logging Interval | 50 steps |

## 📊 Training Metrics

### 📉 Loss
| Metric | Value |
|------|------|
| Final Training Loss | ~3.2 |
| Final Validation Loss | ~4.0 |
| Trend | Stable convergence |

---

### 🔢 Perplexity
| Metric | Value |
|------|------|
| Initial Perplexity | ~400 |
| Final Perplexity | **~53** |
| Behavior | Smooth monotonic decay |


<img width="1798" height="832" alt="training_graphs" src="https://github.com/user-attachments/assets/bfece955-9b69-4d2b-aac5-0af531db31b9" />

---

### 🚀 Throughput
| Metric | Value |
|------|------|
| Tokens / Second | ~15,000 |
| Total Tokens Seen | **1.35B** |
| Training Steps | ~6,000 |

## 📚 Training Data

The model was trained on a diverse and high-quality Hindi corpus:
Raw Hindi text was collected from multiple large-scale public sources and processed through a multi-stage streaming data cleaning pipeline.

| Dataset | Size | Description |
|------|------|------------|
| IndicCorp | 5.5 GB | Books and news articles |
| OSCAR Hindi | 9 GB | Web crawl (daily life Hindi) |
| Hindi Wikipedia | 188 MB | Clean encyclopedic text |

Cleaning steps included:
• Removal of non-Devanagari characters  
• Unicode normalization and whitespace normalization  
• Punctuation cleanup and normalization  
• NSFW and adult content filtering  
• Hindi digit normalization (०१२३४५६७८९ → 0123456789)  
• Minimum length filtering to remove low-information text  
• Wikipedia markup, templates, and metadata removal

All cleaning was performed in a streaming manner, enabling processing of multi-gigabyte corpora on limited hardware.
Data cleaning was performed in multiple streaming passes to ensure quality and scalability without loading the entire dataset into memory.


## 🔤 Custom Tokenizer

- Tokenizer: SentencePiece (BPE)
- Vocabulary Size: 32,768
- Character Coverage: 100%
- Trained on cleaned Hindi corpus
- Handles:
  - Conjunct characters
  - Matras
  - Informal + formal Hindi
    
### 🔄 Tokenization & Binary Conversion

Binary training shards: multiple `.bin` files  
After tokenizer training, the cleaned corpus was tokenized and converted into compact binary format for efficient training.
Key details:
• Each sentence is tokenized and terminated with an EOS token  
• Token IDs are stored as uint16 for memory efficiency  
• Data is written in fixed-size binary chunks  
• Memory-mapped binary files enable fast random access during training

This approach allows near-infinite random sampling during training while keeping RAM usage minimal.

The dataset was specifically curated for decoder-only language modeling and optimized for long-context autoregressive training.
 
### Decoder Block

```markdown
## 🏗 Architecture Overview

HindiGPT-v1 follows a decoder-only Transformer architecture inspired by GPT and LLaMA.

Each decoder block contains:

1. RMSNorm
2. Masked Multi-Head Self Attention with RoPE
3. Residual Connection
4. RMSNorm
5. SwiGLU Feed Forward Network
6. Residual Connection

```

![Decoder Block Architecture](https://github.com/user-attachments/assets/6725f6d2-fa7d-48cd-8048-bf40f3d71463)

![Masked Multi-Head Attention](https://github.com/user-attachments/assets/5d6f42fd-3a49-4267-9ba6-f6c32a82ed3f)

![SwiGLU Feed Forward Network](https://github.com/user-attachments/assets/ccca2d1f-ed0b-46c5-8df8-6b39af72905d)

![Causal Mask Flow](https://github.com/user-attachments/assets/f1c4acbb-7fa0-481c-976d-ed38c50d4040)

```
## 🌀 Rotary Positional Embeddings (RoPE)

Instead of absolute positional embeddings, RoPE is applied directly to query and key vectors.

Benefits:
- Better extrapolation to longer sequences
- Improved relative position awareness
- Used in modern LLMs like LLaMA

## ⚙ Training Details

The model is trained using standard causal language modeling with cross-entropy loss over the vocabulary.
Loss is computed as:
P(tokenₜ | token₁ … tokenₜ₋₁)

Logits and labels are flattened to efficiently compute loss over all tokens
in the batch.

- Optimizer: AdamW
- Learning Rate: 6e-4 → 6e-5 (cosine decay)
- Warmup Steps: 2000
- Gradient Accumulation: 4
- Gradient Clipping: 1.0
- Precision: FP32 (high matmul precision)

### 🔁 Gradient Accumulation

Due to memory constraints, gradient accumulation is used to simulate a larger
effective batch size.

• Micro-batch size: 4  
• Gradient accumulation steps: 4  
• Effective batch size: 16 sequences  

Loss is scaled down per micro-step and gradients are synchronized only after
accumulation, enabling stable large-context training on limited hardware.


### ⚡ Mixed Precision & Performance Optimization

Training is accelerated using automatic mixed precision (AMP) with `bfloat16`
and gradient scaling for numerical stability.

Additional optimizations include:
• `torch.compile` with max-autotune mode
• High-precision matrix multiplication
• Gradient norm clipping to prevent exploding gradients

### 📉 Learning Rate Schedule

A cosine decay learning rate schedule with linear warmup is used:

• Linear warmup for the first 2,000 steps  
• Cosine decay until max training steps  
• Separate max and min learning rates  

This improves early training stability and prevents late-stage overfitting.

### 🧪 Evaluation Strategy

Validation is performed periodically on a held-out dataset.

• Average loss computed over multiple validation batches  
• Perplexity is reported as the primary evaluation metric  
• Evaluation runs in inference mode with no gradient computation


### 📊 Experiment Tracking

All training metrics are logged using Weights & Biases (wandb), including:

• Training loss  
• Validation loss & perplexity  
• Learning rate  
• GPU memory usage  
• Token throughput (tokens/sec)  
• Total tokens processed  

This enables full reproducibility and performance analysis.

```

```
## 🚀 Deployment

The model is deployed on Hugging Face Spaces using Gradio.

- Backend: PyTorch
- Frontend: Gradio
- Hardware: CPU (GPU not enabled)
```

![Live Demo](https://github.com/user-attachments/assets/59c015b5-5b54-4883-8dfe-e99f4fecb250)


🔗 Live Demo:
https://musk12-hindi-gpt-model-built-from-scratch.hf.space/

```
## ⚠ Limitations

- CPU-only deployment leads to slower inference
- Model size is relatively small compared to commercial LLMs
- Focused primarily on Hindi language


This project was built as a full end-to-end learning journey from transformer theory to model training and deployment.

