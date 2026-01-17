# HindiGPT 🚀  
A Decoder-Only Transformer Language Model Trained from Scratch for Hindi

HindiGPT-v1 is a custom decoder-only GPT architecture developed completely from scratch by Ajay Kumar.  
The project includes a domain-specific SentencePiece tokenizer, a RoPE-based multi-head self-attention transformer, and large-scale pretraining on high-quality Hindi corpora (IndicCorp, OSCAR web crawl, and Wikipedia).  
The model is fine-tuned for Hindi question–answering and deployed with a custom Gradio frontend on Hugging Face Spaces.

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
| Vocabulary Size | 32,768 |
| Layers | 12 |
| Hidden Size (d_model) | 512 |
| Attention Heads | 8 |
| Feed Forward Dim | 2048 |
| Sequence Length | 512 |
| Dropout | 0.1 |
| Total Tokens Trained | ~1.35B (1,352,264,099) |

## 📚 Training Data

The model was trained on a diverse and high-quality Hindi corpus:

| Dataset | Size | Description |
|------|------|------------|
| IndicCorp | 5.5 GB | Books and news articles |
| OSCAR Hindi | 9 GB | Web crawl (daily life Hindi) |
| Hindi Wikipedia | 188 MB | Clean encyclopedic text |

All datasets were cleaned, deduplicated, and normalized before training.


## 🔤 Custom Tokenizer

- Tokenizer: SentencePiece (BPE)
- Vocabulary Size: 32,768
- Character Coverage: 100%
- Trained on cleaned Hindi corpus
- Handles:
  - Conjunct characters
  - Matras
  - Informal + formal Hindi
 
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





## 🚀 Deployment

The model is deployed on Hugging Face Spaces using Gradio.

- Backend: PyTorch
- Frontend: Gradio
- Hardware: CPU (GPU not enabled)

🔗 Live Demo:
https://musk12-hindi-gpt-model-built-from-scratch.hf.space/


## ⚠ Limitations

- CPU-only deployment leads to slower inference
- Model size is relatively small compared to commercial LLMs
- Focused primarily on Hindi language


This project was built as a full end-to-end learning journey from transformer theory to model training and deployment.



