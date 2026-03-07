# LLaVA-from-scratch

LLaVA (Large Language and Vision Assistant) implementation from scratch.

## Setup

### 1. Install dependencies

```bash
uv sync
```

### 2. Set Hugging Face Token

To avoid rate limits, set your Hugging Face token:

```bash
export HF_TOKEN="your_token_here"
```

Get your token from https://huggingface.co/settings/tokens

### 3. Train the projector

```bash
uv run python train_projector.py
```

### 4. Train LLaVA

```bash
uv run python train_llava.py
```

## Memory Optimization

The training scripts include:
- **Gradient Accumulation**: Effective batch size with reduced memory
- **Gradient Checkpointing**: 40-50% memory reduction
- **LoRA**: Fine-tune only small adapter layers

If you encounter OOM errors, reduce `batch_size` in the training script.
