# ground_llm

`ground_llm` is a project to build a Large Language Model (LLM) from the ground up. The goal is to implement every component of the LLM pipeline—from model architecture to training—from scratch.

The **tokenizer** has been extracted into its own standalone package: [`bpe_rs`](https://github.com/vvaibhavv11/bpe_rs). It is a high-performance Byte Pair Encoding (BPE) tokenizer written in Rust with Python bindings.

---

## LLM Roadmap

The project will be developed in the following phases:

- [x] **Phase 1: Tokenizer** (Completed — see [`bpe_rs`](https://github.com/vvaibhavv11/bpe_rs))
  - BPE algorithm implementation.
  - Rust-based performance optimization.
  - Vocab and Merge rule persistence.
- [x] **Phase 2: Model Architecture**
  - Pre-norm decoder-only Transformer with a configurable number of layers.
  - Embedding layers, causal Multi-Head Attention, and Gated SiLU Feed-Forward Networks.
  - RMSNorm with learnable gain, rotary positional encoding (RoPE).
- [x] **Phase 3: Training Framework**
  - Full backpropagation for every layer, verified by a finite-difference gradient check.
  - Random-window batching with a train/validation split.
  - Cross-entropy loss, AdamW, gradient clipping, warmup + cosine LR schedule.
  - Training loop with best-validation checkpointing.
- [ ] **Phase 4: Inference & Evaluation**
  - Generation strategies (Greedy, Beam Search, Top-K/Top-P).
  - Benchmarking and evaluation metrics.

---

## Getting Started

### Installation
This project uses `maturin` for building Rust-based Python packages.

**Prerequisites:**
- Rust toolchain (`cargo`, `rustc`)
- Python 3.8+
- `maturin` (`pip install maturin`)

**Build and Install:**
```bash
maturin develop
```

The tokenizer is now a separate package: [`bpe_rs`](../bpe_rs). Install it alongside `ground_llm`:
```bash
pip install bpe_rs
```

### Training

```python
import ground_llm

model = ground_llm.Model(vocab_size=8000, dimensions=256, n_heads=4, n_layers=4, max_seq_len=128)
history = model.train(tokens, steps=2000, batch_size=8, lr=1e-3, checkpoint_path="checkpoint.bin")
ids = model.generate(prompt_tokens, max_new_tokens=100, temperature=0.8, top_k=40)

model = ground_llm.Model.load("checkpoint.bin")
```

`tokens` is a flat list of token ids. `examples/train.py` does the whole thing end to end, tokenizing a text file with `bpe_rs`:

```bash
python examples/train.py corpus.txt --steps 2000
```

Ctrl-C during `train` stops after the current step; the model keeps the weights learned so far.

## Project Structure

- `src/lib.rs`: Python module definition and function exports.
- `src/matrix.rs`: Matrix operations (matmul, RMSNorm, RoPE, softmax, etc.).
- `src/mlp.rs`: Gated SiLU feed-forward network.
- `src/param.rs`: Trainable parameters and linear-layer forward/backward helpers.
- `src/optim.rs`: AdamW optimizer and gradient clipping.
- `src/train.rs`: Transformer model, backprop, loss, training loop, checkpointing, and generation.
- `examples/train.py`: End-to-end training script using `bpe_rs` for tokenization.
- `Cargo.toml`: Rust dependencies.
- `pyproject.toml`: Python project configuration.