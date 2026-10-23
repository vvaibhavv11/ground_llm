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
- [ ] **Phase 2: Model Architecture**
  - Implementation of the Transformer architecture.
  - Embedding layers, Multi-Head Attention, and Feed-Forward Networks.
  - Positional encoding.
- [ ] **Phase 3: Training Framework**
  - Data loading and batching pipelines.
  - Loss functions and optimizer implementation.
  - Training loop with checkpointing.
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

## Project Structure

- `src/lib.rs`: Python module definition and function exports.
- `src/matrix.rs`: Matrix operations (matmul, RMSNorm, RoPE, softmax, etc.).
- `src/mlp.rs`: Gated SiLU feed-forward network.
- `src/train.rs`: Transformer model architecture and training.
- `Cargo.toml`: Rust dependencies.
- `pyproject.toml`: Python project configuration.