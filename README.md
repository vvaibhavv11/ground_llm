# ground_llm

`ground_llm` is a project to build a Large Language Model (LLM) from the ground up. The goal is to implement every component of the LLM pipeline—from data preprocessing and tokenization to model architecture and training—from scratch.

## 🚧 Current Status: Tokenizer Phase

Currently, the project is in its initial stages. The first major milestone—the **Tokenizer**—has been completed.

### Tokenizer Implementation
The tokenizer is implemented as a high-performance Byte Pair Encoding (BPE) system. It is written in Rust for maximum efficiency and exposed as a Python module using PyO3.

**Key Features of the Tokenizer:**
- **Fast BPE Training**: Efficiently learns merge rules from training data.
- **Parallelized Statistics**: Utilizes the `rayon` crate to parallelize the computation of pair frequencies.
- **Python Bindings**: Seamless integration with Python for easy use in the rest of the LLM pipeline.

---

## 🗺️ LLM Roadmap

The project will be developed in the following phases:

- [x] **Phase 1: Tokenizer** (Completed)
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

## 🛠️ Getting Started with the Tokenizer

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

### Usage Examples

#### Training the Tokenizer
```python
import ground_llm

chunks = ["hello world", "hello rust", "bpe tokenizer is fast"]
merges = ground_llm.encode_train(chunks)
print(f"Learned {len(merges)} merge rules.")
```

#### Encoding Text
```python
import ground_llm
import json

with open("merges_record.json", "r") as f:
    merges_data = json.load(f)

merges = [(tuple(m["pair"]), m["id"]) for m in merges_data]
tokens = ground_llm.encode(["hello rust"], merges)
print(f"Tokens: {tokens}")
```

#### Decoding Tokens
```python
import ground_llm
import json

with open("vocab_list.json", "r") as f:
    vocab_list_data = json.load(f)

vocab_list = {item["id"]: item["bytes"] for item in vocab_list_data}
decoded_text = ground_llm.decode_string([101, 256, 102], vocab_list)
print(f"Decoded text: {decoded_text}")
```

## Project Structure

- `src/lib.rs`: Python module definition and function exports.
- `src/encoder.rs`: Core BPE logic.
- `Cargo.toml`: Rust dependencies.
- `pyproject.toml`: Python project configuration.
- `ground_llm.pyi`: Type hints for the Python module.
