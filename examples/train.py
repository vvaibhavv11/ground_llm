"""Train a small ground_llm model on a text file and sample from it.

    python examples/train.py path/to/corpus.txt --steps 2000

Tokenization uses bpe_rs with the 8k merges/vocab stored in this repo.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import bpe_rs
from bpe_rs.pre_tokenize import split_text

import ground_llm

ROOT = Path(__file__).resolve().parent.parent


def load_tokenizer(merges_path: Path, vocab_path: Path):
    merges = [
        ((int(e["pair"][0]), int(e["pair"][1])), int(e["id"]))
        for e in json.loads(merges_path.read_text(encoding="utf-8"))
    ]
    vocab = {
        int(e["id"]): [int(b) for b in e["bytes"]]
        for e in json.loads(vocab_path.read_text(encoding="utf-8"))
    }

    def encode(text: str) -> list[int]:
        return bpe_rs.encode(split_text(text), merges)

    def decode(ids: list[int]) -> str:
        return bpe_rs.decode_string(ids, vocab)

    return encode, decode, max(vocab) + 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("corpus", type=Path, help="UTF-8 text file to train on")
    parser.add_argument("--merges", type=Path, default=ROOT / "merges_record_8000.json")
    parser.add_argument("--vocab", type=Path, default=ROOT / "vocab_list_8000.json")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoint.bin"))
    parser.add_argument("--resume", action="store_true", help="start from --checkpoint")

    model_args = parser.add_argument_group("model")
    model_args.add_argument("--dimensions", type=int, default=256)
    model_args.add_argument("--n-heads", type=int, default=4)
    model_args.add_argument("--n-layers", type=int, default=4)
    model_args.add_argument("--mlp-hidden", type=int, default=688)
    model_args.add_argument("--max-seq-len", type=int, default=128)

    train_args = parser.add_argument_group("training")
    train_args.add_argument("--steps", type=int, default=2000)
    train_args.add_argument("--batch-size", type=int, default=8)
    train_args.add_argument("--lr", type=float, default=1e-3)
    train_args.add_argument("--warmup-steps", type=int, default=100)
    train_args.add_argument("--eval-every", type=int, default=100)
    train_args.add_argument("--log-every", type=int, default=10)

    gen_args = parser.add_argument_group("sampling")
    gen_args.add_argument("--prompt", default="The")
    gen_args.add_argument("--max-new-tokens", type=int, default=100)
    gen_args.add_argument("--temperature", type=float, default=0.8)
    gen_args.add_argument("--top-k", type=int, default=40)
    args = parser.parse_args()

    encode, decode, vocab_size = load_tokenizer(args.merges, args.vocab)

    tokens = encode(args.corpus.read_text(encoding="utf-8"))
    print(f"corpus: {len(tokens):,} tokens")

    if args.resume:
        model = ground_llm.Model.load(str(args.checkpoint))
    else:
        model = ground_llm.Model(
            vocab_size=vocab_size,
            dimensions=args.dimensions,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            mlp_hidden=args.mlp_hidden,
            max_seq_len=args.max_seq_len,
        )
    print(model)

    try:
        model.train(
            tokens,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            warmup_steps=args.warmup_steps,
            eval_every=args.eval_every,
            log_every=args.log_every,
            checkpoint_path=str(args.checkpoint),
        )
    except KeyboardInterrupt:
        print("\ninterrupted; sampling from the current weights")

    out = model.generate(
        encode(args.prompt),
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print("-" * 40)
    print(decode(out))


if __name__ == "__main__":
    main()
