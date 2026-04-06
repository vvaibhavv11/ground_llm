from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import regex as re

import ground_llm

ROOT = Path(__file__).resolve().parent
EXPECTED_EXPORTS = (
    "build_info",
    "encode_train",
    "encode",
    "decode_string",
    "save_vocab_list",
)
# directly taken from the tiktoken library by openai
pat_str = "|".join(
    [
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""\p{N}{1,3}""",
        r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
        r"""\s*[\r\n]+""",
        r"""\s+(?!\S)""",
        r"""\s+""",
    ]
)
TOKEN_RE = re.compile(pat_str)


def summarize_text(text: str, preview: int = 120) -> str:
    compact = " ".join(text.split())
    if len(compact) <= preview:
        return compact
    return f"{compact[:preview]}..."


def split_text_file(text_file: str | Path) -> list[str]:
    return TOKEN_RE.findall(Path(text_file).read_text(encoding="utf-8"))


def main() -> int:
    split_text = split_text_file("./dataset/tokebnizer_dataset.txt")

    encode_test = "hello world"
    # encode_split = TOKEN_RE.findall(encode_test)
    merges = ground_llm.encode_train(split_text)
    # encoded = ground_llm.encode(encode_split, merges)
    # vocab_file = ROOT / "vocab_list.json"
    # saved_vocab = json.loads(vocab_file.read_text(encoding="utf-8"))
    # vocab_list = {entry["id"]: entry["bytes"] for entry in saved_vocab}
    # decoded = ground_llm.decode_string(encoded, vocab_list)

    merges_file = ROOT / "merges_record.json"
    print(f"build_info={ground_llm.build_info()}")
    print(f"module={ground_llm.__name__}")
    print(f"merge_count={len(merges)}")
    # print(f"encoded={encoded}")
    print(f"last_merge_id={merges[-1][1] if merges else None}")
    # print(f"decoded_ok={decoded == encode_test}")
    print(f"merges_file_exists={merges_file.exists()}")
    # print(f"vocab_file_exists={vocab_file.exists()}")

    # if merges_file.exists():
    #     saved = json.loads(merges_file.read_text(encoding="utf-8"))
    #     print(f"saved_merge_count={len(saved)}")
    #     if saved:
    #         print(f"first_saved_merge={saved[0]}")
    #         print(f"last_saved_merge={saved[-1]}")

    # if vocab_file.exists():
    #     print(f"saved_vocab_count={len(saved_vocab)}")
    #     if saved_vocab:
    #         print(f"first_saved_vocab={saved_vocab[0]}")
    #         print(f"last_saved_vocab={saved_vocab[-1]}")

    return 0 if encode_test == encode_test else 1


if __name__ == "__main__":
    sys.exit(main())
