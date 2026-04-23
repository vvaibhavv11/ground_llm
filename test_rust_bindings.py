from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import regex as re

import ground_llm

ROOT = Path(__file__).resolve().parent
Merge = Tuple[Tuple[int, int], int]
VocabList = Dict[int, List[int]]
EXPECTED_EXPORTS = (
    "build_info",
    "encode_train",
    "encode",
    "decode_string",
    "save_vocab_list",
)
# directly taken from the tiktoken library by openai https://github.com/openai/tiktoken but the tiktoken does not contain the training code
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


def load_merges_record(path: str | Path) -> list[Merge]:
    raw_merges: list[dict[str, Any]] = json.loads(
        Path(path).read_text(encoding="utf-8")
    )
    return [
        ((int(entry["pair"][0]), int(entry["pair"][1])), int(entry["id"]))
        for entry in raw_merges
    ]


def load_vocab_list(path: str | Path) -> VocabList:
    raw_vocab: list[dict[str, Any]] = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        int(entry["id"]): [int(byte) for byte in entry["bytes"]] for entry in raw_vocab
    }


def main() -> int:
    merges = load_merges_record(ROOT / "merges_record.json")
    vocab_list = load_vocab_list(ROOT / "vocab_list.json")

    # split_text = split_text_file("./dataset/tokebnizer_dataset.txt")

    encode_test = """
    use std::cmp::Ordering;
    use std::collections::BinaryHeap;

    // Wrapper to reverse byte comparison
    #[derive(Eq, PartialEq)]
    struct ReversedBytes(Vec<u8>);

    impl Ord for ReversedBytes {
        fn cmp(&self, other: &Self) -> Ordering {
            // Reverse comparison (like Python __lt__)
            other.0.cmp(&self.0)
        }
    }

    impl PartialOrd for ReversedBytes {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    // Heap entry
    #[derive(Eq, PartialEq)]
    struct HeapEntry {
        freq: usize,
        lex: (ReversedBytes, ReversedBytes),
        pair: (usize, usize),
    }

    impl Ord for HeapEntry {
        fn cmp(&self, other: &Self) -> Ordering {
            // First compare frequency (max heap)
            match self.freq.cmp(&other.freq) {
                Ordering::Equal => {
                    // Then lexicographically compare reversed bytes
                    match self.lex.0.cmp(&other.lex.0) {
                        Ordering::Equal => self.lex.1.cmp(&other.lex.1),
                        ord => ord,
                    }
                }
                ord => ord,
            }
        }
    }

    impl PartialOrd for HeapEntry {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    // Example usage
    fn main() {
        let vocab: Vec<Vec<u8>> = vec![
            b"a".to_vec(),
            b"b".to_vec(),
            b"c".to_vec(),
        ];

        let pair_frequencies = vec![
            ((0, 1), 10),
            ((1, 2), 5),
            ((0, 2), 10),
        ];

        let mut heap = BinaryHeap::new();

        for (pair, freq) in pair_frequencies {
            let entry = HeapEntry {
                freq,
                lex: (
                    ReversedBytes(vocab[pair.0].clone()),
                    ReversedBytes(vocab[pair.1].clone()),
                ),
                pair,
            };
            heap.push(entry);
        }

        // Pop elements (highest priority first)
        while let Some(entry) = heap.pop() {
            println!(
                "pair: {:?}, freq: {}, lex: ({:?}, {:?})",
                entry.pair,
                entry.freq,
                entry.lex.0 .0,
                entry.lex.1 .0
            );
        }
    }
    """
    encode_split = TOKEN_RE.findall(encode_test)
    # merges = ground_llm.encode_train(split_text)
    encoded = ground_llm.encode(encode_split, merges)
    print(encoded)
    # vocab_file = ROOT / "vocab_list.json"
    # saved_vocab = json.loads(vocab_file.read_text(encoding="utf-8"))
    # vocab_list = {entry["id"]: entry["bytes"] for entry in saved_vocab}
    decoded = ground_llm.decode_string(encoded, vocab_list)
    print(decoded)

    merges_file = ROOT / "merges_record.json"
    print(f"build_info={ground_llm.build_info()}")
    print(f"module={ground_llm.__name__}")
    # print(f"merge_count={len(merges)}")
    # print(f"encoded={encoded}")
    # print(f"last_merge_id={merges[-1][1] if merges else None}")
    print(f"decoded_ok={decoded == encode_test}")
    # print(f"merges_file_exists={merges_file.exists()}")
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

    return 0 if decoded == encode_test else 1


if __name__ == "__main__":
    sys.exit(main())
