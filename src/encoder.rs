use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fs;

use pyo3::exceptions::PyIOError;
use pyo3::prelude::*;
use rayon::prelude::*;

const MERGES_RECORD_FILE: &str = "merges_record.json";
const VOCAB_LIST_FILE: &str = "vocab_list.json";
const MAX_MERGES: usize = 50_000;

type Pair = (u16, u16);
type PairFrequency = (usize, Pair);
type Merge = ((u16, u16), u16);
type WeightedChunk = (Vec<u16>, usize);
type VocabList = HashMap<u16, Vec<u8>>;

fn collapse_chunk_counts(chunks: impl IntoIterator<Item = WeightedChunk>) -> Vec<WeightedChunk> {
    let mut counts: HashMap<Vec<u16>, usize> = HashMap::new();
    for (chunk, weight) in chunks {
        *counts.entry(chunk).or_insert(0) += weight;
    }
    counts.into_iter().collect()
}

fn get_stat(raw_utf8: &[WeightedChunk]) -> HashMap<Pair, usize> {
    raw_utf8
        .par_iter()
        .map(|(chunk, weight)| {
            let mut counts: HashMap<Pair, usize> = HashMap::new();
            for pairs in chunk.windows(2) {
                *counts.entry((pairs[0], pairs[1])).or_insert(0) += *weight;
            }
            counts
        })
        .reduce(HashMap::new, |mut aggregate, counts| {
            for (pair, count) in counts {
                *aggregate.entry(pair).or_insert(0) += count;
            }
            aggregate
        })
}

fn build_pair_heap(counts: &HashMap<Pair, usize>) -> BinaryHeap<PairFrequency> {
    let mut heap: BinaryHeap<PairFrequency> = BinaryHeap::with_capacity(counts.len());
    heap.extend(counts.iter().map(|(&pair, &freq)| (freq, pair)));
    heap
}

fn pop_most_frequent_pair(
    heap: &mut BinaryHeap<PairFrequency>,
    counts: &HashMap<Pair, usize>,
) -> Option<(Pair, usize)> {
    while let Some((freq, pair)) = heap.pop() {
        if counts.get(&pair).copied() == Some(freq) {
            return Some((pair, freq));
        }
    }
    None
}

fn get_chunk_pair_counts(chunk: &[u16]) -> HashMap<Pair, usize> {
    let mut counts: HashMap<Pair, usize> = HashMap::new();
    for pair in chunk.windows(2) {
        *counts.entry((pair[0], pair[1])).or_insert(0) += 1;
    }
    counts
}

fn chunk_contains_pair(chunk: &[u16], target: Pair) -> bool {
    chunk.windows(2).any(|pair| (pair[0], pair[1]) == target)
}

fn update_pair_counts(
    counts: &mut HashMap<Pair, usize>,
    heap: &mut BinaryHeap<PairFrequency>,
    chunk: &[u16],
    merged_chunk: &[u16],
    weight: usize,
) {
    let previous_pairs = get_chunk_pair_counts(chunk);
    let next_pairs = get_chunk_pair_counts(merged_chunk);
    let mut touched_pairs: HashSet<Pair> =
        HashSet::with_capacity(previous_pairs.len() + next_pairs.len());

    for (pair, count) in previous_pairs {
        let total = count * weight;
        let entry = counts
            .get_mut(&pair)
            .expect("pair frequency must exist before merge update");
        *entry -= total;
        if *entry == 0 {
            counts.remove(&pair);
        }
        touched_pairs.insert(pair);
    }

    for (pair, count) in next_pairs {
        *counts.entry(pair).or_insert(0) += count * weight;
        touched_pairs.insert(pair);
    }

    for pair in touched_pairs {
        if let Some(&freq) = counts.get(&pair) {
            heap.push((freq, pair));
        }
    }
}

fn bpe_merge(ids: &[u16], pairs: Pair, idx: u16) -> Vec<u16> {
    let mut new_ids: Vec<u16> = Vec::with_capacity(ids.len());
    let mut i = 0;
    while i < ids.len() {
        if i < ids.len() - 1 && ids[i] == pairs.0 && ids[i + 1] == pairs.1 {
            new_ids.push(idx);
            i += 2;
        } else {
            new_ids.push(ids[i]);
            i += 1;
        }
    }
    return new_ids;
}

fn save_merges_record(merges_record: &[Merge]) -> std::io::Result<()> {
    let mut serialized = String::from("[\n");
    for (index, ((p0, p1), id)) in merges_record.iter().enumerate() {
        let separator = if index + 1 == merges_record.len() {
            ""
        } else {
            ","
        };
        serialized.push_str(&format!(
            "  {{\"pair\":[{p0},{p1}],\"id\":{id}}}{separator}\n"
        ));
    }
    serialized.push(']');
    fs::write(MERGES_RECORD_FILE, serialized)
}

fn build_vocab_list(merges: &[Merge]) -> VocabList {
    let mut vocab_list: VocabList = HashMap::new();
    for idx in 0..256u16 {
        vocab_list.insert(idx, vec![idx as u8]);
    }
    for &((p0, p1), id) in merges {
        let left = &vocab_list[&p0];
        let right = &vocab_list[&p1];
        let mut merged = Vec::with_capacity(left.len() + right.len());
        merged.extend_from_slice(left);
        merged.extend_from_slice(right);
        vocab_list.insert(id, merged);
    }
    vocab_list
}

fn save_vocab_list_record(vocab_list: &VocabList) -> std::io::Result<()> {
    let mut ids: Vec<u16> = vocab_list.keys().copied().collect();
    ids.sort_unstable();

    let mut serialized = String::from("[\n");
    for (index, id) in ids.iter().enumerate() {
        let separator = if index + 1 == ids.len() { "" } else { "," };
        let bytes = vocab_list
            .get(id)
            .unwrap()
            .iter()
            .map(|byte| byte.to_string())
            .collect::<Vec<String>>()
            .join(",");
        serialized.push_str(&format!(
            "  {{\"id\":{id},\"bytes\":[{bytes}]}}{separator}\n"
        ));
    }
    serialized.push(']');
    fs::write(VOCAB_LIST_FILE, serialized)
}

#[pyfunction(name = "build_info")]
pub(crate) fn get_build_info() -> String {
    format!(
        "ground_llm build: capped_encode_train_max_50000:{}:{}",
        env!("CARGO_PKG_VERSION"),
        file!()
    )
}

#[pyfunction]
pub(crate) fn encode_train(chunks: Vec<String>) -> PyResult<Vec<Merge>> {
    let mut data = collapse_chunk_counts(
        chunks
            .into_iter()
            .map(|chunk| (chunk.into_bytes().into_iter().map(u16::from).collect(), 1)),
    );
    let mut pair_counts = get_stat(&data);
    let mut pair_heap = build_pair_heap(&pair_counts);
    let mut merges_record: Vec<Merge> = Vec::new();
    let mut next_idx: u16 = 256;
    loop {
        if merges_record.len() >= MAX_MERGES {
            println!("reached max_merges={MAX_MERGES}");
            break;
        }

        let Some((max_pair, freq)) = pop_most_frequent_pair(&mut pair_heap, &pair_counts) else {
            break;
        };
        if freq < 3 {
            break;
        }

        println!("next_idx={next_idx}");

        let mut next_data: HashMap<Vec<u16>, usize> = HashMap::with_capacity(data.len());
        for (chunk, weight) in data {
            if chunk_contains_pair(&chunk, max_pair) {
                let merged_chunk = bpe_merge(&chunk, max_pair, next_idx);
                update_pair_counts(
                    &mut pair_counts,
                    &mut pair_heap,
                    &chunk,
                    &merged_chunk,
                    weight,
                );
                *next_data.entry(merged_chunk).or_insert(0) += weight;
            } else {
                *next_data.entry(chunk).or_insert(0) += weight;
            }
        }
        data = next_data.into_iter().collect();
        merges_record.push((max_pair, next_idx));
        let Some(updated_idx) = next_idx.checked_add(1) else {
            break;
        };
        next_idx = updated_idx;
    }

    save_merges_record(&merges_record).map_err(|err| {
        PyIOError::new_err(format!("failed to write {MERGES_RECORD_FILE}: {err}"))
    })?;
    save_vocab_list_record(&build_vocab_list(&merges_record))
        .map_err(|err| PyIOError::new_err(format!("failed to write {VOCAB_LIST_FILE}: {err}")))?;

    Ok(merges_record)
}
#[pyfunction]
pub(crate) fn encode(s: Vec<String>, merges: Vec<Merge>) -> Vec<u16> {
    let mut s_utf8: Vec<Vec<u16>> = s
        .into_iter()
        .map(|chunk| chunk.into_bytes().into_iter().map(u16::from).collect())
        .collect();

    for ((p0, p1), id) in merges {
        for chunk in s_utf8.iter_mut() {
            *chunk = bpe_merge(chunk, (p0, p1), id);
        }
    }

    s_utf8.into_iter().flatten().collect()
}

#[pyfunction]
pub(crate) fn decode_string(ids: Vec<u16>, vocab_list: VocabList) -> String {
    let decode_byte: Vec<u8> = ids
        .iter()
        .flat_map(|id| vocab_list.get(id).unwrap())
        .copied()
        .collect();
    let text = String::from_utf8(decode_byte);
    return text.unwrap();
}

// #[pyfunction]
// pub(crate) fn save_vocab_list(merges: Vec<Merge>) -> PyResult<()> {
//     let vocab_list = build_vocab_list(&merges);
//     save_vocab_list_record(&vocab_list)
//         .map_err(|err| PyIOError::new_err(format!("failed to write {VOCAB_LIST_FILE}: {err}")))?;
//     Ok(())
// }

#[cfg(test)]
mod tests {
    use super::{
        bpe_merge, build_pair_heap, chunk_contains_pair, collapse_chunk_counts, get_stat,
        pop_most_frequent_pair, update_pair_counts,
    };
    use std::collections::HashMap;

    #[test]
    fn collapse_chunk_counts_merges_duplicates() {
        let collapsed = collapse_chunk_counts(vec![
            (vec![1, 2, 3], 1),
            (vec![4, 5], 2),
            (vec![1, 2, 3], 3),
        ]);

        let aggregated: std::collections::HashMap<Vec<u16>, usize> =
            collapsed.into_iter().collect();
        assert_eq!(aggregated.get(&vec![1, 2, 3]), Some(&4));
        assert_eq!(aggregated.get(&vec![4, 5]), Some(&2));
    }

    #[test]
    fn get_stat_respects_chunk_weights() {
        let stats = get_stat(&[(vec![1, 2, 1], 3), (vec![1, 2], 2)]);

        assert_eq!(stats.get(&(1, 2)), Some(&5));
        assert_eq!(stats.get(&(2, 1)), Some(&3));
    }

    #[test]
    fn pop_most_frequent_pair_uses_heap_ordering() {
        let counts = get_stat(&[(vec![1, 2, 1], 3), (vec![7, 8, 7, 8], 1), (vec![1, 2], 2)]);
        let mut heap = build_pair_heap(&counts);
        let top_pair = pop_most_frequent_pair(&mut heap, &counts);

        assert_eq!(top_pair, Some(((1, 2), 5)));
    }

    #[test]
    fn incremental_pair_updates_match_full_recount() {
        let data = vec![(vec![1, 2, 1, 2], 2), (vec![2, 3], 1)];
        let max_pair = (1, 2);
        let next_idx = 256;
        let mut pair_counts = get_stat(&data);
        let mut pair_heap = build_pair_heap(&pair_counts);
        let mut next_data: HashMap<Vec<u16>, usize> = HashMap::new();

        for (chunk, weight) in data {
            if chunk_contains_pair(&chunk, max_pair) {
                let merged_chunk = bpe_merge(&chunk, max_pair, next_idx);
                update_pair_counts(
                    &mut pair_counts,
                    &mut pair_heap,
                    &chunk,
                    &merged_chunk,
                    weight,
                );
                *next_data.entry(merged_chunk).or_insert(0) += weight;
            } else {
                *next_data.entry(chunk).or_insert(0) += weight;
            }
        }

        let next_data: Vec<(Vec<u16>, usize)> = next_data.into_iter().collect();
        assert_eq!(pair_counts, get_stat(&next_data));
        assert_eq!(
            pop_most_frequent_pair(&mut pair_heap, &pair_counts),
            Some(((256, 256), 2))
        );
    }
}
