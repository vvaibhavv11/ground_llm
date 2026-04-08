use std::collections::HashMap;
use std::fs;

use pyo3::exceptions::PyIOError;
use pyo3::prelude::*;
use rayon::prelude::*;

const MERGES_RECORD_FILE: &str = "merges_record.json";
const VOCAB_LIST_FILE: &str = "vocab_list.json";
const MAX_MERGES: usize = 50_000;

type Merge = ((u16, u16), u16);
type VocabList = HashMap<u16, Vec<u8>>;

fn get_stat(raw_utf8: &[Vec<u16>]) -> HashMap<(u16, u16), usize> {
    raw_utf8
        .par_iter()
        .map(|chunk| {
            let mut counts: HashMap<(u16, u16), usize> = HashMap::new();
            for pairs in chunk.windows(2) {
                *counts.entry((pairs[0], pairs[1])).or_insert(0) += 1;
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

fn bpe_merge(ids: &[u16], pairs: (u16, u16), idx: u16) -> Vec<u16> {
    let mut new_ids: Vec<u16> = Vec::new();
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
        vocab_list.insert(
            id,
            [vocab_list[&p0].clone(), vocab_list[&p1].clone()].concat(),
        );
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
    let mut data: Vec<Vec<u16>> = chunks
        .into_iter()
        .map(|chunk| chunk.into_bytes().into_iter().map(u16::from).collect())
        .collect();
    let mut counts: HashMap<Vec<u16>, usize> = HashMap::new();
    for word in data {
        *counts.entry(word).or_insert(0) += 1;
    }
    let mut merges_record: Vec<Merge> = Vec::new();
    let mut next_idx: u16 = 256;
    loop {
        if merges_record.len() >= MAX_MERGES {
            println!("reached max_merges={MAX_MERGES}");
            break;
        }

        let state = get_stat(&data);
        let Some((&max_pair, &freq)) = state.iter().max_by_key(|(_, rank)| *rank) else {
            break;
        };
        if freq < 3 {
            break;
        }

        println!("next_idx={next_idx}");

        data = data
            .into_par_iter()
            .map(|chunk| bpe_merge(&chunk, max_pair, next_idx))
            .collect();
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
