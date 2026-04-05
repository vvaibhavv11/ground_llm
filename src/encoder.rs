use std::collections::HashMap;
use std::fs;

use pyo3::exceptions::PyIOError;
use pyo3::prelude::*;

const MERGES_RECORD_FILE: &str = "merges_record.json";

fn get_stat(raw_utf8: &Vec<Vec<u16>>) -> HashMap<(u16, u16), usize> {
    let mut counts: HashMap<(u16, u16), usize> = HashMap::new();
    for chunk in raw_utf8 {
        for pairs in chunk.windows(2) {
            *counts.entry((pairs[0], pairs[1])).or_insert(0) += 1;
        }
    }
    return counts;
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

fn save_merges_record(merges_record: &[((u16, u16), u16)]) -> std::io::Result<()> {
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

#[pyfunction(name = "build_info")]
pub(crate) fn get_build_info() -> String {
    format!(
        "ground_llm build: unbounded_encode_train:{}:{}",
        env!("CARGO_PKG_VERSION"),
        file!()
    )
}

#[pyfunction]
pub(crate) fn encode_train(chunks: Vec<String>) -> PyResult<Vec<((u16, u16), u16)>> {
    let mut data: Vec<Vec<u16>> = chunks
        .into_iter()
        .map(|chunk| chunk.into_bytes().into_iter().map(u16::from).collect())
        .collect();
    let mut merges_record: Vec<((u16, u16), u16)> = Vec::new();
    let mut next_idx: u16 = 256;
    loop {
        let state = get_stat(&data);
        let Some((&max_pair, &freq)) = state.iter().max_by_key(|(_, rank)| *rank) else {
            break;
        };
        if freq < 3 {
            break;
        }
        for chunk in data.iter_mut() {
            *chunk = bpe_merge(chunk, max_pair, next_idx);
        }
        merges_record.push((max_pair, next_idx));
        let Some(updated_idx) = next_idx.checked_add(1) else {
            break;
        };
        next_idx = updated_idx;
    }

    save_merges_record(&merges_record).map_err(|err| {
        PyIOError::new_err(format!("failed to write {MERGES_RECORD_FILE}: {err}"))
    })?;

    Ok(merges_record)
}
#[pyfunction]
pub(crate) fn encode(s: Vec<String>, merges: Vec<((u16, u16), u16)>) -> Vec<u16> {
    let s_utf8: Vec<Vec<u16>> = s
        .into_iter()
        .map(|chunk| chunk.into_bytes().into_iter().map(u16::from).collect())
        .collect();
    let mut final_utf8: Vec<u16> = Vec::new();
    for ((p0, p1), id) in merges {
        for chunk in &s_utf8 {
            final_utf8.extend(bpe_merge(&chunk, (p0, p1), id));
        }
    }
    return final_utf8;
}

#[pyfunction]
pub(crate) fn decode_string(ids: Vec<u16>, merges: Vec<((u16, u16), u16)>) -> String {
    let mut vocab_list: HashMap<u16, Vec<u8>> = HashMap::new();
    for idx in 0..256 {
        vocab_list.insert(idx, vec![idx as u8]);
    }
    for ((p0, p1), id) in merges {
        vocab_list.insert(
            id,
            [vocab_list[&p0].clone(), vocab_list[&p1].clone()].concat(),
        );
    }
    let decode_byte: Vec<u8> = ids
        .iter()
        .flat_map(|id| vocab_list.get(id).unwrap())
        .copied()
        .collect();
    let text = String::from_utf8(decode_byte);
    return text.unwrap();
}
