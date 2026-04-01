use std::collections::HashMap;

use pyo3::prelude::*;

fn get_stat(raw_utf8: &[u16]) -> HashMap<(u16, u16), usize> {
    let mut counts: HashMap<(u16, u16), usize> = HashMap::new();
    for pairs in raw_utf8.windows(2) {
        *counts.entry((pairs[0], pairs[1])).or_insert(0) += 1;
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

#[pyfunction]
fn encode_train(s: String) {
    let s_utf8 = s.into_bytes();
    let vocab_size: u16 = 276;
    let num_merges = vocab_size - 256;
    let mut copy_s_utf8: Vec<u16> = s_utf8.into_iter().map(u16::from).collect();
    let mut merges_record: Vec<((u16, u16), u16)> = Vec::new();
    for i in 0..num_merges {
        let state = get_stat(&copy_s_utf8);
        let Some((&max_pair, _)) = state.iter().max_by_key(|(_, rank)| *rank) else {
            break;
        };
        let idx = 256 + i;
        copy_s_utf8 = bpe_merge(&copy_s_utf8, max_pair, idx);
        merges_record.push((max_pair, idx));
    }
}
#[pyfunction]
fn encode(s: String, merges: Vec<((u16, u16), u16)>) -> Vec<u16> {
    let s_utf8 = s.into_bytes();
    let mut copy_s_utf8: Vec<u16> = s_utf8.into_iter().map(u16::from).collect();
    for ((p0, p1), id) in merges {
        copy_s_utf8 = bpe_merge(&copy_s_utf8, (p0, p1), id);
    }
    return copy_s_utf8;
}

#[pyfunction]
fn decode_string(ids: Vec<u16>, merges: Vec<((u16, u16), u16)>) -> String {
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
