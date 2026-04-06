use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

mod encoder;

/// A Python module implemented in Rust.
#[pymodule]
fn ground_llm(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(encoder::get_build_info, m)?)?;
    m.add_function(wrap_pyfunction!(encoder::encode_train, m)?)?;
    m.add_function(wrap_pyfunction!(encoder::encode, m)?)?;
    m.add_function(wrap_pyfunction!(encoder::decode_string, m)?)?;
    // m.add_function(wrap_pyfunction!(encoder::save_vocab_list, m)?)?;
    Ok(())
}
