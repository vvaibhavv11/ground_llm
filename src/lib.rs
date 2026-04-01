use pyo3::prelude::*;

mod encoder;

/// A Python module implemented in Rust.
#[pymodule]
fn tokenizer(py: Python, m: &PyModule) -> PyResult<()> {}
