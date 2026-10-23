use pyo3::prelude::*;

mod matrix;
mod mlp;
mod train;

/// A Python module implemented in Rust.
#[pymodule]
fn ground_llm(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
