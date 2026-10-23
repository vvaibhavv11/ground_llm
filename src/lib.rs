use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

mod matrix;
mod mlp;
mod optim;
mod param;
mod train;

use train::{Config, TrainConfig, Transformer};

/// Python handle to a Transformer language model.
#[pyclass(name = "Model")]
struct Model {
    inner: Transformer,
}

#[pymethods]
impl Model {
    #[new]
    #[pyo3(signature = (vocab_size=8000, dimensions=512, n_heads=8, n_layers=4, mlp_hidden=1380, max_seq_len=256))]
    fn new(
        vocab_size: usize,
        dimensions: usize,
        n_heads: usize,
        n_layers: usize,
        mlp_hidden: usize,
        max_seq_len: usize,
    ) -> PyResult<Self> {
        let config = Config {
            dimensions,
            n_heads,
            vocab_size,
            head_dim: if n_heads > 0 { dimensions / n_heads } else { 0 },
            mlp_hidden,
            n_layers,
            max_seq_len,
        };
        config.validate().map_err(PyValueError::new_err)?;
        Ok(Self {
            inner: Transformer::new(config),
        })
    }

    /// Train on a flat list of token ids. Logs progress every `log_every` steps.
    /// Ctrl-C stops training and raises KeyboardInterrupt; the model keeps the
    /// weights learned so far.
    #[pyo3(signature = (
        tokens,
        steps=1000,
        batch_size=8,
        seq_len=None,
        lr=3e-4,
        min_lr=None,
        warmup_steps=100,
        weight_decay=0.1,
        grad_clip=1.0,
        val_fraction=0.1,
        eval_every=100,
        eval_batches=8,
        log_every=10,
        checkpoint_path=None,
        seed=0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn train<'py>(
        &mut self,
        py: Python<'py>,
        tokens: Vec<u16>,
        steps: usize,
        batch_size: usize,
        seq_len: Option<usize>,
        lr: f32,
        min_lr: Option<f32>,
        warmup_steps: usize,
        weight_decay: f32,
        grad_clip: f32,
        val_fraction: f32,
        eval_every: usize,
        eval_batches: usize,
        log_every: usize,
        checkpoint_path: Option<String>,
        seed: u64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let cfg = TrainConfig {
            steps,
            batch_size,
            seq_len: seq_len.unwrap_or(self.inner.config.max_seq_len),
            lr,
            min_lr: min_lr.unwrap_or(lr / 10.0),
            warmup_steps,
            weight_decay,
            grad_clip,
            val_fraction,
            eval_every,
            eval_batches,
            checkpoint_path,
            seed,
        };

        let model = &mut self.inner;
        let mut interrupted: Option<PyErr> = None;

        let result = py.detach(|| {
            train::train(model, &tokens, &cfg, |stats| {
                Python::attach(|py| {
                    let should_log = stats.val_loss.is_some()
                        || (log_every > 0 && stats.step % log_every == 0)
                        || stats.step == 1;
                    if should_log {
                        let mut line = format!(
                            "step {:>6} | loss {:.4} | lr {:.2e} | grad norm {:.3} | {:.0} tok/s",
                            stats.step, stats.loss, stats.lr, stats.grad_norm, stats.tokens_per_sec
                        );
                        if let Some(val) = stats.val_loss {
                            line.push_str(&format!(" | val loss {:.4}", val));
                        }
                        let _ = py_print(py, &line);
                    }

                    match py.check_signals() {
                        Ok(()) => true,
                        Err(err) => {
                            interrupted = Some(err);
                            false
                        }
                    }
                })
            })
        });

        if let Some(err) = interrupted {
            return Err(err);
        }
        let history = result.map_err(PyValueError::new_err)?;

        let out = PyDict::new(py);
        out.set_item("train_loss", history.train_loss)?;
        out.set_item("val_loss", history.val_loss)?;
        Ok(out)
    }

    /// Mean next-token cross-entropy over `tokens` (uses windows of max_seq_len).
    fn loss(&mut self, py: Python<'_>, tokens: Vec<u16>) -> PyResult<f32> {
        let max_len = self.inner.config.max_seq_len;
        if tokens.len() < 2 {
            return Err(PyValueError::new_err("need at least 2 tokens"));
        }
        check_tokens(&tokens, self.inner.config.vocab_size)?;

        let model = &mut self.inner;
        Ok(py.detach(|| {
            let mut total = 0.0;
            let mut count = 0;
            for chunk_start in (0..tokens.len() - 1).step_by(max_len) {
                let end = (chunk_start + max_len).min(tokens.len() - 1);
                let n = end - chunk_start;
                total += model.loss(&tokens[chunk_start..end], &tokens[chunk_start + 1..end + 1])
                    * n as f32;
                count += n;
            }
            total / count as f32
        }))
    }

    /// Sample continuation tokens. temperature=0 means greedy decoding.
    #[pyo3(signature = (prompt, max_new_tokens=100, temperature=1.0, top_k=None, seed=None))]
    fn generate(
        &mut self,
        py: Python<'_>,
        prompt: Vec<u16>,
        max_new_tokens: usize,
        temperature: f32,
        top_k: Option<usize>,
        seed: Option<u64>,
    ) -> PyResult<Vec<u16>> {
        use rand::prelude::*;

        if prompt.is_empty() {
            return Err(PyValueError::new_err(
                "prompt must contain at least one token",
            ));
        }
        check_tokens(&prompt, self.inner.config.vocab_size)?;

        let model = &mut self.inner;
        Ok(py.detach(|| {
            let mut rng = match seed {
                Some(s) => StdRng::seed_from_u64(s),
                None => StdRng::from_rng(&mut rand::rng()),
            };
            train::generate(
                model,
                &prompt,
                max_new_tokens,
                temperature,
                top_k.unwrap_or(0),
                &mut rng,
            )
        }))
    }

    fn save(&self, path: &str) -> PyResult<()> {
        self.inner
            .save(path)
            .map_err(|e| PyIOError::new_err(e.to_string()))
    }

    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let inner = Transformer::load(path).map_err(|e| PyIOError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn num_params(&self) -> usize {
        self.inner.num_params()
    }

    #[getter]
    fn config<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let c = &self.inner.config;
        let out = PyDict::new(py);
        out.set_item("vocab_size", c.vocab_size)?;
        out.set_item("dimensions", c.dimensions)?;
        out.set_item("n_heads", c.n_heads)?;
        out.set_item("n_layers", c.n_layers)?;
        out.set_item("mlp_hidden", c.mlp_hidden)?;
        out.set_item("max_seq_len", c.max_seq_len)?;
        Ok(out)
    }

    fn __repr__(&self) -> String {
        let c = &self.inner.config;
        format!(
            "Model(vocab_size={}, dimensions={}, n_heads={}, n_layers={}, mlp_hidden={}, max_seq_len={}, params={})",
            c.vocab_size,
            c.dimensions,
            c.n_heads,
            c.n_layers,
            c.mlp_hidden,
            c.max_seq_len,
            self.inner.num_params()
        )
    }
}

fn check_tokens(tokens: &[u16], vocab_size: usize) -> PyResult<()> {
    match tokens.iter().find(|&&t| t as usize >= vocab_size) {
        Some(bad) => Err(PyValueError::new_err(format!(
            "token id {} out of range for vocab_size {}",
            bad, vocab_size
        ))),
        None => Ok(()),
    }
}

/// Print through Python so output shows up in notebooks too.
fn py_print(py: Python<'_>, line: &str) -> PyResult<()> {
    py.import("builtins")?.getattr("print")?.call1((line,))?;
    Ok(())
}

/// A Python module implemented in Rust.
#[pymodule]
fn ground_llm(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Model>()?;
    Ok(())
}
