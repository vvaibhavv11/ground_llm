from typing import Dict, List, Literal, Optional, Tuple, TypedDict

class TrainHistory(TypedDict):
    train_loss: List[float]
    val_loss: List[Tuple[int, float]]

class Model:
    def __init__(
        self,
        vocab_size: int = 8000,
        dimensions: int = 512,
        n_heads: int = 8,
        n_layers: int = 4,
        mlp_hidden: int = 1380,
        max_seq_len: int = 256,
    ) -> None: ...
    def train(
        self,
        tokens: List[int],
        steps: int = 1000,
        batch_size: int = 8,
        seq_len: Optional[int] = None,
        lr: float = 3e-4,
        min_lr: Optional[float] = None,
        warmup_steps: int = 100,
        weight_decay: float = 0.1,
        grad_clip: float = 1.0,
        val_fraction: float = 0.1,
        eval_every: int = 100,
        eval_batches: int = 8,
        log_every: int = 10,
        checkpoint_path: Optional[str] = None,
        seed: int = 0,
        device: Literal["cpu", "gpu", "auto"] = "cpu",
    ) -> TrainHistory: ...
    def loss(self, tokens: List[int]) -> float: ...
    def generate(
        self,
        prompt: List[int],
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[int]: ...
    def save(self, path: str) -> None: ...
    @staticmethod
    def load(path: str) -> Model: ...
    def num_params(self) -> int: ...
    @property
    def config(self) -> Dict[str, int]: ...
