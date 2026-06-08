from dataclasses import dataclass
@dataclass
class Config:
    batch_size: int = 4096
    lr: float = 0.001
    epochs: int = 100
    seq_len: int = 100
    embedding_dim: int = 64
    max_category_len: int = 1
    json_path: str = 'data/reviews_Electronics_5.json'
    data_path: str = 'data/meta_Electronics.json'
    weight_decay: float = 1e-4