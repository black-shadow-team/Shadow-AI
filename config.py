from dataclasses import dataclass

@dataclass
class ShadowConfig:
    vocab_size: int = 256
    hidden_size: int = 256
    layers: int = 6
    heads: int = 8
    sequence_length: int = 128
    batch_size: int = 8
    learning_rate: float = 3e-4
    epochs: int = 5
