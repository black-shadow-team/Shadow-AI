import torch
import torch.nn as nn


class ShadowModel(nn.Module):
    """Small neural-only Transformer baseline for controlled SHADOW experiments."""

    def __init__(
        self,
        vocab_size: int = 256,
        hidden_size: int = 256,
        layers: int = 6,
        heads: int = 8,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=heads,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(tokens)
        hidden = self.encoder(hidden)
        return self.output(hidden)
