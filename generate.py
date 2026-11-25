from pathlib import Path

import torch

from config import ShadowConfig
from shadow.model import ShadowModel
from shadow.tokenizer import ShadowTokenizer


def generate(prompt: str, length: int = 100) -> str:
    config = ShadowConfig()
    checkpoint = Path("checkpoints/shadow.pt")
    if not checkpoint.exists():
        raise FileNotFoundError("Run python train.py first")

    model = ShadowModel(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        layers=config.layers,
        heads=config.heads,
    )
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()

    tokenizer = ShadowTokenizer()
    x = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    with torch.no_grad():
        for _ in range(length):
            logits = model(x[:, -config.sequence_length:])
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            x = torch.cat([x, next_token[:, None]], dim=1)

    return tokenizer.decode(x[0].tolist())


if __name__ == "__main__":
    print(generate("Hello", length=100))
