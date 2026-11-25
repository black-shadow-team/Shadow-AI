from pathlib import Path

import torch

from config import ShadowConfig
from shadow.model import ShadowModel


def main() -> None:
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

    validation = Path("data/validation.txt")
    if not validation.exists():
        validation.write_text("SHADOW measures performance honestly.", encoding="utf-8")

    data = torch.tensor(list(validation.read_text(encoding="utf-8").encode("utf-8")))
    if len(data) <= config.sequence_length:
        print("validation_bytes_too_short=0")
        print("perplexity=not_available")
        return

    x = data[:-1].unsqueeze(0)
    y = data[1:].unsqueeze(0)
    x = x[:, :config.sequence_length]
    y = y[:, :config.sequence_length]
    with torch.no_grad():
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, config.vocab_size),
            y.reshape(-1),
        )
    print(f"validation_loss={loss.item():.4f}")
    print(f"perplexity={torch.exp(loss).item():.4f}")


if __name__ == "__main__":
    main()
