from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import ShadowConfig
from shadow.dataset import TextDataset
from shadow.model import ShadowModel


def main() -> None:
    config = ShadowConfig()
    train_path = Path("data/train.txt")
    if not train_path.exists():
        train_path.write_text(
            "SHADOW AI is an experimental hybrid neural-symbolic system.\n",
            encoding="utf-8",
        )

    text = train_path.read_text(encoding="utf-8")
    dataset = TextDataset(text, config.sequence_length)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model = ShadowModel(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        layers=config.layers,
        heads=config.heads,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        for x, y in loader:
            logits = model(x)
            loss = loss_fn(logits.reshape(-1, config.vocab_size), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        mean_loss = total_loss / max(1, len(loader))
        print(f"epoch={epoch + 1} loss={mean_loss:.4f}")

    checkpoint = Path("checkpoints/shadow.pt")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint)


if __name__ == "__main__":
    main()
