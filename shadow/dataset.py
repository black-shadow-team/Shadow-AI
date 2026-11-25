from torch.utils.data import Dataset
import torch


class TextDataset(Dataset):
    def __init__(self, text: str, seq_len: int = 128):
        self.data = torch.tensor(list(text.encode("utf-8")), dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len - 1)

    def __getitem__(self, index: int):
        x = self.data[index:index + self.seq_len]
        y = self.data[index + 1:index + self.seq_len + 1]
        return x, y
