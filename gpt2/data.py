from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, dataset_path: str, train_config: dataclass):
        super().__init__()
        self.train_config = train_config
        with open(dataset_path, "r") as f:
            self.text = f.read()

    def __len__(self):
        pass

    def __getitem__(self, index):
        buf = self.text[
            index : min(
                index + self.train_config.batch_size * self.train_config.seq_len + 1,
                len(self.text),
            )
        ]
        x = buf[0:-1]
        y = buf[1:]
        return x, y
