import torch
import numpy as np


class SingleImageDataset():

    def __init__(self, file: np.ndarray):
        self.images = [file]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return {"image": self.images[index], "sample_id": torch.tensor(index, dtype=torch.int64)}
