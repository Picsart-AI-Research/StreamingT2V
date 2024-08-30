from pathlib import Path
from torch.utils.data import Dataset

from dataloader.single_image_dataset import SingleImageDataset


class SingleImageDatasetFactory():

    def __init__(self, file: Path):
        self.data_path = file

    def get_dataset(self, max_samples: int = None) -> Dataset:
        return SingleImageDataset(file=self.data_path)
