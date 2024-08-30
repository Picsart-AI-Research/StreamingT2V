import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import (EVAL_DATALOADERS)
from dataloader.dataset_factory import SingleImageDatasetFactory


class VideoDataModule(pl.LightningDataModule):

    def __init__(self,
                 workers: int,
                 predict_dataset_factory: SingleImageDatasetFactory = None,
                 ) -> None:
        super().__init__()
        self.num_workers = workers

        self.video_data_module = {}
        # TODO read size from loaded unet via unet.sample_sizes
        self.predict_dataset_factory = predict_dataset_factory

    def setup(self, stage: str) -> None:
        if stage == "predict":
            self.video_data_module["predict"] = self.predict_dataset_factory.get_dataset(
            )

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return torch.utils.data.DataLoader(self.video_data_module["predict"],
                                           batch_size=1,
                                           pin_memory=True,
                                           num_workers=self.num_workers,
                                           collate_fn=None,
                                           shuffle=False,
                                           drop_last=False)
