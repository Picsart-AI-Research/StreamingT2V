from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import EVAL_DATALOADERS

from t2v_enhanced.model.datasets.video_dataset import Annotations
import json


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.model_id = datasets["reconstruction_dataset"].model_id

    def __getitem__(self, idx):
        sample = {ds: self.datasets[ds].__getitem__(
            idx) for ds in self.datasets}
        return sample

    def __len__(self):
        return min(len(self.datasets[d]) for d in self.datasets)


class CustomPromptsDataset(torch.utils.data.Dataset):

    def __init__(self, prompt_cfg: Dict[str, str]):
        super().__init__()

        if prompt_cfg["type"] == "prompt":
            self.prompts = [prompt_cfg["content"]]
        elif prompt_cfg["type"] == "file":
            file = Path(prompt_cfg["content"])
            if file.suffix == ".npy":
                self.prompts = np.load(file.as_posix())
            elif file.suffix == ".txt":
                with open(prompt_cfg["content"]) as f:
                    lines = [line.rstrip() for line in f]
                self.prompts = lines
            elif file.suffix == ".json":
                with open(prompt_cfg["content"],"r") as file:
                    metadata = json.load(file)
                if "videos_root" in prompt_cfg:
                    videos_root = Path(prompt_cfg["videos_root"])
                    video_path = [str(videos_root / sample["page_dir"] /
                                  f"{sample['videoid']}.mp4") for sample in metadata]
                else:
                    video_path = [str(sample["page_dir"] /
                                  f"{sample['videoid']}.mp4") for sample in metadata]
                self.prompts = [sample["prompt"] for sample in metadata]
                self.video_path = video_path




        transformed_prompts = []
        for prompt in self.prompts:
            transformed_prompts.append(
                Annotations.clean_prompt(prompt))
        self.prompts = transformed_prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index):
        output = {"prompt": self.prompts[index]}
        if hasattr(self,"video_path"):
            output["video"] = self.video_path[index]
        return output


class PromptReader(pl.LightningDataModule):
    def __init__(self, prompt_cfg: Dict[str, str]):
        super().__init__()
        self.predict_dataset = CustomPromptsDataset(prompt_cfg)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=1, pin_memory=False, shuffle=False, drop_last=False)
