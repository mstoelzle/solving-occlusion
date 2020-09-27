import pathlib
import torch
from torchvision.datasets.folder import is_image_file
from typing import *

from .base_dataset import BaseDataset
from src.enums import *


class TrasysPlanetaryDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.samples = []
        self.make_dataset()

    def make_dataset(self):
        for sample_dir in sorted(self.dataset_path.iterdir()):
            if sample_dir.is_dir():
                sample_dict = {}
                for filepath in sorted(sample_dir.iterdir()):
                    if filepath.is_file() and is_image_file(str(filepath)):
                        filename = filepath.stem
                        if filename == "height":
                            sample_dict[ChannelEnum.ELEVATION_MAP] = filepath
                        elif filename == "occlusion":
                            sample_dict[ChannelEnum.BINARY_OCCLUSION_MAP] = filepath
                        else:
                            continue

                self.samples.append(sample_dict)

    def __getitem__(self, idx: int) -> Dict[Union[str, ChannelEnum], torch.Tensor]:
        sample_dict = self.samples[idx]

        data = self.prepare_item(sample_dict)

        return data

    def __len__(self):
        return len(self.samples)
