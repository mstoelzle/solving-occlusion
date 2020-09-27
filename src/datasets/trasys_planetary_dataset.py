import pathlib
import torch
from torchvision.datasets.folder import default_loader as torchvision_default_loader, is_image_file
from typing import *

from .base_dataset import BaseDataset
from src.enums import *


class TrasysPlanetaryDataset(BaseDataset):
    def __init__(self, dataset_path: pathlib.Path, **kwargs):
        super().__init__(**kwargs)

        self.dataset_path = dataset_path
        self.samples = []

        for sample_dir in sorted(dataset_path.iterdir()):
            if sample_dir.is_dir():
                sample_dict = {}
                for filepath in sorted(sample_dir.iterdir()):
                    if filepath.is_file() and is_image_file(str(filepath)):
                        filename = filepath.stem
                        if filename == "height":
                            sample_dict[ChannelEnum.ELEVATION_MAP] = filepath
                        elif filename == "occlusion":
                            sample_dict[ChannelEnum.OCCLUDED_ELEVATION_MAP] = filepath
                        else:
                            continue

                self.samples.append(sample_dict)

        self.loader = torchvision_default_loader

    def __getitem__(self, idx: int) -> Dict[Union[str, ChannelEnum], torch.Tensor]:
        sample_dict = self.samples[idx]

        for channel, path in sample_dict.items():
            img = self.loader(path)

    def __len__(self):
        return len(self.samples)
