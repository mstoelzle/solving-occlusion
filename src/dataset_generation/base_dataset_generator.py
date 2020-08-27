from abc import ABC, abstractmethod
import numpy as np
import random
import torch


class BaseDatasetGenerator(ABC):
    def __init__(self, **kwargs):
        self.config = kwargs

        # set seed
        seeds = self.config.get("seeds", [101])
        assert len(seeds) > 0
        self.seed = seeds[0]
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    @abstractmethod
    def run(self):
        pass

    def save_to_dataset(self):
        pass
