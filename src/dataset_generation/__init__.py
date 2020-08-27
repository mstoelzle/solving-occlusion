from .base_dataset_generator import BaseDatasetGenerator
from .synthetic_dataset_generator import SyntheticDatasetGenerator

dataset_generators = {"synthetic": SyntheticDatasetGenerator}


def pick_dataset_generator(**kwargs):
    return dataset_generators[kwargs["type"]](**kwargs)
