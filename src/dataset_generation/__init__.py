from .anybotics_rosbag_dataset_generator import AnyboticsRosbagDatasetGenerator
from .base_dataset_generator import BaseDatasetGenerator
from .synthetic_dataset_generator import SyntheticDatasetGenerator
from src.enums import *

dataset_generators = {DatasetGenerationTypeEnum.SYNTHETIC: SyntheticDatasetGenerator,
                      DatasetGenerationTypeEnum.ANYMAL_ROSBAG: AnyboticsRosbagDatasetGenerator}


def pick_dataset_generator(**kwargs):
    return dataset_generators[DatasetGenerationTypeEnum(kwargs["type"])](**kwargs)
