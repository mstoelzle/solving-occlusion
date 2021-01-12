from .anybotics_rosbag_dataset_generator import AnyboticsRosbagDatasetGenerator
from .base_dataset_generator import BaseDatasetGenerator
from .ga_slam_msgpack_dataset_generator import GASlamMsgpackDatasetGenerator
from .synthetic_dataset_generator import SyntheticDatasetGenerator
from src.enums import *

dataset_generators = {DatasetGenerationTypeEnum.SYNTHETIC: SyntheticDatasetGenerator,
                      DatasetGenerationTypeEnum.ANYMAL_ROSBAG: AnyboticsRosbagDatasetGenerator,
                      DatasetGenerationTypeEnum.ROCK_GA_SLAM_MSGPACK: GASlamMsgpackDatasetGenerator}


def pick_dataset_generator(**kwargs):
    return dataset_generators[DatasetGenerationTypeEnum(kwargs["type"])](**kwargs)
