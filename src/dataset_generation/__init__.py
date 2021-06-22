from .anymal_rosbag_dataset_generator_v2 import AnymalRosbagDatasetGenerator
from .base_dataset_generator import BaseDatasetGenerator
from .ga_slam_msgpack_dataset_generator import GASlamMsgpackDatasetGenerator
from .ga_slam_pocolog_dataset_generator import GASlamPocologDatasetGenerator
from .synthetic_dataset_generator import SyntheticDatasetGenerator
from src.enums import *

dataset_generators = {DatasetGenerationTypeEnum.SYNTHETIC: SyntheticDatasetGenerator,
                      DatasetGenerationTypeEnum.ANYMAL_ROSBAG: AnymalRosbagDatasetGenerator,
                      DatasetGenerationTypeEnum.ROCK_GA_SLAM_MSGPACK: GASlamMsgpackDatasetGenerator,
                      DatasetGenerationTypeEnum.ROCK_GA_SLAM_POCOLOG: GASlamPocologDatasetGenerator}


def pick_dataset_generator(**kwargs):
    return dataset_generators[DatasetGenerationTypeEnum(kwargs["type"])](**kwargs)
