from enum import Enum


class DatasetGenerationTypeEnum(Enum):
    SYNTHETIC = "synthetic"
    ANYMAL_ROSBAG = "anymal_rosbag"
    ROCK_GA_SLAM_MSGPACK = "rock_ga_slam_msgpack"
    ROCK_GA_SLAM_POCOLOG = "rock_ga_slam_pocolog"
