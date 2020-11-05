from enum import Enum


class ChannelEnum(Enum):
    PARAMS = "params"
    GROUND_TRUTH_ELEVATION_MAP = "ground_truth_elevation_map"
    RECONSTRUCTED_ELEVATION_MAP = "reconstructed_elevation_map"
    OCCLUDED_ELEVATION_MAP = "occluded_elevation_map"
    BINARY_OCCLUSION_MAP = "binary_occlusion_map"
    COMPOSED_ELEVATION_MAP = "composed_elevation_map"
