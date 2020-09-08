from enum import Enum


class ChannelEnum(Enum):
    ELEVATION_MAP = "elevation_map"
    RECONSTRUCTED_ELEVATION_MAP = "reconstructed_elevation_map"
    OCCLUDED_ELEVATION_MAP = "occluded_elevation_map"
    BINARY_OCCLUSION_MAP = "binary_occlusion_map"