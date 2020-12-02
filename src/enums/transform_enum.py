from enum import Enum


class TransformEnum(Enum):
    WHITE_NOISE = "white_noise"
    RANDOM_VERTICAL_SCALE = "random_vertical_scale"
    RANDOM_VERTICAL_OFFSET = "random_vertical_offset"
    RANDOM_OCCLUSION = "random_occlusion"
