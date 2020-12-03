from enum import Enum


class TransformEnum(Enum):
    WHITE_NOISE = "white_noise"
    RANGE_ADJUSTED_WHITE_NOISE = "range_adjusted_white_noise"
    GAUSSIAN_FILTERED_WHITE_NOISE = "gaussian_filtered_white_noise"
    RANDOM_VERTICAL_SCALE = "random_vertical_scale"
    RANDOM_VERTICAL_OFFSET = "random_vertical_offset"
    RANDOM_OCCLUSION = "random_occlusion"
    RANDOM_OCCLUSION_DILATION = "random_occlusion_dilation"
