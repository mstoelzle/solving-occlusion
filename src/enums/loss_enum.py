from enum import Enum


class LossEnum(Enum):
    LOSS = "loss"
    RECONSTRUCTION = "reconstruction"
    KLD = "kld"
    RECONSTRUCTION_OCCLUSION = "reconstruction_occlusion"
