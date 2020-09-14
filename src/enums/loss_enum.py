from enum import Enum


class LossEnum(Enum):
    LOSS = "loss"
    RECONSTRUCTION = "reconstruction"
    RECONSTRUCTION_OCCLUSION = "reconstruction_occlusion"
    KLD = "kld"
    VQ = "vq"
