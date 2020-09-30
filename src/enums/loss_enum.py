from enum import Enum


class LossEnum(Enum):
    LOSS = "loss"
    RECONSTRUCTION = "reconstruction"
    RECONSTRUCTION_OCCLUSION = "reconstruction_occlusion"
    RECONSTRUCTION_NON_OCCLUSION = "reconstruction_non_occlusion"
    KLD = "kld"
    VQ = "vq"
    STYLE = "style"
    PERCEPTUAL = "perceptual"
    TOTAL_VARIATION = "total_variation"
