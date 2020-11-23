from enum import Enum


class LossEnum(Enum):
    LOSS = "loss"
    L1_REC_ALL = "l1_rec_all"
    L1_REC_OCC = "l1_rec_occ"
    L1_REC_NOCC = "l1_rec_nocc"
    L1_COMP_ALL = "l1_comp_all"
    MSE_REC_ALL = "mse_rec_all"
    MSE_REC_OCC = "mse_rec_occ"
    MSE_REC_NOCC = "mse_rec_nocc"
    MSE_COMP_ALL = "mse_comp_all"
    PSNR_REC_ALL = "psnr_rec_all"
    PSNR_REC_OCC = "psnr_rec_occ"
    PSNR_REC_NOCC = "psnr_rec_nocc"
    PSNR_COMP_ALL = "psnr_comp_all"
    KLD = "kld"
    VQ = "vq"
    STYLE = "style"
    PERCEPTUAL = "perceptual"
    TV = "total_variation"
