from .base_model import *
from .baseline.lsq_plane_fit_baseline import LsqPlaneFitBaseline
from .baseline.interpolation_baseline import InterpolationBaseline
from .baseline.open_cv_baseline import OpenCVBaseline
from .partialconv.partialconv_unet import PartialConvUNet
from .vae.base_vae import BaseVAE
from .vae.vanilla_vae import VanillaVAE
from .vae.vq_vae import VQVAE
from .unet.unet import UNet
from .vae.unet_vae import UNetVAE

MODELS = {
    "cubic": InterpolationBaseline,
    "linear": InterpolationBaseline,
    "lsq_plane_fit": LsqPlaneFitBaseline,
    "NavierStokes": OpenCVBaseline,
    "nearest": InterpolationBaseline,
    "PartialConvUNet": PartialConvUNet,
    "PatchMatch": OpenCVBaseline,
    "Telea": OpenCVBaseline,
    "UNet": UNet,
    "UNetVAE": UNetVAE,
    "VanillaVAE": VanillaVAE,
    "VQVAE": VQVAE
}


def pick_model(**kwargs) -> torch.nn:
    return MODELS[kwargs["name"]](**kwargs)
