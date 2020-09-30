from .base_model import *
from .baseline.open_cv_baseline import OpenCVBaseline
from .partialconv.partialconv_unet import PartialConvUNet
from .vae.base_vae import BaseVAE
from .vae.vanilla_vae import VanillaVAE
from .vae.vq_vae import VQVAE
from .unet.unet_model import UNet

MODELS = {"VanillaVAE": VanillaVAE,
          "NavierStokes": OpenCVBaseline,
          "Telea": OpenCVBaseline,
          "PartialConvUNet": PartialConvUNet,
          "PatchMatch": OpenCVBaseline,
          "VQVAE": VQVAE,
          "UNet": UNet}


def pick_model(**kwargs) -> torch.nn:
    return MODELS[kwargs["name"]](**kwargs)
