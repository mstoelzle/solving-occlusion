from .base_model import *
from .baseline.open_cv_baseline import OpenCVBaseline
from .vae.base_vae import BaseVAE
from .vae.vanilla_vae import VanillaVAE
from .vae.vq_vae import VQVAE

models = {"VanillaVAE": VanillaVAE,
          "NavierStokes": OpenCVBaseline,
          "Telea": OpenCVBaseline,
          "PatchMatch": OpenCVBaseline,
          "VQVAE": VQVAE}


def pick_model(**kwargs) -> torch.nn:
    return models[kwargs["name"]](**kwargs)
