from .base_model import *
from .baseline.open_cv_baseline import OpenCVBaseline
from .vae.base_vae import *
from .vae.vanilla_vae import *

models = {"VanillaVAE": VanillaVAE,
          "NavierStokes": OpenCVBaseline,
          "Telea": OpenCVBaseline}


def pick_model(**kwargs) -> torch.nn:
    return models[kwargs["name"]](**kwargs)
