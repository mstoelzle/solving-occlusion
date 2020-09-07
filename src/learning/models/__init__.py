from .base_model import *
from .vae.base_vae import *
from .vae.vanilla_vae import *

models = {"VanillaVAE": VanillaVAE}


def pick_model(**kwargs) -> torch.nn:
    return models[kwargs["name"]](**kwargs)
