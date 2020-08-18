import torch

from .base_model import *
from .base_vae import *
from .vanilla_vae import *

models = {"VanillaVAE": VanillaVAE}


def pick_model(**kwargs) -> torch.nn:
    return models[kwargs["name"]](**kwargs)
