from enum import Enum


class ModelUncertaintyMethod(Enum):
    MONTE_CARLO_DROPOUT = "monte_carlo_dropout"
    MONTE_CARLO_VAE = "monte_carlo_vae"
