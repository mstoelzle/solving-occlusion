import numpy as np
import torch


class Transformer:
    def __init__(self, purpose: str, transforms: list):
        self.purpose = purpose

        self.transforms = transforms
        self.rng = np.random.RandomState(seed=1)

    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        transformed_input = input

        for transform in self.transforms:
            if transform["type"] == "random_noise":
                stdev = transform["stdev"]

                deterministic = transform.get("deterministic", True)
                if deterministic:
                    noise = self.rng.normal(loc=0, scale=stdev, size=tuple(input.size()))
                else:
                    noise = np.random.normal(loc=0, scale=stdev, size=tuple(input.size()))

                noise = input.new_tensor(noise, dtype=input.dtype)

                transformed_input = transformed_input + noise

        return transformed_input
