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
                transformed_input = self.random_noise(transform, transformed_input)
            elif transform["type"] == "random_scale":
                transformed_input = self.random_scale(transform, transformed_input)
            elif transform["type"] == "random_offset":
                transformed_input = self.random_offset(transform, transformed_input)
            else:
                raise NotImplementedError

        return transformed_input

    def random_noise(self, transform: dict, input: np.array) -> np.array:
        stdev = transform["stdev"]

        deterministic = transform.get("deterministic", True)
        if deterministic:
            noise = self.rng.normal(loc=0, scale=stdev, size=tuple(input.size()))
        else:
            noise = np.random.normal(loc=0, scale=stdev, size=tuple(input.size()))

        noise = input.new_tensor(noise, dtype=input.dtype)

        transformed_input = input + noise

        return transformed_input

    def random_scale(self, transform: dict, input: np.array) -> np.array:
        min, max = transform["min"], transform["max"]

        deterministic = transform.get("deterministic", True)
        if deterministic:
            scale = self.rng.uniform(low=min, high=max)
        else:
            scale = np.random.uniform(low=min, high=max)

        transformed_input = scale * input

        return transformed_input

    def random_offset(self, transform: dict, input: np.array) -> np.array:
        min, max = transform["min"], transform["max"]

        deterministic = transform.get("deterministic", True)
        if deterministic:
            offset = self.rng.uniform(low=min, high=max)
        else:
            offset = np.random.uniform(low=min, high=max)

        transformed_input = offset + input

        return transformed_input
