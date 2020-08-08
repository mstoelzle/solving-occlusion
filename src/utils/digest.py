import multiprocessing as mp
from typing import Optional, Callable, Any, Dict

import torch
from torch.utils.tensorboard import SummaryWriter

from .log import get_logger

import torch.distributions


class TensorboardDigest:

    known_methods = ("sum_last_gradient", "fisher_information")

    def __init__(self, **config):
        self.config = self.validate_config(config)

        self.logger = get_logger("tensorboard_digest")
        self.logdir = config["logdir"]
        self.writer = SummaryWriter(self.logdir)
        self.step = 0
        self.data = []

    def __enter__(self):
        self.logger.info(f"Digest reset. Will save results to {self.logdir}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __del__(self):
        self.logger.info("Closing Summary writer")
        self.writer.close()

    def write(self, tag, args, step, fn=None):
        if fn is None:
            value = args[0]
        else:
            value = fn(*args)
        self.writer.add_scalar(tag, value, step)

    def digest(self, model: torch.nn.Module):
        for method in self.config.get("methods", []):
            if method == "sum_last_gradient":
                pass
            elif method == "fisher_information":

                data = torch.stack([point[1] for point in self.data], dim=0).sum(dim=0) / len(self.data)

                self.write("fim", [model, data], self.step, compute_gradient_perturbation_sensitivity)

        self.step += 1

    def validate_config(self, config: Dict) -> Dict:
        # this is called during __init__ so that we notice problems with the config as early as possible
        for method in config.get("methods", []):
            if method not in self.known_methods:
                raise NotImplementedError(f"unknown digest method {method} must be one of {self.known_methods}")
        return config

    def cache_data(self, batch_idx, data: torch.tensor):
        if batch_idx % 100 == 0:
            self.data.append((batch_idx, data))


def compute_gradient_perturbation_sensitivity(model: torch.nn.Module, model_input: torch.tensor) -> int:

    with torch.enable_grad():
        fim = {name: 0 for name, param in model.named_parameters() if param.requires_grad}

        pred_target0 = model(model_input)
        outdx = torch.distributions.Categorical(logits=pred_target0).sample().unsqueeze(1).detach()
        samples = pred_target0.gather(1, outdx)
        idx, batch_size = 0, model_input.size(0)
        while idx < batch_size:
            model.zero_grad()
            torch.autograd.backward(samples[idx], retain_graph=True)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    fim[name] += torch.sum(param.grad * param.grad).detach()
            idx += 1
    fim = sum([fim[name] for name in fim])
    return fim
