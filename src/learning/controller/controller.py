import pathlib
import csv
import copy
from typing import Dict

import torch

from src.utils.log import get_logger

logger = get_logger("controller")


class Controller:
    def __init__(self, **kwargs):
        """
        The Controller manages when training should be terminated. Its states contain model dicts and validation losses
        for each epoch. It wil stop the iteration if the training has converged - this occurs for example when the
        validation loss has not improved for the last 'max_num_better_results'. It will also stop the iteration if
        the maximum number of epochs has been reached. Finally it can return the best state after the iteration to
        reset the model to the state in which it achieved the best validation loss.
        We include a boolean option 'epoch_stop' that bases the learning stopping only on the number of epochs and
        returns the model of the last trained epoch instead of the model with the best validation loss.
        :param kwargs:
        """
        # If "get()" does not find keyword, the value is None
        self.max_num_epochs = kwargs.get("max_num_epochs")
        self.max_num_better_results = kwargs.get("max_num_better_results")
        self.epoch_stop: bool = kwargs.get("epoch_stop", False)
        assert (self.epoch_stop is True and self.max_num_epochs is not None and self.max_num_better_results is None) \
            or (self.max_num_better_results is not None and self.epoch_stop is False) \
            or (self.max_num_epochs is not None and self.epoch_stop is False)
        self.states = []
        self.logger = logger

    def __iter__(self):
        self.epoch = 0
        return self

    def reset(self):
        self.states = []

    def __next__(self):
        convergence_reached: bool = False

        if len(self.states) != 0 and self.max_num_better_results is not None:
            """
            Take minimal loss. Compare to all previous self.max_n_better states.
            If this minimal loss is found in the last self.max_n_better states
            then continue training. Learning is still making progress. Else stop. 
            """
            min_loss: float = min([state["loss"] for state in self.states])
            min_max_n_better: float = min([state["loss"] for state in self.states[-self.max_num_better_results:]])
            convergence_reached = min_loss < min_max_n_better

        max_epoch_reached: bool = False
        if self.max_num_epochs is not None:
            max_epoch_reached = self.epoch >= self.max_num_epochs

        if max_epoch_reached:
            self.logger.info(
                f"Training finished because the maximum number of epochs ({self.max_num_epochs}) was reached")
        elif convergence_reached:
            self.logger.info("Training stopped because convergence was reached.")

        if max_epoch_reached or convergence_reached:
            raise StopIteration

        self.epoch += 1
        return self.epoch - 1

    def add_state(self, epoch: int, loss: torch.Tensor, model_dict: Dict) -> None:
        loss_value = float(loss)
        self.states.append({"epoch": epoch, "loss": loss_value, "model_dict": copy.deepcopy(model_dict)})
        self.discard_model_dicts()

    def discard_model_dicts(self) -> None:
        """
        Discards all model dicts which are worse than the current best state to improve memory footprint.
        :return: None
        """
        if self.epoch_stop is True:
            sorted_states = sorted(self.states, key=lambda k: k['epoch'], reverse=True)
        else:
            sorted_states = sorted(self.states, key=lambda k: k['loss'])

        for state in sorted_states[1:]:
            state.update({"model_dict": {}})

    def get_best_state(self) -> Dict:
        if self.epoch_stop is True:
            sorted_states = sorted(self.states, key=lambda k: k['epoch'], reverse=True)
        else:
            sorted_states = sorted(self.states, key=lambda k: k['loss'])
        return sorted_states[0]
