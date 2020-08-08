import pathlib
from typing import Dict

import torch
import torch.nn.functional as F

from src.learning.loss.base_loss import BaseLoss
from src.utils.log import get_logger

logger = get_logger("supervised_learning_loss")


class SupervisedLearningLoss(BaseLoss):
    def __init__(self, logdir: pathlib.Path, **kwargs):
        super().__init__(logger, logdir, **kwargs)

        loss_function_name = self.config.get("loss_function", {}).get("name", "cross-entropy")
        if loss_function_name == "hinge-loss":
            self.loss_functions = {"train": {"name": "hinge-loss", "callable": hinge_loss_fct},
                                   "val": {"name": "hinge-loss", "callable": hinge_loss_fct},
                                   "test": {"name": "hinge-loss", "callable": hinge_loss_fct}}
        elif loss_function_name == "cross-entropy":
            self.loss_functions = {"train": {"name": "cross-entropy", "callable": F.cross_entropy},
                                   "val": {"name": "cross-entropy", "callable": F.cross_entropy},
                                   "test": {"name": "cross-entropy", "callable": F.cross_entropy}}
        else:
            raise NotImplementedError(f"The loss function with name {loss_function_name} is not implemented.")

        self.add_loss_fct_wrappers()  # add threshold and scaling wrappers to loss function

    def __call__(self, tensors: Dict) -> torch.Tensor:
        batch_result = {}

        if self.accuracy_bools[self.purpose]:
            batch_result.update(self.compute_accuracy(tensors['labeled_data_pred_targets'],
                                                      tensors['labeled_data_targets']))

        class_weights = self.get_class_weights(tensors['labeled_data_pred_targets'])

        loss_function_callable = self.loss_functions[self.purpose]["callable"]
        loss_fct_output = loss_function_callable(tensors['labeled_data_pred_targets'],
                                                 tensors['labeled_data_targets'],
                                                 reduction="none", weight=class_weights)

        loss = self.loss_aggregator(loss_fct_output)

        batch_result.update({"loss": loss, "batch_size": tensors['labeled_data_targets'].size(0)})

        self.process_batch_result(batch_result)

        return loss


def hinge_loss_fct(labeled_data_targets, labeled_data_pred_targets,
                   reduction="mean", weight=None) -> torch.Tensor:
    modified_labeled_data_pred_targets = labeled_data_pred_targets.clone()
    modified_labeled_data_pred_targets[labeled_data_pred_targets == 0] = -1
    hinge_loss = torch.mean(
        torch.clamp(1 - torch.t(labeled_data_targets) * modified_labeled_data_pred_targets, min=0))
    return hinge_loss
