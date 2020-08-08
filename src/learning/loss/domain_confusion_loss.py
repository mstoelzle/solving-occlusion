import pathlib
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from path_learning.loss.base_loss import BaseLoss, calc_domain_distance
from path_learning.utils.log import get_logger

logger = get_logger("domain_confusion_loss")


class DomainConfusionLoss(BaseLoss):
    def __init__(self, logdir: pathlib.Path, **kwargs):
        super().__init__(logger, logdir, **kwargs)

        self.accuracy_bools = {"train": False, "val": False, "test": True}

        self.loss_functions = {"train": {"name": "domain-confusion", "callable": self.domain_confusion_loss_fct},
                               "val": {"name": "domain-confusion", "callable": self.domain_confusion_loss_fct},
                               "test": {"name": "cross-entropy", "callable": F.cross_entropy}}
        self.add_loss_fct_wrappers()  # add threshold and scaling wrappers to loss function

        loss_function_config = self.config.get("loss_function", {})
        self.domain_distance_metric = loss_function_config.get('domain_distance_metric', "fid")
        # weather to normalize the activations by the own magnitude
        self.normalize_activations = loss_function_config.get('normalize_activations', True)

        # the classification has always a weight of 1
        # the first element of the regularization weight list belongs to the final layer (e.g. labels)
        # the second element of the regularization weight list belongs to the second-to-last activation layer
        if "regularization_weights" in loss_function_config:
            self.regularization_weights = [0.0, 0.0]
            for idx, regularization_weight in enumerate(loss_function_config['regularization_weights']):
                assert 0.0 <= regularization_weight, "The regularization weight must be larger than 0.0"
                self.regularization_weights[idx] = regularization_weight
        else:
            self.regularization_weights = [0.2, 0.2]

    def __call__(self, tensors: Dict) -> torch.Tensor:
        batch_result = {}

        if self.accuracy_bools[self.purpose]:
            batch_result.update(self.compute_accuracy(tensors['labeled_data_pred_targets'],
                                                      tensors['labeled_data_targets']))

        loss_function_name = self.loss_functions[self.purpose]['name']
        loss_function_callable = self.loss_functions[self.purpose]['callable']

        class_weights = self.get_class_weights(tensors['labeled_data_pred_targets'])

        if loss_function_name == "cross-entropy":
            loss_fct_output = loss_function_callable(tensors['labeled_data_pred_targets'],
                                                     tensors['labeled_data_targets'],
                                                     reduction="none", weight=class_weights)
        elif loss_function_name == "domain-confusion":
            if 'labeled_data_activations' not in tensors:
                tensors["labeled_data_activations"] = []
            if 'unlabeled_data_activations' not in tensors:
                tensors["unlabeled_data_activations"] = []
            loss_fct_output = loss_function_callable(labeled_data_targets=tensors['labeled_data_targets'],
                                                     labeled_data_pred_targets=tensors['labeled_data_pred_targets'],
                                                     unlabeled_data_pred_targets=tensors['unlabeled_data_pred_targets'],
                                                     reduction="none", weight=class_weights,
                                                     labeled_data_activations=tensors['labeled_data_activations'],
                                                     unlabeled_data_activations=tensors['unlabeled_data_activations'])
        else:
            raise NotImplementedError(f"The loss function with name {loss_function_name} has not been implemented for "
                                      f"Domain confusion loss.")

        if type(loss_fct_output) == tuple:
            loss = self.loss_aggregator(loss_fct_output[0])
            batch_result["loss_blame"] = self.loss_aggregator(loss_fct_output[1], dim=0)
        else:
            loss = self.loss_aggregator(loss_fct_output)

        batch_result.update({"loss": loss, "batch_size": tensors['labeled_data_targets'].size(0)})

        self.process_batch_result(batch_result)

        return loss

    def domain_confusion_loss_fct(self,
                                  labeled_data_targets, labeled_data_pred_targets, unlabeled_data_pred_targets,
                                  reduction='mean', weight=None, labeled_data_activations=[],
                                  unlabeled_data_activations=[]) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.isnan(labeled_data_pred_targets).any() or torch.isnan(unlabeled_data_pred_targets).any():
            raise ValueError("The predictions of the labeled or unlabeled dataloader contains NaNs (Not a number)")

        if labeled_data_pred_targets.size(0) != unlabeled_data_pred_targets.size(0):
            logger.warning(f"Number of labeled and unlabeled batch samples do not match in domain_confusion_loss_fct "
                           f"({labeled_data_pred_targets.size(0)} to {unlabeled_data_pred_targets.size(0)})")
            min_batch_size = min(labeled_data_pred_targets.size(0), unlabeled_data_pred_targets.size(0))

            labeled_data_targets = labeled_data_targets[0:min_batch_size]
            labeled_data_pred_targets = labeled_data_pred_targets[0:min_batch_size, :]
            unlabeled_data_pred_targets = unlabeled_data_pred_targets[0:min_batch_size, :]

            if len(labeled_data_activations) > 0:
                for activation_idx, labeled_data_activation in enumerate(labeled_data_activations):
                    labeled_data_activations[activation_idx] = labeled_data_activation[0:min_batch_size, :]

            if len(unlabeled_data_activations) > 0:
                for activation_idx, unlabeled_data_activation in enumerate(unlabeled_data_activations):
                    unlabeled_data_activations[activation_idx] = unlabeled_data_activation[0:min_batch_size, :]

        if labeled_data_pred_targets.size(1) != unlabeled_data_pred_targets.size(1):
            raise Exception("The tensor dimension (number of classes) of the source and target domain "
                            "features must match.")

        classification_loss = F.cross_entropy(labeled_data_pred_targets, labeled_data_targets,
                                              reduction=reduction, weight=weight)

        # we can only construct the covariance matrix if we have more than one labeled sample
        if labeled_data_pred_targets.size(0) > 1:
            (_, unlabeled_data_argmax_targets) = unlabeled_data_pred_targets.max(dim=1)

            if self.regularization_weights[0] > 0.0:
                label_loss = calc_domain_distance(data_domain_1=labeled_data_pred_targets,
                                                  data_domain_2=unlabeled_data_pred_targets,
                                                  domain_distance_metric=self.domain_distance_metric,
                                                  normalize_activations=self.normalize_activations,
                                                  num_classes=labeled_data_pred_targets.size(1),
                                                  targets_domain_1=labeled_data_targets,
                                                  targets_domain_2=unlabeled_data_argmax_targets,
                                                  pred_targets_domain_1=labeled_data_pred_targets,
                                                  pred_targets_domain_2=unlabeled_data_pred_targets)
            else:
                label_loss = classification_loss.new_tensor(0.0)

            if self.regularization_weights[1] > 0.0 and len(labeled_data_activations) > 0:
                # for now we just take the last layer to calculate the activation loss
                labeled_data_activation = labeled_data_activations[-1]
                unlabeled_data_activation = unlabeled_data_activations[-1]
                second_to_last_layer_loss = calc_domain_distance(data_domain_1=labeled_data_activation,
                                                                 data_domain_2=unlabeled_data_activation,
                                                                 domain_distance_metric=self.domain_distance_metric,
                                                                 normalize_activations=self.normalize_activations,
                                                                 num_classes=labeled_data_pred_targets.size(1),
                                                                 targets_domain_1=labeled_data_targets,
                                                                 targets_domain_2=unlabeled_data_argmax_targets,
                                                                 pred_targets_domain_1=labeled_data_pred_targets,
                                                                 pred_targets_domain_2=unlabeled_data_pred_targets)
            else:
                second_to_last_layer_loss = classification_loss.new_tensor(0.0)

            domain_confusion_loss = self.regularization_weights[0] * label_loss \
                                    + self.regularization_weights[1] * second_to_last_layer_loss

            # bring it to same tensor dim as classification_loss
            domain_confusion_loss = domain_confusion_loss * \
                                    classification_loss.new_ones(size=classification_loss.size())

            if reduction == 'mean':
                domain_confusion_loss = domain_confusion_loss / labeled_data_pred_targets.size(1)
            elif reduction == 'none':
                pass
            else:
                raise NotImplementedError(f"The following reduction method is not implemented for the domain confusion "
                                          f"loss function: {reduction}")
        else:
            domain_confusion_loss = classification_loss.new_zeros(size=classification_loss.size())

        loss = classification_loss + domain_confusion_loss

        loss_blame = torch.stack((classification_loss, domain_confusion_loss), dim=1)

        return loss, loss_blame
