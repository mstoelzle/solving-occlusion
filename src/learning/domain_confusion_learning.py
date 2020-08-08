from typing import List, Tuple

import torch

from .base_learning import BaseLearning
from src.learning.loss.domain_confusion_loss import DomainConfusionLoss
from src.learning.tasks import Task
from ..utils.log import get_logger

logger = get_logger("domain_confusion_learning")


class DomainConfusionLearning(BaseLearning):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.activations: List = []

    def reset(self):
        super().reset()
        self.activations = []

    def train(self, task: Task):
        self.task = task
        self.task.loss = DomainConfusionLoss(self.task.logdir, **self.task.config["loss"])

        self.set_model(task.model_to_train)

        if max(self.task.loss.regularization_weights[1:]) > 0.0:
            def record_activations(layer, layer_input: Tuple, layer_output: torch.Tensor):
                self.activations.append(layer_input[0].detach())

            num_linear_layers = 0
            for layer in self.model.children():
                if isinstance(layer, torch.nn.Linear):
                    layer.register_forward_hook(record_activations)
                    num_linear_layers += 1
            assert num_linear_layers > 0

        return self.train_epoches()

    def train_epoch(self, epoch) -> None:
        self.model.train()

        with self.task.loss.new_epoch(epoch, "train"):
            labeled_dataloader = self.task.labeled_dataloader.dataloaders['train']
            unlabeled_dataloader = self.task.unlabeled_dataloader.dataloaders['train']
            for batch_idx, (labeled_data, unlabeled_data) in enumerate(zip(labeled_dataloader, unlabeled_dataloader)):
                self.optimizer.zero_grad()

                labeled_data_features = labeled_data[0]
                labeled_data_targets = labeled_data[1]
                unlabeled_data_features = unlabeled_data[0]

                labeled_data_features = labeled_data_features.to(self.device)
                labeled_data_targets = labeled_data_targets.to(self.device)
                unlabeled_data_features = unlabeled_data_features.to(self.device)

                loss_tensors = {"labeled_data_targets": labeled_data_targets}
                if max(self.task.loss.regularization_weights[1:]) > 0.0:
                    labeled_data_pred_targets = self.model(labeled_data_features)
                    loss_tensors['labeled_data_activations'] = self.activations.copy()
                    self.activations.clear()

                    unlabeled_data_pred_targets = self.model(unlabeled_data_features)
                    loss_tensors['unlabeled_data_activations'] = self.activations.copy()
                    self.activations.clear()
                else:
                    labeled_data_pred_targets = self.model(labeled_data_features)
                    unlabeled_data_pred_targets = self.model(unlabeled_data_features)
                loss_tensors["labeled_data_pred_targets"] = labeled_data_pred_targets
                loss_tensors["unlabeled_data_pred_targets"] = unlabeled_data_pred_targets
                loss = self.task.loss(loss_tensors)

                loss.backward()
                self.optimizer.step()

    def validate_epoch(self, epoch: int):
        self.model.eval()
        with self.task.loss.new_epoch(epoch, "val"), torch.no_grad():
            labeled_dataloader = self.task.labeled_dataloader.dataloaders['val']
            unlabeled_dataloader = self.task.unlabeled_dataloader.dataloaders['val']
            for batch_idx, (labeled_data, unlabeled_data) in enumerate(zip(labeled_dataloader, unlabeled_dataloader)):
                labeled_data_features = labeled_data[0]
                labeled_data_targets = labeled_data[1]
                unlabeled_data_features = unlabeled_data[0]

                labeled_data_features = labeled_data_features.to(self.device)
                labeled_data_targets = labeled_data_targets.to(self.device)
                unlabeled_data_features = unlabeled_data_features.to(self.device)

                loss_tensors = {"labeled_data_targets": labeled_data_targets}
                if max(self.task.loss.regularization_weights[1:]) > 0.0:
                    labeled_data_pred_targets = self.model(labeled_data_features)
                    loss_tensors['labeled_data_activations'] = self.activations.copy()
                    self.activations.clear()

                    unlabeled_data_pred_targets = self.model(unlabeled_data_features)
                    loss_tensors['unlabeled_data_activations'] = self.activations.copy()
                    self.activations.clear()
                else:
                    labeled_data_pred_targets = self.model(labeled_data_features)
                    unlabeled_data_pred_targets = self.model(unlabeled_data_features)
                loss_tensors["labeled_data_pred_targets"] = labeled_data_pred_targets
                loss_tensors["unlabeled_data_pred_targets"] = unlabeled_data_pred_targets
                self.task.loss(loss_tensors)

        self.controller.add_state(epoch, self.task.loss.get_epoch_loss(), self.model.state_dict())
