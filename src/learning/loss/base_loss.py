import csv
import logging
import pathlib
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.nn import functional as F

from .domain_distance_metrics.cmmd import conditional_maximum_mean_discrepancy
from .domain_distance_metrics.coral import coral
from .domain_distance_metrics.fid import frechet_inception_distance
from .domain_distance_metrics.lmmd import local_maximum_mean_discrepancy
from .domain_distance_metrics.mmd import maximum_mean_discrepancy
from src.utils.log import get_logger

logger = get_logger("base_loss")


class BaseLoss(ABC):
    def __init__(self, logger: logging.Logger, logdir: pathlib.Path, **kwargs):
        self.config = kwargs
        self.logger = logger
        self.logdir = logdir

        self.loss_functions: Optional[Dict[str, Dict]] = {}
        self.report_frequency: int = self.config["report_frequency"]
        self.loss_aggregator = pick_loss_aggregator(self.config["loss_aggregator"])
        self.batch_results = []  # must be reset at the beginning of each epoch

        self.accuracy_bools = {"train": True, "val": True, "test": True}

        self.epoch_losses = {"train": {}, "val": {}, "test": {}}

        self.purpose: Optional[str] = None
        self.epoch: Optional[int] = None

        self.class_weights: List = self.config.get("loss_function", {}).get("class_weights", None)

    def new_epoch(self, epoch: int, purpose: str):
        self.purpose = purpose
        self.epoch = epoch
        return self

    def __enter__(self):
        self.batch_results = []

    def __exit__(self, exc_type, exc_val, exc_tb):
        epoch_result = self.compute_epoch_result()
        self.epoch_losses[self.purpose].update({str(self.epoch): epoch_result["loss"]})

        if self.purpose != "test":
            if self.accuracy_bools[self.purpose]:
                self.logger.info(f"Epoch {self.epoch} - {self.purpose} average loss - {epoch_result['loss']:.4f} "
                                 f"- average accuracy {epoch_result['accuracy']:.4f}")
            else:
                self.logger.info(f"Epoch {self.epoch} - {self.purpose} average loss - {epoch_result['loss']:.4f}")
        else:
            if self.accuracy_bools[self.purpose]:
                self.logger.info(f"Testing Loss: {epoch_result['loss']:.4f} "
                                 f"- average accuracy {epoch_result['accuracy']:.4f}")
            else:
                self.logger.info(f"Testing Loss: {epoch_result['loss']:.4f}")

        self.write_logfile(self.epoch, epoch_result)

    def write_logfile(self, epoch, epoch_result: Dict):
        logfile_path = self.logdir / f"{self.purpose}_losses.csv"
        if logfile_path.exists():
            write_mode = "a"
        else:
            write_mode = "w"

        write_dict = {"epoch": epoch, "loss": float(epoch_result["loss"]),
                      "accuracy": float(epoch_result.get("accuracy", None))}

        if "loss_blame" in epoch_result:
            for loss_idx, loss in enumerate(epoch_result["loss_blame"]):
                fieldname = f"loss_blame_{str(loss_idx).zfill(2)}"
                write_dict[fieldname] = float(loss)

        if "accuracy_per_class" in epoch_result:
            for class_idx, class_accuracy in enumerate(epoch_result["accuracy_per_class"]):
                fieldname = f"class_accuracy_{str(class_idx).zfill(2)}"
                write_dict[fieldname] = float(class_accuracy)

        with open(str(logfile_path), write_mode, newline='') as fp:
            fieldnames = list(write_dict.keys())
            writer = csv.DictWriter(fp, fieldnames=fieldnames)

            if write_mode == "w":
                writer.writeheader()

            writer.writerow(write_dict)

    def get_epoch_loss(self):
        return self.epoch_losses[self.purpose][str(self.epoch)]

    def compute_epoch_result(self) -> Dict:
        epoch_result = {}

        num_samples: int = 0
        total_loss: float = 0.0
        total_loss_blame = None

        total_correct: int = 0
        total_correct_per_class = None
        num_samples_per_class = None

        for batch_idx, batch_result in enumerate(self.batch_results):
            total_loss += batch_result["loss"] * batch_result["batch_size"]
            num_samples += batch_result["batch_size"]

            if "loss_blame" in batch_result:
                if total_loss_blame is None:
                    total_loss_blame = batch_result["loss_blame"] * batch_result["batch_size"]
                else:
                    total_loss_blame += batch_result["loss_blame"] * batch_result["batch_size"]

            if self.accuracy_bools[self.purpose]:
                total_correct += batch_result["correct"]

                if "correct_per_class" in batch_result and "num_samples_per_class" in batch_result:
                    if total_correct_per_class is None or num_samples_per_class is None:
                        total_correct_per_class = batch_result["correct_per_class"]
                        num_samples_per_class = batch_result["num_samples_per_class"]
                    else:
                        total_correct_per_class += batch_result["correct_per_class"]
                        num_samples_per_class += batch_result["num_samples_per_class"]

        epoch_result["loss"] = total_loss / float(num_samples)

        if total_loss_blame is not None:
            epoch_result["loss_blame"] = total_loss_blame / float(num_samples)

        epoch_result["accuracy"] = float(total_correct) / float(num_samples)

        if total_correct_per_class is not None and num_samples_per_class is not None:
            epoch_result["accuracy_per_class"] = (total_correct_per_class / num_samples_per_class.float())

        return epoch_result

    @abstractmethod
    def __call__(self, tensors: Dict) -> torch.Tensor:
        pass

    def compute_accuracy(self, pred_targets: torch.Tensor, targets: torch.Tensor) -> Dict:
        batch_result = {}
        batch_size = pred_targets.size(0)
        num_classes = pred_targets.size(1)

        # Compute accuracy
        if pred_targets.size(1) == 1:
            max_pred = pred_targets.clone()
            max_pred[pred_targets > 0] = 1
            max_pred[pred_targets <= 0] = 0
        else:
            # get the index of the max log-probability
            max_pred = pred_targets.argmax(dim=1, keepdim=True)

        correct_batch = max_pred.eq(targets.view_as(max_pred))
        correct = correct_batch.sum().item()

        batch_result["correct"] = correct
        batch_result["num_samples"] = batch_size
        batch_result["accuracy"] = correct / batch_size

        correct_per_class = targets.new_zeros((num_classes,))
        num_samples_per_class = targets.new_zeros((num_classes,))
        accuracy_per_class = pred_targets.new_zeros((num_classes,))

        for class_idx in range(num_classes):
            correct_per_class[class_idx] = correct_batch[targets == class_idx].sum().item()
            num_samples_per_class[class_idx] = (targets == class_idx).sum().item()
            accuracy_per_class[class_idx] = correct_per_class[class_idx] / num_samples_per_class[class_idx]

        batch_result["correct_per_class"] = correct_per_class
        batch_result["num_samples_per_class"] = num_samples_per_class
        batch_result["accuracy_per_class"] = accuracy_per_class

        return batch_result

    def process_batch_result(self, batch_result: Dict):
        # Aggregate results
        self.batch_results.append(batch_result)

        if len(self.batch_results) % self.report_frequency == 0 and self.purpose == "train":
            self.logger.info(f"Epoch {self.epoch} - Batch {len(self.batch_results)} - {self.purpose} loss "
                             f"- {batch_result['loss']:.4f}")

    def add_loss_fct_wrappers(self):
        if "loss_function" in self.config:
            loss_function_config = self.config['loss_function']

            if "scaling" in loss_function_config:
                for purpose in ["train", "val", "test"]:
                    self.loss_functions[purpose]["callable"] = scaling_wrapper(self.loss_functions[purpose]["callable"],
                                                                               loss_function_config)

            if "threshold_low" in loss_function_config and "threshold_high" in loss_function_config:
                for purpose in ["train", "val", "test"]:
                    self.loss_functions[purpose]["callable"] = thresholder_wrapper(
                        self.loss_functions[purpose]["callable"],
                        loss_function_config)

    def get_class_weights(self, sample_tensor: torch.Tensor) -> torch.Tensor:
        class_weights = None
        if self.class_weights is not None:
            class_weights = sample_tensor.new_tensor(self.class_weights)

        return class_weights


def thresholder_wrapper(loss_func, loss_fct_dict):
    def wrapper(*args, **kwargs):
        loss_fct_output = loss_func(*args, **kwargs)
        if isinstance(loss_fct_output, torch.Tensor) or type(loss_fct_output) == float:
            loss_fct_output = torch.clamp(input=loss_fct_output,
                                          min=loss_fct_dict["threshold_low"],
                                          max=loss_fct_dict["threshold_high"])
        elif type(loss_fct_output) == tuple:
            loss_fct_output = tuple([torch.clamp(input=x,
                                                 min=loss_fct_dict["threshold_low"],
                                                 max=loss_fct_dict["threshold_high"]) for x in loss_fct_output])
        else:
            raise ValueError
        return loss_fct_output
    return wrapper


def scaling_wrapper(loss_func, loss_fct_dict):
    def wrapper(*args, **kwargs):
        loss_fct_output = loss_func(*args, **kwargs)
        if isinstance(loss_fct_output, torch.Tensor) or type(loss_fct_output) == float:
            loss_fct_output = loss_fct_dict["scaling"] * loss_fct_output
        elif type(loss_fct_output) == tuple:
            loss_fct_output = tuple([loss_fct_dict["scaling"] * x for x in loss_fct_output])
        else:
            raise ValueError
        return loss_fct_output
    return wrapper


def pick_loss_aggregator(aggregator_dict: dict) -> Callable:
    if aggregator_dict["name"] == "mean":
        return torch.mean


def calc_domain_distance(data_domain_1: torch.Tensor, data_domain_2: torch.Tensor,
                         domain_distance_metric: str = "fid", normalize_activations=True, num_classes=None,
                         targets_domain_1: torch.Tensor = None, targets_domain_2: torch.Tensor = None,
                         pred_targets_domain_1: torch.Tensor = None,
                         pred_targets_domain_2: torch.Tensor = None) -> torch.Tensor:
    # Input: two 2D torch tensors (e.g. vectors) with the dimension (batch_size, feature_size)
    # In the case of lmmd, the order of the label predictions is important:
    # Per definition, the labeled source domain is defined as domain 1 and the unlabeled target domain as domain 2
    batch_size = data_domain_1.size(0)
    layer_dim = data_domain_1.size(1)

    assert data_domain_1.size() == data_domain_2.size(), "The tensors from both domains need to have the same size."

    if torch.isnan(data_domain_1).any() or torch.isnan(data_domain_2).any():
        logger.error("The tensor_domain_1 and / or tensor_domain_2 contains NaN elements in calc_domain_distance\n"
                     "--> skipping the domain distance")
        return data_domain_1.new_tensor(0.0)

    # weather to normalize the activations by the own magnitude
    if normalize_activations:
        # activation average over all batches from both domains
        mean_activations = torch.mean(torch.cat((data_domain_1, data_domain_2), dim=0), dim=0)
        # sum together over all activations
        sum_mean_activations = torch.sum(torch.abs(mean_activations))

        data_domain_1_normalized = data_domain_1 / sum_mean_activations
        data_domain_2_normalized = data_domain_2 / sum_mean_activations
    else:
        data_domain_1_normalized = data_domain_1 / layer_dim
        data_domain_2_normalized = data_domain_2 / layer_dim

    if domain_distance_metric == "fid":
        domain_distance = frechet_inception_distance(data_domain_1_normalized, data_domain_2_normalized)

        if normalize_activations:
            domain_distance = 10. ** 4 * domain_distance
        else:
            domain_distance = 10. ** 2 * domain_distance

    elif domain_distance_metric == "fid_legacy":
        domain_distance = frechet_inception_distance(data_domain_1, data_domain_2)

    elif domain_distance_metric == "mean_diff":
        mean_domain_1 = torch.mean(data_domain_1_normalized, dim=0)
        mean_domain_2 = torch.mean(data_domain_2_normalized, dim=0)

        # we are calculating the p=2 norm of the mean activations
        domain_distance = torch.dist(mean_domain_1, mean_domain_2, p=2)

    elif domain_distance_metric == "mengfan_wu":
        # Mengfan Wu's implementation
        std_labeled, mean_labeled = torch.std_mean(data_domain_1_normalized, dim=0)
        std_unlabeled, mean_unlabeled = torch.std_mean(data_domain_2_normalized, dim=0)
        diff_mean = torch.dist(mean_labeled, mean_unlabeled)
        diff_std = torch.dist(std_labeled, std_unlabeled, p=1)

        domain_distance = 10. * diff_mean + diff_std

    elif domain_distance_metric == "mmd":
        # the alternative option is to use the torch-two-sample library
        # mmd_statistic = MMDStatistic(batch_size, batch_size)
        # domain_distance = mmd_statistic(domain_1_normalized, domain_2_normalized, alphas=[1. / 2.])**2

        domain_distance = maximum_mean_discrepancy(data_domain_1, data_domain_2) ** 2

    elif domain_distance_metric == "cosine_similarity":
        mean_domain_1 = torch.mean(data_domain_1_normalized, dim=0)
        mean_domain_2 = torch.mean(data_domain_2_normalized, dim=0)

        domain_distance = data_domain_1.new_tensor(1.0) - F.cosine_similarity(mean_domain_1, mean_domain_2, dim=0)

    elif domain_distance_metric == "coral":
        domain_distance = coral(data_domain_1, data_domain_2)

    elif domain_distance_metric == "cmmd":
        domain_distance = conditional_maximum_mean_discrepancy(num_classes,
                                                               data_domain_1, data_domain_2,
                                                               targets_domain_1, targets_domain_2)

    elif domain_distance_metric == "lmmd":
        # In the case of lmmd, the order of the label predictions is important:
        # Per definition, the labeled source domain is defined as domain 1 and the unlabeled target domain as domain 2
        domain_distance = local_maximum_mean_discrepancy(num_classes,
                                                         data_domain_1, data_domain_2,
                                                         targets_domain_1, pred_targets_domain_2)

    else:
        raise NotImplementedError(f"The domain distance metric {domain_distance_metric} is not implemented.")

    return domain_distance
