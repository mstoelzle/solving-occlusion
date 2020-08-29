import datetime
import json
import math
import pathlib
from typing import List, Tuple, Dict

import pandas as pd

from ..log import get_logger
from ..result import ExperimentSetResult

COLUMNS: List[str] = ["Experiment Date", "Experiment Time", "Task Date", " Task Time", "Experiment", "Seed", "Hash",
                      "Complete", "Task-step", "Task Type", "model-to-train", "Dataloaders", "Domains", "Loss config",
                      "Epoch",
                      "Loss-train", "Loss-val", "Loss-test", "Loss-blame-train", "Loss-blame-val", "Loss-blame-test",
                      "Acc-train", "Acc-val", "Acc-test",
                      "Acc-per-class-train", "Acc-per-class-val", "Acc-per-class-test",
                      "Task Duration", "User", "Logdir"]

logger = get_logger("gather_results")


def gather_results_df(exp_set_names: List[str]) -> pd.DataFrame:
    """
    :param exp_set_names: List of names of experiments to gather results for.
    :return: A pd.DataFrame containing the results for the experiments.
    """
    logger.info(f"Gathering Results for Experiment Sets {exp_set_names}")

    df = pd.DataFrame(columns=COLUMNS)

    for exp_set_name in exp_set_names:
        count = 0
        for exp_set_dir in [child for child in logdir.iterdir() if child.is_dir() and exp_set_name in child.stem]:
            experiment_set = ExperimentSetResult(exp_set_dir)

            if exp_set_name == experiment_set.name:
                for experiment in experiment_set.experiments:
                    for task in experiment.tasks:

                        new_row = (experiment_set.start_date_str,
                                   experiment_set.start_time_str,
                                   task.start_date_str,
                                   task.start_time_str,
                                   experiment_set.name,
                                   experiment.seed,
                                   experiment.hash,
                                   task.complete,
                                   task.uid,
                                   task.config["task_type"],
                                   task.config["model_to_train"],
                                   json.dumps(task.config["dataloaders"]),
                                   json.dumps(task.config["domains"]),
                                   json.dumps(task.config["loss"]),
                                   task.best_epoch,
                                   task.train_loss,
                                   task.val_loss,
                                   task.test_loss,
                                   json.dumps(task.train_loss_blame),
                                   json.dumps(task.val_loss_blame),
                                   json.dumps(task.test_loss_blame),
                                   task.train_acc,
                                   task.val_acc,
                                   task.test_acc,
                                   json.dumps(task.train_acc_per_class),
                                   json.dumps(task.val_acc_per_class),
                                   json.dumps(task.test_acc_per_class),
                                   task.duration,
                                   experiment_set.user,
                                   str(task.dir.resolve()))

                        df.loc[count] = new_row
                        count += 1

    logger.info("Sorting results before upload")
    df = df.sort_values(by=["Experiment Date", "Experiment Time", 'Seed', 'Task-step'])
    df = df.fillna("")
    return df

