import json
import sys
import logging
from typing import Text

from src.dataset_generation import pick_dataset_generator
from src.utils import project_path
from src.learning.experiments import ExperimentSet
from src.utils.log import init_logging
from src.utils.command_line_handler import CommandLineHandler


def command_line():
    args = CommandLineHandler.handle()
    config_path = args.config_path
    return main(config_path)


def main(config_path: Text):
    init_logging()
    experiment_filepath = project_path(config_path)
    with open(experiment_filepath, "r") as exp_fp:
        config = json.load(exp_fp)

    dataset_generator = pick_dataset_generator(**config["dataset_generation"])
    dataset_generator.run()


if __name__ == '__main__':
    command_line()
    sys.exit(0)
