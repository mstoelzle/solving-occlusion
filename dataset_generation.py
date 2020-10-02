import json
import sys
import logging
from typing import Text

from src.dataset_generation import pick_dataset_generator
from src.utils import project_path
from src.learning.experiments import ExperimentSet
from src.utils.log import init_logging, create_logdir
from src.utils.command_line_handler import CommandLineHandler


def command_line():
    args = CommandLineHandler.handle()
    config_path = args.config_path
    return main(config_path)


def main(config_path: Text):
    init_logging()
    config_filepath = project_path(config_path)
    with open(config_filepath, "r") as exp_fp:
        config = json.load(exp_fp)

    dataset_generator = pick_dataset_generator(name=config["name"], remote=config.get("remote", False),
                                               **config["dataset_generation"])
    with dataset_generator:
        dataset_generator.run()


if __name__ == '__main__':
    command_line()
    sys.exit(0)
