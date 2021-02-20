import json
import sys
import logging
from typing import Text

from src.utils import project_path
from src.utils.log import init_logging, create_base_logger, create_logdir
from src.utils.command_line_handler import CommandLineHandler
from src.visualization.results_plotter import ResultsPlotter


def command_line():
    args = CommandLineHandler.handle()
    # path to config.json in experiment log dir
    config_path = args.config_path
    return main(config_path)


def main(config_path: Text):
    init_logging()
    config_filepath = project_path(config_path)
    with open(config_filepath, "r") as exp_fp:
        config = json.load(exp_fp)

    logdir = create_logdir(f"visualization_{config_filepath.parent.name}")
    logger = create_base_logger(logdir, name="visualization")

    with open(str(logdir / "config.json"), "w") as fp:
        json.dump(config, fp, indent=4)

    for dir in config_filepath.parent.iterdir():
        dirname = dir.name
        if dir.is_dir() and dirname.startswith("seed"):
            dirname_split = dirname.split("_")
            seed = dirname_split[1]
            seed_logdir = logdir / dirname

            results_hdf5_path = dir / "learning_results.hdf5"
            results_plotter = ResultsPlotter(results_hdf5_path=results_hdf5_path, logdir=seed_logdir,
                                             remote=config.get("remote", False),
                                             **config["experiment"].get("visualization", {}))
            results_plotter.plot()


if __name__ == '__main__':
    command_line()
    sys.exit(0)
