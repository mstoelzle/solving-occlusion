import getpass
import logging
import datetime
import json
import os
import pathlib
import psutil

LOGGER_ROOT = "solving_occlusion"


def create_base_logger(logdir: pathlib.Path, name: str = "") -> logging.Logger:
    logger = get_logger(name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(logdir / "run.log")
    fmt_str = '{"time": "%(asctime)s", "logger_name": %(name)s", "level":"%(levelname)s", "message": "%(message)s"}'
    formatter = logging.Formatter(fmt_str)
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    return logger


def get_logger(name: str = "") -> logging.Logger:
    logger_name = LOGGER_ROOT
    if len(name) != 0:
        logger_name += f".{name}"
    return logging.getLogger(logger_name)


def get_timestring():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def init_logging(level=logging.INFO):
    logging.basicConfig(level=level)


def create_logdir(name: str):
    """
    creates a logdir for an ExperimentSet instance. It contains all other logdirs for all Experiments.
    :return: an existing logdir path.
    """
    logdir = pathlib.Path("logs") / f"{get_timestring()}_{name}_{getpass.getuser()}"
    logdir.mkdir(parents=True, exist_ok=False)

    return logdir


def log_memory_usage(name: str, logger: logging.Logger = None):
    process = psutil.Process(os.getpid())
    msg = f"Used memory {name}: {(process.memory_info().rss / 1000000):.1f} MBytes"

    if logger is None:
        print(msg)
    else:
        logger.info(msg)
