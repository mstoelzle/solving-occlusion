import logging
import datetime
import pathlib


ROOT_DIR = pathlib.Path("/media/sdb/path_learning")
LOGDIR = ROOT_DIR / "logs"

LOGGER_ROOT = "path_learning.experiment"


def create_base_logger(logdir: pathlib.Path) -> logging.Logger:
    logger = get_logger("")
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


