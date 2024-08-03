import sys
from pathlib import Path
import logging
from typing import TextIO


def open_log_file(log_file_path: Path) -> TextIO:
    log_file_fid = log_file_path.open(mode="a", buffering=1)
    return log_file_fid


def redirect_output_to_fid(log_file_fid: TextIO) -> None:
    sys.stdout = log_file_fid
    sys.stderr = log_file_fid


def configure_optuna_logging(log_file_fid: TextIO):
    logger = logging.getLogger("optuna")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(log_file_fid)  # Line buffering
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s:%(message)s'))

    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(handler)


def set_redirection(log_file_path: Path, include_optuna: bool = False) -> TextIO:
    print(f"stdout and stderr will be redirected to {str(log_file_path)}")
    print(f"Output can be viewed in real time by running the following command in another terminal:\n"
          f"tail -f {str(log_file_path)}")
    log_file_fid = open_log_file(log_file_path=log_file_path)
    redirect_output_to_fid(log_file_fid=log_file_fid)
    if include_optuna:
        configure_optuna_logging(log_file_fid=log_file_fid)
    return log_file_fid


# def flushed_print(*args, **kwargs):
#     __builtins__.print(*args, **kwargs, flush=True)