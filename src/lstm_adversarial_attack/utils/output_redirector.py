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


def configure_optuna_logging(fid: TextIO):
    logger = logging.getLogger("optuna")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(fid)  # Line buffering
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s:%(message)s'))

    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(handler)

def flushed_print(*args, **kwargs):
    __builtins__.print(*args, **kwargs, flush=True)


