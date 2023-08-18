import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

import lstm_adversarial_attack.config_paths as cfp


# https://stackoverflow.com/a/75036813
class MyFormatter(logging.Formatter):
    def formatTime(self, record: logging.LogRecord, datefmt: str = None):
        if not datefmt:
            return super().formatTime(record, datefmt=datefmt)

        return datetime.fromtimestamp(record.created).strftime(datefmt)


class SimpleLogWriter:
    def __init__(
        self,
        name: str,
        log_file: Path,
        data_col_names: tuple[str, ...],
        output_format: str = "%(asctime)s,%(message)s",
        date_format: str = "%Y-%m-%d %H:%M:%S.%f",
    ):
        self._data_col_names = data_col_names
        self._log_file = log_file
        self._name = name
        # self._name = f"{self._name_prefix}_{uuid.uuid1().int >> 64}"
        self._output_format = output_format
        self._date_format = date_format
        self._logger = logging.getLogger(self._name)
        self._post_init()

    def _post_init(self):
        self._validate_log_file()
        self._set_up_logger()

    def _validate_log_file(self):
        if not self._log_file.exists():
            with self._log_file.open(mode="w") as out_file:
                headers = ",".join(("timestamp", *self._data_col_names))
                out_file.write(f"{headers}\n")
        else:
            existing_data = pd.read_csv(self._log_file)
            assert existing_data.shape[1] == len(self._data_col_names) + 1

    def _set_up_logger(self):
        handler = logging.FileHandler(self._log_file)
        formatter = MyFormatter(
            self._output_format,
            self._date_format,
        )
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)
        self._logger.setLevel(logging.INFO)

    def write_data(self, data: tuple[str, ...]):
        assert len(data) == len(self._data_col_names)
        self._logger.info(",".join(data))

    @property
    def log_file(self) -> Path:
        return self._log_file


if __name__ == "__main__":
    my_logger = SimpleLogWriter(
        name_prefix="test_logger",
        log_file=cfp.HYPERPARAMETER_OUTPUT_DIR / "test_log.csv",
        data_col_names=("a", "b", "c"),
    )
    my_logger.write_data(data=("1", "2", "3"))

    df_result = pd.read_csv(my_logger.log_file)
