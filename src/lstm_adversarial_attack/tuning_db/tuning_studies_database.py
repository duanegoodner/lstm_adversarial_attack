import os
from functools import cached_property
from pathlib import Path

from dotenv import load_dotenv
from optuna.storages import RDBStorage

import lstm_adversarial_attack.config_paths as cfp


class OptunaDatabase:
    def __init__(self, dotenv_path: Path):
        self._dotenv_path = dotenv_path

    @property
    def _db_url(self) -> str:
        load_dotenv(dotenv_path=self._dotenv_path)
        user = os.getenv("OPTUNA_DATABASE_USER")
        password = os.getenv("OPTUNA_DATABASE_PASSWORD")
        host = os.getenv("OPTUNA_DATABASE_HOST")
        database_name = os.getenv("OPTUNA_DATABASE_NAME")
        return f"postgresql://{user}:{password}@{host}/{database_name}"

    @cached_property
    def storage(self) -> RDBStorage:
        return RDBStorage(url=self._db_url)


TUNING_STUDIES_DATABASE = OptunaDatabase(dotenv_path=cfp.OPTUNA_DB_DOTENV_PATH)
TUNING_STUDIES_STORAGE = TUNING_STUDIES_DATABASE.storage


# if __name__ == "__main__":
#     db = OptunaDatabase(dotenv_path=cfp.OPTUNA_DB_DOTENV_PATH)
#     storage = db.storage