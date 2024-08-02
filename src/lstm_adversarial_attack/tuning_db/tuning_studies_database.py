import datetime
import os
from functools import cached_property
from pathlib import Path
from typing import Any
from urllib.parse import quote

import optuna.study
from dotenv import load_dotenv
from optuna.storages import RDBStorage

import lstm_adversarial_attack.config_paths as cfp


def get_db_dotenv_info(
    db_name_var: str,
    dotenv_path: Path = cfp.TUNING_DBS_DOTENV_PATH,
    username_var: str = "TUNING_DBS_USER",
    password_file_var: str = "TUNING_DBS_PASSWORD_FILE",
    host_var: str = "POSTGRES_DBS_HOST",
) -> dict:
    load_dotenv(dotenv_path=dotenv_path)
    password_file = os.getenv(password_file_var)
    with Path(password_file).open(mode="r") as in_file:
        password = in_file.read()

    return {
        "user": os.getenv(username_var),
        "password": password,
        "db_name": os.getenv(db_name_var),
        "host": os.getenv(host_var),
    }


class OptunaDatabase:
    def __init__(
        self,
        user: str,
        password: str,
        db_name: str,
        host: str,
        db_dialect: str = "postgresql",
        db_driver: str = "psycopg2",
    ):
        self._user = user
        self._password = password
        self._db_name = db_name
        self._host = host
        self._db_dialect = db_dialect
        self._db_driver = db_driver

    @classmethod
    def from_dotenv_info(
        cls,
        dot_env_path: Path,
        env_var_user: str,
        env_var_password_file: str,
        env_var_db_name: str,
    ):
        pass

    @property
    def _encoded_password(self) -> str:
        return quote(self._password, safe="")

    @property
    def _db_url(self) -> str:
        return (
            f"{self._db_dialect}+{self._db_driver}://"
            f"{self._user}:{self._encoded_password}@"
            f"{self._host}/{self._db_name}"
        )

    @property
    def db_name(self) -> str:
        return self._db_name

    @cached_property
    def storage(self) -> RDBStorage:
        return RDBStorage(url=self._db_url)

    @property
    def study_summaries(self) -> list[optuna.study.StudySummary]:
        return optuna.study.get_all_study_summaries(storage=self.storage)

    def is_in_db(self, study_name: str) -> bool:
        db_study_names = [item.study_name for item in self.study_summaries]
        return study_name in db_study_names

    def get_study_summary(self, study_name: str) -> optuna.study.StudySummary:
        return [
            summary
            for summary in self.study_summaries
            if summary.study_name == study_name
        ][0]

    def get_best_params(self, study_name: str) -> dict[str, Any]:
        study_summary = self.get_study_summary(study_name=study_name)
        return study_summary.best_trial.params

    def get_all_studies(self) -> list[optuna.Study]:
        # reduce logging verbosity to avoid msg about creating study from db
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study_names = [item.study_name for item in self.study_summaries]
        return [
            optuna.create_study(
                study_name=study_name,
                storage=self.storage,
                load_if_exists=True,
            )
            for study_name in study_names
        ]

    @staticmethod
    def get_last_update_time(study: optuna.Study) -> datetime.datetime:
        trial_complete_times = [
            trial.datetime_complete
            for trial in study.trials
            if trial.datetime_complete is not None
        ]
        return (
            max(trial_complete_times)
            if len(trial_complete_times) > 0
            else datetime.datetime(1, 1, 1)
        )

    def get_latest_study(self) -> optuna.Study:
        sorted_studies = sorted(
            self.get_all_studies(),
            key=lambda x: self.get_last_update_time(x),
            reverse=True,
        )
        return sorted_studies[0]


model_tuning_db_info = get_db_dotenv_info(db_name_var="MODEL_TUNING_DB_NAME")
MODEL_TUNING_DB = OptunaDatabase(**model_tuning_db_info)
MODEL_TUNING_STORAGE = MODEL_TUNING_DB.storage

attack_tuning_db_info = get_db_dotenv_info(db_name_var="ATTACK_TUNING_DB_NAME")
ATTACK_TUNING_DB = OptunaDatabase(**attack_tuning_db_info)
ATTACK_TUNING_STORAGE = ATTACK_TUNING_DB.storage

# if __name__ == "__main__":
# model_tuning_db = OptunaDatabase(env_var_db_name="MODEL_TUNING_DB_NAME")
# attack_tuning_db = OptunaDatabase(env_var_db_name="ATTACK_TUNING_DB_NAME")
