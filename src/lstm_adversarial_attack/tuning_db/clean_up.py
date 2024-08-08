"""
Dev tools for cleaning up database and/or file storage from tuning studies.
For safety, recommend instantiating TuningOutputCleaner objects in console, and
then manually running the remove_... methods.
"""

import sys
from typing import List

import optuna
from pathlib import Path
from optuna.storages import RDBStorage


import tuning_studies_database as tsd

sys.path.append(str(Path(__file__).parent.parent.parent))
from lstm_adversarial_attack.config.read_write import PATH_CONFIG_READER


def rmdir_recursive(directory: Path):
    """
    Recursively delete the contents of a directory.
    """
    if not directory.is_dir():
        raise ValueError(f"{directory} is not a directory.")

    for item in directory.iterdir():
        if item.is_dir():
            rmdir_recursive(item)
        else:
            item.unlink()
    directory.rmdir()


class TuningOutputCleaner:
    """
    Identifies tuning-study-output-directories and RDB-stored-studies for a
    particular tuning study type (e.g. model tuning, or attack tuning). Offers
    methods for removing directories / studies which do not have a partner
    study / directory.
    """

    def __init__(self, storage: RDBStorage, path_config_key: str):
        self.storage = storage
        self.path_config_key = path_config_key
        self.confirm_dir_db_session_id_overlap()

    @staticmethod
    def get_session_id_from_study_name(study_name: str) -> str:
        return "".join([char for char in study_name if char.isdigit()])

    @property
    def output_root_dir(self) -> Path:
        return Path(
            PATH_CONFIG_READER.read_path(config_key=self.path_config_key)
        )

    @property
    def study_names(self) -> List[str]:
        return optuna.study.get_all_study_names(storage=self.storage)

    @property
    def session_ids_from_study_names(self) -> List[str]:
        return [
            self.get_session_id_from_study_name(study_name=study_name)
            for study_name in self.study_names
        ]

    @property
    def session_ids_from_dirs(self) -> List[str]:
        return [item.name for item in self.output_root_dir.iterdir()]

    @property
    def sessions_with_study_and_dir(self) -> List[str]:
        set_intersection = set(self.session_ids_from_study_names)
        return sorted(list(set_intersection))

    def confirm_dir_db_session_id_overlap(self):
        assert len(self.sessions_with_study_and_dir) > 0

    def has_matching_dir(self, study_name: str) -> bool:
        return (
            self.get_session_id_from_study_name(study_name=study_name)
            in self.session_ids_from_dirs
        )

    def has_matching_study(self, tuning_session_dir: Path) -> bool:
        return tuning_session_dir.name in self.session_ids_from_study_names

    @property
    def unmatched_studies(self) -> List[str]:
        return [
            item
            for item in self.study_names
            if self.get_session_id_from_study_name(study_name=item)
            not in self.session_ids_from_dirs
        ]

    @property
    def unmatched_dirs(self) -> List[Path]:
        return [
            item
            for item in self.output_root_dir.iterdir()
            if item.name not in self.session_ids_from_study_names
        ]

    def remove_unmatched_studies(self) -> List[str]:
        deleted_study_names = []
        for study_name in self.unmatched_studies:
            study_id = self.storage.get_study_id_from_name(
                study_name=study_name
            )
            self.storage.delete_study(study_id=study_id)
            deleted_study_names.append(study_name)
        return deleted_study_names

    def remove_unmatched_dirs(self) -> List[Path]:
        deleted_dirs = []
        for session_dir in self.unmatched_dirs:
            rmdir_recursive(directory=session_dir)
            deleted_dirs.append(session_dir)
        return deleted_dirs


if __name__ == "__main__":
    storage_to_output_dir = {
        "MODEL_TUNING_STORAGE": "model.tuner_driver.output_dir",
        "ATTACK_TUNING_STORAGE": "attack.tuner_driver.output_dir",
    }

    model_tuning_cleaner = TuningOutputCleaner(
        storage=tsd.MODEL_TUNING_STORAGE,
        path_config_key="model.tuner_driver.output_dir",
    )

    attack_tuning_cleaner = TuningOutputCleaner(
        storage=tsd.ATTACK_TUNING_STORAGE,
        path_config_key="attack.tuner_driver.output_dir",
    )
