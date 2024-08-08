import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Callable

sys.path.append(str(Path(__file__).parent.parent.parent))
from lstm_adversarial_attack.config.read_write import (
    CONFIG_MODIFIER,
    CONFIG_READER,
    PATH_CONFIG_READER,
)
from lstm_adversarial_attack.attack.attack_data_structs import (
    ATTACK_DRIVER_SUMMARY_IO,
    ATTACK_TUNER_DRIVER_SUMMARY_IO,
)


def get_config_value(config_key: str):
    """
    Gets value from config.toml
    """
    return CONFIG_READER.get_value(config_key=config_key)


def set_config_value(config_key: str, value: Any):
    """
    Sets value in config.toml
    """
    CONFIG_MODIFIER.set(config_key=config_key, value=value)


def get_session_output_dirs(path_config_key: str) -> List[Path]:
    output_root = Path(
        PATH_CONFIG_READER.read_path(config_key=path_config_key)
    )
    return list(output_root.iterdir())


@dataclass
class SessionIDs:
    db_queries: str = None
    preprocess: str = None
    model_tuning: str = None
    cv_training: str = None
    attack_tuning_sparse_small: str = None
    attack_tuning_sparse_small_max: str = None
    attack_sparse_small: str = None
    attack_sparse_small_max: str = None

    output_root_path_keys = {
        "db_queries": "db.output_root",
        "preprocess": "preprocess.output_root",
        "model_tuning": "model.tuner_driver.output_dir",
        "cv_training": "model.cv_driver.output_dir",
        "attack_tuning": "attack.tuner_driver.output_dir",
        "attack": "attack.attack_driver.output_dir",
    }

    @property
    def special_setters(self) -> dict[str, Callable]:
        return {
            "attack_tuning_sparse_small": self.set_attack_tuning_session_id,
            "attack_tuning_sparse_small_max": self.set_attack_tuning_session_id,
            "attack_sparse_small": self.set_attack_session_id,
            "attack_sparse_small_max": self.set_attack_session_id,
        }

    @property
    def attack_tuning_session_output_dirs(self) -> List[Path]:
        return get_session_output_dirs(
            path_config_key=self.output_root_path_keys["attack_tuning"]
        )

    @property
    def attack_session_output_dirs(self) -> List[Path]:
        return get_session_output_dirs(
            path_config_key=self.output_root_path_keys["attack"]
        )

    @staticmethod
    def get_attack_tuning_objective_name(session_output_dir: Path) -> str:

        attack_tuning_session_id = session_output_dir.name
        attack_tuner_driver_summary_path = (
            session_output_dir
            / f"attack_tuner_driver_summary_{attack_tuning_session_id}.json"
        )
        attack_tuner_driver_summary = (
            ATTACK_TUNER_DRIVER_SUMMARY_IO.import_to_struct(
                path=attack_tuner_driver_summary_path
            )
        )
        return attack_tuner_driver_summary.settings.objective_name

    def set_attack_tuning_session_id(
        self, attr_name: str, session_id: str = None
    ):

        objective_name = attr_name.strip("attack_tuning_")

        if session_id is None:
            session_output_dirs = [
                item
                for item in self.attack_tuning_session_output_dirs
                if self.get_attack_tuning_objective_name(
                    session_output_dir=item
                )
                == objective_name
            ]
            session_id = max(session_output_dirs).name

        output_root_dir = Path(
            PATH_CONFIG_READER.read_path(
                config_key=self.output_root_path_keys["attack_tuning"]
            )
        )

        session_output_dir = output_root_dir / str(session_id)
        assert session_output_dir.exists()
        assert (
            self.get_attack_tuning_objective_name(
                session_output_dir=session_output_dir
            )
            == objective_name
        )

        setattr(self, attr_name, str(session_id))

    def get_attack_objective_name(
        self, attack_session_output_dir: Path
    ) -> str:
        attack_driver_summary_path = (
            attack_session_output_dir
            / f"attack_driver_summary_{attack_session_output_dir.name}.json"
        )
        attack_driver_summary = ATTACK_DRIVER_SUMMARY_IO.import_to_struct(
            path=attack_driver_summary_path
        )
        attack_tuning_session_id = attack_driver_summary.attack_tuning_id

        attack_tuning_output_dir = (
            Path(
                PATH_CONFIG_READER.read_path(
                    config_key=self.output_root_path_keys["attack_tuning"]
                )
            )
            / attack_tuning_session_id
        )
        return self.get_attack_tuning_objective_name(
            session_output_dir=attack_tuning_output_dir
        )

    def set_attack_session_id(self, attr_name: str, session_id: str = None):
        objective_name = attr_name.strip("attack_")

        if session_id is None:
            session_output_dirs = [
                item
                for item in self.attack_session_output_dirs
                if self.get_attack_objective_name(
                    attack_session_output_dir=item
                )
                == objective_name
            ]
            session_id = max(session_output_dirs).name

        output_root_dir = Path(
            PATH_CONFIG_READER.read_path(
                config_key=self.output_root_path_keys["attack"]
            )
        )

        session_output_dir = output_root_dir / str(session_id)
        assert session_output_dir.exists()
        assert (
            self.get_attack_objective_name(
                attack_session_output_dir=session_output_dir
            )
            == objective_name
        )

        setattr(self, attr_name, str(session_id))

    def set_standard(self, attr_name: str, session_id: str | int = None):
        output_root_path = Path(
            PATH_CONFIG_READER.read_path(
                config_key=self.output_root_path_keys[attr_name]
            )
        )
        if session_id is None:
            session_id = max(
                [item.name for item in output_root_path.iterdir()]
            )

        assert (output_root_path / str(session_id)).exists()
        setattr(self, attr_name, str(session_id))

    def set(self, attr_name: str, session_id: str | int = None):
        if attr_name in self.special_setters:
            setter = self.special_setters[attr_name]
            setter(attr_name=attr_name, session_id=session_id)
        else:
            self.set_standard(
                attr_name=attr_name,
                session_id=session_id,
            )


if __name__ == "__main__":
    session_ids = SessionIDs()

    session_ids.set(attr_name="db_queries", session_id=20240808074353330946)
    session_ids.set(attr_name="preprocess", session_id=20240806222518919168)
    session_ids.set(attr_name="model_tuning", session_id=20240807155921818488)
    session_ids.set(attr_name="cv_training", session_id=20240807161712566160)
    session_ids.set(
        attr_name="attack_tuning_sparse_small", session_id=20240807174443459947
    )
    session_ids.set(
        attr_name="attack_tuning_sparse_small_max",
        session_id=20240807174140493489,
    )
    session_ids.set(attr_name="attack_sparse_small")
    session_ids.set(attr_name="attack_sparse_small_max")
