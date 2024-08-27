import pprint
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Callable
from enum import Enum, auto

sys.path.append(str(Path(__file__).parent.parent.parent))
from lstm_adversarial_attack.config.read_write import (
    CONFIG_MODIFIER,
    CONFIG_READER,
    PATH_CONFIG_READER,
)


def get_config_value(config_key: str) -> str:
    """
    Gets value from config.toml
    :param config_key: config.toml key as dotted string
    :return: value corresponding to config.toml key
    """

    return CONFIG_READER.get_value(config_key=config_key)


def set_config_value(config_key: str, value: Any):
    """
    Sets value from config.toml
    :param config_key: config.toml key as dotted string
    :param value: value to assign to key
    :return: None
    """
    CONFIG_MODIFIER.set(config_key=config_key, value=value)


class SessionType(Enum):
    DB_QUERIES = auto()
    PREPROCESS = auto()
    MODEL_TUNING = auto()
    CV_TRAINING = auto()
    ATTACK_TUNING = auto()
    ATTACK = auto()
    ATTACK_ANALYSIS = auto()


@dataclass
class SessionInfo:
    session_type: SessionType
    session_id: int | str
    comment: str


class PipelineInfo:
    """
    Container metadata of data-generating sessions. Intended for use in Jupyter
     notebooks.
    """
    def __init__(
        self,
        sessions: dict[str, SessionInfo] = None,
        next_session_index: int = 1,
    ):
        if sessions is None:
            sessions = {}
        self.sessions = sessions
        self.next_next_session_index = next_session_index

    _output_root_path_keys = {
        SessionType.DB_QUERIES: "db.output_root",
        SessionType.PREPROCESS: "preprocess.output_root",
        SessionType.MODEL_TUNING: "model.tuner_driver.output_dir",
        SessionType.CV_TRAINING: "model.cv_driver.output_dir",
        SessionType.ATTACK_TUNING: "attack.tuner_driver.output_dir",
        SessionType.ATTACK: "attack.attack_driver.output_dir",
    }

    def _get_output_root(self, session_type: SessionType) -> Path:
        return Path(
            PATH_CONFIG_READER.read_path(
                self._output_root_path_keys[session_type]
            )
        )

    def _get_newest_session_id_from_dirs(
        self, session_type: SessionType
    ) -> str:
        root_output_dir = self._get_output_root(session_type)
        return max([item.name for item in root_output_dir.iterdir()])

    def store_session(
        self,
        session_type: SessionType,
        session_id: str | int = None,
        comment: str = None,
    ):
        """
        Stores info for a data generating session
        :param session_type: type of session
        :param session_id: ID of session
        :param comment: optional comment
        :return: None
        """
        if session_id is None:
            session_id = self._get_newest_session_id_from_dirs(
                session_type=session_type
            )

        root_output_dir = self._get_output_root(session_type)
        session_output_dir = root_output_dir / str(session_id)
        assert session_output_dir.exists()

        if session_id in self.sessions:
            print(f"{session_id} already stored")
            return

        self.sessions[str(session_id)] = SessionInfo(
            session_type=session_type,
            session_id=session_id,
            comment=comment,
        )
        print(f"{session_id} stored")

    def get_stored_session(
        self,
        session_type: SessionType,
        session_id: str | int = None,
    ) -> SessionInfo:
        """
        Gets info on a stored session. If PipelineInfo object has more than one
        entry for session of session_type, must specify session ID.
        :param session_type: Type of session to retrieve
        :param session_id: ID of session
        :return: info for session
        """

        session_info = None

        # if session_id not specified, we still retrieve a session if there
        # is exactly on session of type session_type.
        if session_id is None:
            all_sessions_of_type = {
                key: val
                for key, val in list(self.sessions.items())
                if val.session_type == session_type
            }
            if len(all_sessions_of_type) == 1:
                session_info = all_sessions_of_type[
                    list(all_sessions_of_type.keys())[0]
                ]
            elif len(all_sessions_of_type) == 0:
                print(f"No sessions found for type {session_type}")
            elif len(all_sessions_of_type) > 1:
                print(
                    f"Multiple sessions of type {session_type}. "
                    f"Must specify session ID"
                )

        # if sesson_id is specified, we confirm it is of session_type, and
        # then retrieve it
        if session_id is not None:
            if (
                str(session_id) in self.sessions
                and self.sessions[str(session_id)].session_type == session_type
            ):
                session_info = self.sessions[str(session_id)]

            if str(session_id) not in self.sessions:
                print(f"No session found for type {session_type}")
            elif self.sessions[str(session_id)].session_type != session_type:
                print(f"Session {session_id} does not have type {session_type}")

        # Note that return value can be None if session was not retreived above
        print(f"Retrieved session: {session_info.session_id} ")
        return session_info


if __name__ == "__main__":
    pipeline_info = PipelineInfo()
    pipeline_info.store_session(session_type=SessionType.DB_QUERIES)
    pipeline_info.store_session(
        session_type=SessionType.DB_QUERIES, session_id=20240812154615226070
    )
    db_session = pipeline_info.get_stored_session(session_type=SessionType.DB_QUERIES)
    preprocess_session = pipeline_info.get_stored_session(
        session_type=SessionType.PREPROCESS
    )
