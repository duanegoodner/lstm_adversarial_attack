import json
import toml
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


class ConfigReader:
    def __init__(self, config_path: Path = None):
        if config_path is None:
            config_path = Path("config.toml")
        self._config_path = config_path

    def read_module_config(self, sub_package_name: str,
                           param_names: list[str]) -> dict:
        module_config = {}
        with self._config_path.open(mode="r") as config_file:
            full_config = toml.load(config_file)
            sub_package_config = toml.load(config_file)[sub_package_name]
        for param_name in param_names:
            module_config[param_name] = sub_package_config.get(
                param_name, None)
        return module_config


class ModuleSettingsBuilder:
    def __init__(self, sub_package_name: str, param_names: list[str],
                 settings_constructor: Callable[..., dataclass],
                 custom_config_path: Path = None
                 ):
        self._sub_package_name = sub_package_name
        self._param_names = param_names
        self._settings_constructor = settings_constructor
        self._custom_config_path = custom_config_path

    def build(self) -> dataclass:
        config_reader = ConfigReader(config_path=self._custom_config_path)
        module_config = config_reader.read_module_config(
            sub_package_name=self._sub_package_name,
            param_names=self._param_names
        )
        return self._settings_constructor(**module_config)


@dataclass
class NewPrefilterSettings:
    """
    Container for objects imported by Prefilter
    """

    min_age: int
    min_los_hospital: int
    min_los_icu: int
    bg_data_cols: list[str]
    lab_data_cols: list[str]
    vital_data_cols: list[str]


@dataclass
class ModuleASettings:
    param_a1: int
    param_a2: int


if __name__ == "__main__":
    settings_builder = ModuleSettingsBuilder(
        sub_package_name="preprocess",
        param_names=["min_age", "min_los_hospital", "min_los_icu",
                     "bg_data_cols", "lab_data_cols", "vital_data_cols"],
        settings_constructor=NewPrefilterSettings
    )
    settings = settings_builder.build()
    print(settings)
