import json
from abc import ABC
from pathlib import Path


class ConfigReader:
    def __init__(self, config_path: Path = Path("config.json")):
        self._config_path = config_path

    def read_module_config(self, sub_package_name: str,
                           param_names: list[str]) -> dict:
        module_config = {}
        with self._config_path.open(mode="r") as config_file:
            sub_package_config = json.load(config_file)
        for param_name in param_names:
            module_config[param_name] = sub_package_config.get(
                param_name, None)
        return module_config


class ConfigInfo(ABC):
    def __init__(self, sub_package_name: str, param_names: list[str]):
        config_reader = ConfigReader()


if __name__ == "__main__":
    config_reader = ConfigReader()
