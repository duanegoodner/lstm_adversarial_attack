import pprint
import toml
from pathlib import Path
from typing import Any


class ConfigReader:
    def __init__(self, config_path: Path = None):
        if config_path is None:
            config_path = Path(__file__).parent / "config.toml"
        self._config_path = config_path
        with self._config_path.open(mode="r") as config_file:
            self._config = toml.load(config_file)

    def get_config_value(self, config_key: str) -> Any:
        result = self._config.copy()
        for sub_key in config_key.split("."):
            if result.get(sub_key) is None:
                return None
            else:
                result = result.get(sub_key)

        return result


class PathConfigReader(ConfigReader):
    def __init__(self, config_path: Path = None):
        if config_path is None:
            config_path = Path(__file__).parent / "config_paths.toml"
        super().__init__(config_path=config_path)

    @property
    def project_root(self) -> Path:
        return Path(__file__).parent.parent.parent

    def _to_absolute_path(self, path_rel_project_root: str | Path) -> str:
        relative_path = Path(path_rel_project_root)
        if relative_path.is_absolute():
            raise TypeError(
                f"{path_rel_project_root} is an absolute path. "
                f"Must be a relative path."
            )
        return str(self.project_root / relative_path)

    def read_path(
        self, config_key: str, extension: str = ""
    ) -> str | list[str] | dict[str, str]:
        config_val = self.get_config_value(config_key=f"paths.{config_key}")
        if type(config_val) is str:
            return self.read_dotted_val_str(
                path_val=config_val, extension=extension
            )
        # prefer to use tuple instead of list for array-like items, but not guaranteed yet
        if type(config_val) is tuple or type(config_val) is list:
            return [
                self.read_dotted_val_str(path_val=entry, extension=extension)
                for entry in config_val
            ]
        if type(config_val) is dict:
            return {
                key: self.read_dotted_val_str(
                    path_val=val, extension=extension
                )
                for key, val in config_val.items()
            }

    def read_dotted_val_str(
        self, path_val: str, extension: str = ""
    ) -> str | list[str] | dict[str, str]:
        path_components = path_val.split("::")
        end_path = f"{path_components[-1]}{extension}"
        if len(path_components) == 1:
            return self._to_absolute_path(end_path)
        else:
            return self.read_path(
                config_key=f"{path_components[0]}", extension=f"/{end_path}"
            )


class ConfigModifier:
    def __init__(self, config_path: Path = None):
        if config_path is None:
            config_path = Path(__file__).parent / "config.toml"
        self._config_path = config_path

    def set(self, dotted_key: str, value: Any):
        with self._config_path.open(mode="r") as config_file:
            config = toml.load(config_file)

        keys = dotted_key.split(".")
        current_level = config
        for key in keys[:-1]:
            if key not in current_level:
                current_level[key] = {}
            current_level = current_level[key]

        current_level[keys[-1]] = value

        with self._config_path.open(mode="w") as config_file:
            toml.dump(config, config_file)

CONFIG_READER = ConfigReader()


PATH_CONFIG_READER = PathConfigReader()


CONFIG_MODIFIER = ConfigModifier()


if __name__ == "__main__":
    config_reader = ConfigReader()
    config_value = config_reader.get_config_value(
        config_key="preprocess.bg_data_cols"
    )
    pprint.pprint(config_value)
