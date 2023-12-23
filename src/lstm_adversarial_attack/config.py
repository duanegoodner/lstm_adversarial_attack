import json
import pprint
import toml
from pathlib import Path
from typing import Callable, Any


class ConfigReader:
    def __init__(self, config_path: Path = None):
        if config_path is None:
            config_path = Path(__file__).parent / "config.toml"
        self._config_path = config_path

    @property
    def project_root(self) -> Path:
        return Path(__file__).parent.parent.parent

    def _to_absolute_path(self, path_rel_project_root: str | Path) -> str:
        relative_path = Path(path_rel_project_root)
        if relative_path.is_absolute():
            raise TypeError(
                f"{path_rel_project_root} is an absolute path. "
                f"Must be a relative path.")
        return str(self.project_root / relative_path)

    def get_config_value(self, config_key: str) -> Any:
        with self._config_path.open(mode="r") as config_file:
            project_config = toml.load(config_file)

        result = project_config.copy()
        for sub_key in config_key.split("."):
            result = result[sub_key]

        return result

    def read_path(self, config_key: str) -> str | list[str]:
        path_rel_project_root = self.get_config_value(config_key=config_key)
        value_type = type(path_rel_project_root)
        assert value_type == str or value_type == list
        if value_type == str:
            return self._to_absolute_path(
                path_rel_project_root=path_rel_project_root)
        if value_type == list:
            return [self._to_absolute_path(path_rel_project_root=path) for path
                    in path_rel_project_root]


if __name__ == "__main__":
    config_reader = ConfigReader()
    config_value = config_reader.get_config_value(
        config_key="preprocess.bg_data_cols")
    pprint.pprint(config_value)

# class ModuleSettingsBuilder:
#     def __init__(self, sub_package_name: str, param_names: list[str],
#                  settings_constructor: Callable[..., dataclass],
#                  custom_config_path: Path = None
#                  ):
#         self._sub_package_name = sub_package_name
#         self._param_names = param_names
#         self._settings_constructor = settings_constructor
#         self._custom_config_path = custom_config_path
#
#     def build(self) -> dataclass:
#         config_reader = ConfigReader(config_path=self._custom_config_path)
#         module_config = config_reader.read_module_config(
#             sub_package_name=self._sub_package_name,
#             param_names=self._param_names
#         )
#         return self._settings_constructor(**module_config)
#
#
# @dataclass
# class NewPrefilterSettings:
#     """
#     Container for objects imported by Prefilter
#     """
#
#     min_age: int
#     min_los_hospital: int
#     min_los_icu: int
#     bg_data_cols: list[str]
#     lab_data_cols: list[str]
#     vital_data_cols: list[str]
#
#
# @dataclass
# class ModuleASettings:
#     param_a1: int
#     param_a2: int


# if __name__ == "__main__":
# settings_builder = ModuleSettingsBuilder(
#     sub_package_name="preprocess",
#     param_names=["min_age", "min_los_hospital", "min_los_icu",
#                  "bg_data_cols", "lab_data_cols", "vital_data_cols"],
#     settings_constructor=NewPrefilterSettings
# )
# settings = settings_builder.build()
# print(settings)
