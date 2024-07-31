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

    @property
    def project_root(self) -> Path:
        return Path(__file__).parent.parent.parent

    def get_config_value(self, config_key: str) -> Any:
        result = self._config.copy()
        for sub_key in config_key.split("."):
            if result.get(sub_key) is None:
                return None
            else:
                result = result.get(sub_key)

        # We want arrays in config file to be retrieved as tuples (immutable), not lists
        # if type(result) is list:
        #     result = tuple(result)

        return result

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
            return self.read_dotted_val_str(path_val=config_val, extension=extension)
        # prefer to use tuple instead of list for array-like items, but not guaranteed yet
        if type(config_val) is tuple or type(config_val) is list:
            return [
                self.read_dotted_val_str(path_val=entry, extension=extension)
                for entry in config_val
            ]
        if type(config_val) is dict:
            return {
                key: self.read_dotted_val_str(path_val=val, extension=extension)
                for key, val in config_val.items()
            }

    def read_dotted_val_str(
        self, path_val: str, extension: str = ""
    ) -> str | list[str] | dict[str, str]:
        # path_val = self.get_config_value(config_key=config_key)
        path_components = path_val.split("::")
        end_path = f"{path_components[-1]}{extension}"
        if len(path_components) == 1:
            return self._to_absolute_path(end_path)
        else:
            return self.read_path(
                config_key=f"{path_components[0]}", extension=f"/{end_path}"
            )


CONFIG_READER = ConfigReader()


if __name__ == "__main__":
    config_reader = ConfigReader()
    config_value = config_reader.get_config_value(config_key="preprocess.bg_data_cols")
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
