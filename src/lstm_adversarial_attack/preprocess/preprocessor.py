from __future__ import annotations

import sys
import time
from abc import ABC, abstractmethod
from dataclasses import Field, dataclass, fields
from pathlib import Path
from typing import Any, Callable

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.preprocess.encode_decode as edc
import lstm_adversarial_attack.preprocess.encode_decode_structs as eds
import lstm_adversarial_attack.preprocess.resource_data_structs as rds
from lstm_adversarial_attack.config import CONFIG_READER


@dataclass
class PreprocessModuleResources(ABC):
    module_name: str

    def __post_init__(self):
        # config_reader = config.ConfigReader()
        for object_field in fields(self):
            if (
                object_field.name != "module_name"
                and getattr(self, object_field.name) is None
            ):
                value = CONFIG_READER.get_config_value(
                    f"preprocess.{self.module_name}.resources.{object_field.name}"
                )


@dataclass
class PreprocessModuleSettings(ABC):
    preprocess_id: str
    module_name: str
    output_dir: str | Path = None

    @property
    def standard_fields(self) -> tuple[Field, ...]:
        return fields(PreprocessModuleSettings)

    @property
    def module_specific_fields(self) -> tuple[Field, ...]:
        return tuple(
            [
                object_field
                for object_field in fields(self)
                if object_field not in self.standard_fields
            ]
        )

    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = (
                Path(
                    CONFIG_READER.read_path(
                        config_key="preprocess.output_root"
                    )
                )
                / self.preprocess_id
                / CONFIG_READER.get_config_value(
                    f"preprocess.output_dir_names.{self.module_name}"
                )
            )

        for object_field in self.module_specific_fields:
            if getattr(self, object_field.name) is None:
                attr = CONFIG_READER.get_config_value(
                    f"preprocess.{object_field.name}"
                )
                setattr(self, object_field.name, attr)


@dataclass
class PreprocessModule(ABC):
    def __init__(
        self,
        resources: dataclass,
        settings: PreprocessModuleSettings,
        output_constructors: dataclass,
    ):
        self._resources = resources
        self._settings = settings
        self._output_constructors = output_constructors

        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def settings(self) -> dataclass:
        return self._settings

    @property
    def output_dir(self) -> Path:
        return Path(self.settings.output_dir)

    @property
    def output_constructors(self) -> dataclass:
        return self._output_constructors

    @abstractmethod
    def process(
        self,
    ) -> dict[str, rds.OutgoingPreprocessResource]:
        pass

    @property
    def summary(self) -> eds.PreprocessModuleSummary:
        return eds.PreprocessModuleSummary(
            output_dir=str(self.output_dir),
            output_constructors={
                key: val.__name__
                for key, val in self._output_constructors.__dict__.items()
            },
            resources={
                key: {
                    "resource_name": val.__class__.__name__,
                    "resource_id": str(val.resource_id),
                }
                for key, val in self._resources.resources_dict.items()
            },
            settings=self._settings.__dict__,
        )

    def save_summary(self):
        edc.PreprocessModuleSummaryWriter().export(
            obj=self.summary,
            path=self.output_dir / f"{self.__class__.__name__}_summary.json",
        )


@dataclass
class ModuleInfo:
    resource_collection_ids: dict[str, str]
    module_name: str
    module_constructor: Callable[..., PreprocessModule]
    resources_constructor: Callable[..., dataclass]
    settings_constructor: Callable[..., PreprocessModuleSettings]
    # resources_info: list[rds.ResourceInfoNew]
    default_data_source_type: rds.DataSourceType
    output_constructors: dataclass = None
    output_dir: Path = None
    save_output: bool = True

    def build_module(
        self, resource_pool: dict[str, rds.OutgoingPreprocessResource]
    ) -> PreprocessModule:
        return self.module_constructor(
            resources=self.resources_constructor(
                collection_ids=self.resource_collection_ids,
                module_name=self.module_name,
                default_data_source_type=self.default_data_source_type,
                resource_pool=resource_pool,
            ),
            settings=self.settings_constructor(
                module_name=self.module_name,
                preprocess_id=self.resource_collection_ids["preprocess"],
            ),
            output_constructors=self.output_constructors,
        )


class Preprocessor:
    def __init__(
        self,
        preprocess_id: str,
        modules_info: list[ModuleInfo],
        save_checkpoints: bool = False,
        available_resources: dict[str, Any] = None,
    ):
        # TODO Consider making run_output_root a data member
        self.preprocess_id = preprocess_id
        run_output_root = (
            Path(CONFIG_READER.read_path(config_key="preprocess.output_root"))
            / preprocess_id
        )
        run_output_root.mkdir(parents=True, exist_ok=True)

        self.modules_info = modules_info
        if available_resources is None:
            available_resources = {}
        self.available_resources = available_resources

        self.module_resources = []
        self.save_checkpoints = save_checkpoints

    @staticmethod
    def export_resources(
        module_output: dict[str, rds.OutgoingPreprocessResource],
        output_dir: Path,
    ):
        for key, outgoing_resource in module_output.items():
            outgoing_resource.export(
                path=output_dir / f"{key}{outgoing_resource.file_ext}"
            )

    def run_preprocess_module(
        self, module: PreprocessModule, save_output: bool
    ):
        process_start = time.time()
        module_output = module.process()
        process_end = time.time()
        print(
            f"{module.__class__.__name__} process time ="
            f" {process_end - process_start}"
        )
        if save_output:
            export_start = time.time()
            self.export_resources(
                module_output=module_output, output_dir=module.output_dir
            )
            module.save_summary()
            export_end = time.time()
            print(
                f"{module.__class__.__name__} export time ="
                f" {export_end - export_start}"
            )
            print(f"Output saved in {str(module.output_dir)}\n")
        self.available_resources.update(module_output)

    def run_all_modules(self):
        print(f"Starting preprocess session {self.preprocess_id}\n")

        for module_info in self.modules_info:
            print(f"Running {module_info.module_constructor.__name__}")
            init_start = time.time()
            module = module_info.build_module(
                resource_pool=self.available_resources
            )
            init_end = time.time()
            print(
                f"{module.__class__.__name__} init time ="
                f" {init_end - init_start}"
            )
            self.run_preprocess_module(
                module=module,
                save_output=self.save_checkpoints or module_info.save_output,
            )


        return self.available_resources
