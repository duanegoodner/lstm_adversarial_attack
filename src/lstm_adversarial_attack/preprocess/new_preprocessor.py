from __future__ import annotations

import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.preprocess.resource_data_structs as rds


class NewPreprocessModule(ABC):
    def __init__(
        self,
        resources: dataclass,
        output_dir: Path,
        settings: dataclass,
        output_info: dataclass,
    ):
        self._resources = resources
        self._output_dir = output_dir
        self._settings = settings
        self._output_info = output_info

    @property
    def settings(self) -> dataclass:
        return self._settings

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    @property
    def output_info(self) -> dataclass:
        return self._output_info

    @abstractmethod
    def process(
        self,
    ) -> dict[str, rds.OutgoingPreprocessResource]:
        pass


@dataclass
class ModuleInfo:
    module_constructor: Callable[..., NewPreprocessModule]
    resources_constructor: Callable[..., dataclass]
    individual_resources_info: list[rds.SingleResourceInfo]
    output_info: dataclass = None
    output_dir: Path = None
    settings: dataclass = None

    def build_module(
        self, resource_pool: dict[str, rds.OutgoingPreprocessResource]
    ) -> NewPreprocessModule:
        module_resources = {}
        for item in self.individual_resources_info:
            module_resources.update(
                item.build_resource(resource_pool=resource_pool)
            )
        resources = self.resources_constructor(**module_resources)
        return self.module_constructor(
            resources=resources,
            output_dir=self.output_dir,
            settings=self.settings,
            output_info=self.output_info,
        )


class NewPreprocessor:
    def __init__(
        self,
        modules_info: list[ModuleInfo],
        save_checkpoints: bool = False,
        available_resources: dict[str, Any] = None,
    ):
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
        self, module: NewPreprocessModule, save_output: bool
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
            export_end = time.time()
            print(
                f"{module.__class__.__name__} export time ="
                f" {export_end - export_start}\n"
            )
        self.available_resources.update(module_output)

    def run_all_modules(self):
        for idx, module_info in enumerate(self.modules_info):
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
                save_output=self.save_checkpoints
                or (idx == len(self.modules_info) - 1),
            )

        return self.available_resources
