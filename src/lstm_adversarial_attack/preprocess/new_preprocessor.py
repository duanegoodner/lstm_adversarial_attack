from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

sys.path.append(str(Path(__file__).parent.parent.parent))

import lstm_adversarial_attack.config_paths as cfp
import lstm_adversarial_attack.preprocess.preprocess_input_classes as pic
import lstm_adversarial_attack.preprocess.resource_data_structs as rds


class NewPreprocessModule(ABC):
    def __init__(
        self,
        resources: dataclass,
        output_dir: Path,
        settings: dataclass,
    ):
        self._resources = resources
        self._output_dir = output_dir
        self._settings = settings

    @property
    def settings(self) -> dataclass:
        return self._settings

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    @abstractmethod
    def process(
        self,
    ) -> dict[str, rds.OutgoingPreprocessResource]:
        pass


class NewPreprocessor:
    def __init__(
        self,
        prefilter: Callable[..., NewPreprocessModule],
        icustay_measurement_combiner: Callable[..., NewPreprocessModule],
        admission_list_builder: Callable[..., NewPreprocessModule],
        feature_builder: Callable[..., NewPreprocessModule],
        feature_finalizer: Callable[..., NewPreprocessModule],
        inputs: pic.PrefilterResourceRefs = None,
        save_checkpoints: bool = False,
        available_resources: dict[str, Any] = None,
    ):
        self.prefilter = prefilter
        self.icustay_measurement_combiner = icustay_measurement_combiner
        self.admission_list_builder = admission_list_builder
        self.feature_builder = feature_builder
        self.feature_finalizer = feature_finalizer
        if inputs is None:
            inputs = pic.PrefilterResourceRefs()
        self.inputs = inputs
        self.save_checkpoints = save_checkpoints
        if available_resources is None:
            available_resources = {}
        self.available_resources = available_resources

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
        self,
        module_constructor: Callable[..., NewPreprocessModule],
        module_resources: dataclass,
    ):
        print(f"Running {module_constructor.__name__}")
        module = module_constructor(resources=module_resources)
        module_output = module.process()
        if self.save_checkpoints:
            self.export_resources(
                module_output=module_output, output_dir=module.output_dir
            )
        self.available_resources.update(module_output)

    def preprocess(self):
        prefilter_resources = rds.NewPrefilterResources(
            icustay=rds.IncomingCSVDataFrame(
                resource_id=cfp.PREFILTER_INPUT_FILES["icustay"]
            ),
            bg=rds.IncomingCSVDataFrame(
                resource_id=cfp.PREFILTER_INPUT_FILES["bg"]
            ),
            vital=rds.IncomingCSVDataFrame(
                resource_id=cfp.PREFILTER_INPUT_FILES["vital"]
            ),
            lab=rds.IncomingCSVDataFrame(
                resource_id=cfp.PREFILTER_INPUT_FILES["lab"]
            ),
        )
        self.run_preprocess_module(
            module_constructor=self.prefilter,
            module_resources=prefilter_resources,
        )

        combiner_resources = rds.NewICUStayMeasurementMergerResources(
            prefiltered_icustay=rds.IncomingFeatherDataFrame(
                resource_id="prefiltered_icustay",
                resource_pool=self.available_resources,
            ),
            prefiltered_bg=rds.IncomingFeatherDataFrame(
                resource_id="prefiltered_bg",
                resource_pool=self.available_resources,
            ),
            prefiltered_lab=rds.IncomingFeatherDataFrame(
                resource_id="prefiltered_lab",
                resource_pool=self.available_resources,
            ),
            prefiltered_vital=rds.IncomingFeatherDataFrame(
                resource_id="prefiltered_vital",
                resource_pool=self.available_resources,
            ),
        )
        self.run_preprocess_module(
            module_constructor=self.icustay_measurement_combiner,
            module_resources=combiner_resources,
        )

        list_builder_resources = rds.NewAdmissionListBuilderResources(
            icustay_bg_lab_vital=rds.IncomingFeatherDataFrame(
                resource_id="icustay_bg_lab_vital",
                resource_pool=self.available_resources,
            )
        )
        self.run_preprocess_module(
            module_constructor=self.admission_list_builder,
            module_resources=list_builder_resources
        )

        feature_builder_resources = rds.NewFeatureBuilderResources(
            full_admission_list=rds.IncomingFullAdmissionData(
                resource_id="full_admission_list",
                resource_pool=self.available_resources,
            ),
            bg_lab_vital_summary_stats=rds.IncomingFeatherDataFrame(
                resource_id="bg_lab_vital_summary_stats",
                resource_pool=self.available_resources,
            ),
        )
        self.run_preprocess_module(
            module_constructor=self.feature_builder,
            module_resources=feature_builder_resources
        )

        feature_finalizer_resources = rds.NewFeatureFinalizerResources(
            processed_admission_list=rds.IncomingFullAdmissionData(
                resource_id="processed_admission_list",
                resource_pool=self.available_resources,
            )
        )
        self.run_preprocess_module(
            module_constructor=self.feature_finalizer,
            module_resources=feature_finalizer_resources
        )

        return self.available_resources
