import numpy as np
import pandas as pd
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.preprocess.preprocess_input_classes as pic
import lstm_adversarial_attack.preprocess.preprocess_resource as pr
import lstm_adversarial_attack.resource_io as rio


@dataclass
class NewPreprocessResource(ABC):
    def export(self, output_dir: Path):
        exported_resources = []

        for key, val in self.__dict__.items():
            rio.export_to_pickle(
                resource=val, path=output_dir / f"{key}.pickle"
            )
            exported_resources.append(
                pr.ExportedPreprocessResource(
                    path=output_dir, data_type=type(val).__name__
                )
            )


@dataclass
class NewPrefilterResources(NewPreprocessResource):
    icustay: pd.DataFrame
    bg: pd.DataFrame
    vital: pd.DataFrame
    lab: pd.DataFrame


@dataclass
class NewPrefilterOutput(NewPreprocessResource):
    icustay: pd.DataFrame
    bg: pd.DataFrame
    vital: pd.DataFrame
    lab: pd.DataFrame


@dataclass
class NewICUStayMeasurementMergerResources(NewPreprocessResource):
    icustay: pd.DataFrame
    bg: pd.DataFrame
    lab: pd.DataFrame
    vital: pd.DataFrame


@dataclass
class NewICUStayMeasurementMergerOutput(NewPreprocessResource):
    icustay_bg_lab_vital: pd.DataFrame
    bg_lab_vital_summary_stats: pd.DataFrame


@dataclass
class NewFullAdmissionData:
    """
    Container used as elements list build by FullAdmissionListBuilder
    """

    subject_id: np.ndarray
    hadm_id: np.ndarray
    icustay_id: np.ndarray
    admittime: np.ndarray
    dischtime: np.ndarray
    hospital_expire_flag: np.ndarray
    intime: np.ndarray
    outtime: np.ndarray
    time_series: pd.DataFrame


#  https://stackoverflow.com/a/65392400  (need this to work with dill)
NewFullAdmissionData.__module__ = __name__


@dataclass
class NewAdmissionListBuilderResources(NewPreprocessResource):
    icustay_bg_lab_vital: pd.DataFrame


@dataclass
class NewAdmissionListBuilderOutput(NewPreprocessResource):
    admission_list: list[NewFullAdmissionData]


@dataclass
class NewFeatureBuilderResources(NewPreprocessResource):
    admission_list: list[NewFullAdmissionData]
    bg_lab_vital_summary_stats: pd.DataFrame


@dataclass
class NewFeatureBuilderOutput(NewPreprocessResource):
    processed_admission_list: list[NewFullAdmissionData]


@dataclass
class NewFeatureFinalizerResources(NewPreprocessResource):
    processed_admission_list: list[NewFullAdmissionData]


@dataclass
class NewFeatureFinalizerOutput(NewPreprocessResource):
    measurement_col_names: tuple[str]
    measurement_data_list: list[np.ndarray]
    in_hospital_mortality_list: list[int]


class NewPreprocessModule(ABC):
    def __init__(
        self,
        resources: NewPreprocessResource,
        output_dir: Path,
        settings: dataclass
    ):
        self.resources = resources
        self._output_dir = output_dir
        self._settings = settings

    @property
    def settings(self) -> dataclass:
        return self._settings

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    @abstractmethod
    def process(self) -> NewPreprocessResource:
        pass


# class AbstractPrefilter(ABC):
#     @abstractmethod
#     def process(self) -> NewPrefilterOutput:
#         pass
#
#     @abstractmethod
#     def output_dir(self) -> Path:
#         pass
#
#
# class AbstractICUMeasurementCombiner(ABC):
#     @abstractmethod
#     def process(self) -> NewICUStayMeasurementMergerOutput:
#         pass
#
#     @abstractmethod
#     def output_dir(self) -> Path:
#         pass
#
#
# class AbstractAdmissionListBuilder(ABC):
#     @abstractmethod
#     def process(self) -> NewAdmissionListBuilderOutput:
#         pass
#
#     @abstractmethod
#     def output_dir(self) -> Path:
#         pass
#
#
# class AbstractFeatureBuilder(ABC):
#     @abstractmethod
#     def process(self) -> NewFeatureBuilderOutput:
#         pass
#
#     @abstractmethod
#     def output_dir(self) -> Path:
#         pass
#
#
# class AbstractFeatureFinalizer(ABC):
#     @abstractmethod
#     def process(self) -> NewFeatureFinalizerOutput:
#         pass
#
#     @abstractmethod
#     def output_dir(self) -> Path:
#         pass


class NewPreprocessor:
    def __init__(
        self,
        prefilter: Callable[..., NewPreprocessModule],
        icustay_measurement_combiner: Callable[
            ..., NewPreprocessModule
        ],
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

    def get_prefilter_resources(self) -> NewPrefilterResources:
        prefilter_resources = NewPrefilterResources(
            icustay=pd.read_csv(self.inputs.icustay),
            bg=pd.read_csv(self.inputs.bg),
            vital=pd.read_csv(self.inputs.vital),
            lab=pd.read_csv(self.inputs.lab),
        )
        return prefilter_resources

    def run_prefilter(self, prefilter_resources: NewPrefilterResources):
        instantiated_prefilter = self.prefilter(resources=prefilter_resources)
        prefilter_output = instantiated_prefilter.process()
        if self.save_checkpoints:
            prefilter_output.export(
                output_dir=instantiated_prefilter.output_dir
            )
        self.available_resources["prefiltered_bg"] = prefilter_output.bg
        self.available_resources["prefiltered_icustay"] = (
            prefilter_output.icustay
        )
        self.available_resources["prefiltered_lab"] = prefilter_output.lab
        self.available_resources["prefiltered_vital"] = prefilter_output.vital
        # self.available_resources["prefilter_output"] = prefilter_output
        # return prefilter_output

    def run_icustay_measurement_combiner(
        self,
        icustay_measurement_combiner_resources: NewICUStayMeasurementMergerResources,
    ):
        instantiated_measurement_combiner = self.icustay_measurement_combiner(
            resources=icustay_measurement_combiner_resources
        )
        measurement_combiner_output = (
            instantiated_measurement_combiner.process()
        )
        if self.save_checkpoints:
            measurement_combiner_output.export(
                output_dir=instantiated_measurement_combiner.output_dir
            )
        self.available_resources["bg_lab_vital_summary_stats"] = (
            measurement_combiner_output.bg_lab_vital_summary_stats
        )
        self.available_resources["icustay_bg_lab_vital"] = (
            measurement_combiner_output.icustay_bg_lab_vital
        )

    def run_admission_list_builder(
        self,
        admission_list_builder_resources: NewAdmissionListBuilderResources,
    ):
        instantiated_admission_list_builder = self.admission_list_builder(
            resources=admission_list_builder_resources
        )
        admission_list_builder_output = (
            instantiated_admission_list_builder.process()
        )
        if self.save_checkpoints:
            admission_list_builder_output.export(
                output_dir=instantiated_admission_list_builder.output_dir
            )
        self.available_resources["admission_list"] = (
            admission_list_builder_output.admission_list
        )

    def run_feature_builder(
        self, feature_builder_resources: NewFeatureBuilderResources
    ):
        instantiated_feature_builder = self.feature_builder(
            resources=feature_builder_resources
        )
        feature_builder_output = instantiated_feature_builder.process()
        if self.save_checkpoints:
            feature_builder_output.export(
                output_dir=instantiated_feature_builder.output_dir
            )
        self.available_resources["processed_admission_list"] = (
            feature_builder_output.processed_admission_list
        )

    def run_feature_finalizer(
        self, feature_finalizer_resources: NewFeatureFinalizerResources
    ):
        instantiated_feature_finalizer = self.feature_finalizer(
            resources=feature_finalizer_resources
        )
        feature_finalizer_output = instantiated_feature_finalizer.process()
        if self.save_checkpoints:
            feature_finalizer_output.export(
                output_dir=instantiated_feature_finalizer.output_dir
            )
        self.available_resources["feature_finalizer_output"] = (
            feature_finalizer_output
        )
        self.available_resources["in_hospital_mortality_list"] = (
            feature_finalizer_output.in_hospital_mortality_list
        )
        self.available_resources["measurement_col_names"] = (
            feature_finalizer_output.measurement_col_names
        )
        self.available_resources["measurement_data_list"] = (
            feature_finalizer_output.measurement_data_list
        )

    def preprocess(self):
        prefilter_resources = self.get_prefilter_resources()
        self.run_prefilter(prefilter_resources=prefilter_resources)

        measurement_combiner_resources = NewICUStayMeasurementMergerResources(
            # **self.available_resources["prefilter_output"].__dict__
            bg=self.available_resources["prefiltered_bg"],
            icustay=self.available_resources["prefiltered_icustay"],
            lab=self.available_resources["prefiltered_lab"],
            vital=self.available_resources["prefiltered_vital"],
        )
        self.run_icustay_measurement_combiner(
            icustay_measurement_combiner_resources=measurement_combiner_resources
        )

        admission_list_builder_resources = NewAdmissionListBuilderResources(
            icustay_bg_lab_vital=self.available_resources[
                "icustay_bg_lab_vital"
            ]
        )
        self.run_admission_list_builder(
            admission_list_builder_resources=admission_list_builder_resources
        )
        #
        feature_builder_resources = NewFeatureBuilderResources(
            admission_list=self.available_resources["admission_list"],
            bg_lab_vital_summary_stats=self.available_resources[
                "bg_lab_vital_summary_stats"
            ],
        )
        self.run_feature_builder(
            feature_builder_resources=feature_builder_resources
        )

        feature_finalizer_resources = NewFeatureFinalizerResources(
            processed_admission_list=self.available_resources[
                "processed_admission_list"
            ]
        )
        self.run_feature_finalizer(
            feature_finalizer_resources=feature_finalizer_resources
        )

        return self.available_resources
