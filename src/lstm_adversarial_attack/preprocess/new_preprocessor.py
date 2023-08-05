from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path, PosixPath
from typing import Any, Callable, TypeVar


sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.preprocess.preprocess_input_classes as pic
import lstm_adversarial_attack.preprocess.preprocess_resource as pr
import lstm_adversarial_attack.resource_io as rio


class NotExportable(Exception):
    def __init__(self, obj: object, msg: str = "Object is not exportable"):
        self._object = obj
        self._msg = msg

    def __str__(self):
        return f"{self._object} -> {self._msg}"


_T = TypeVar("_T")


class IncomingPreprocessResource(ABC):
    def __init__(
        self,
        resource_id: str | Path,
        resource_pool: dict[str, Any] = None,
    ):
        if type(resource_id) == str:
            assert resource_pool is not None

        self._resource_id = resource_id
        self._resource_pool = resource_pool
        if type(self._resource_id) == str:
            self._item = self._resource_pool[self._resource_id]
        else:
            self._item = self._import_object()

    def _import_object(self) -> _T:
        pass

    @property
    def item(self) -> _T:
        return self._item

    @property
    def resource_id(self) -> str | Path:
        return self._resource_id

    @property
    def resource_pool(self) -> dict[str, _T | Any] | None:
        return self._resource_pool


class IncomingFeatherDataFrame(IncomingPreprocessResource):
    def _import_object(self) -> pd.DataFrame:
        return rio.feather_to_df(path=self.resource_id)


class IncomingCSVDataFrame(IncomingPreprocessResource):
    def _import_object(self) -> pd.DataFrame:
        return pd.read_csv(filepath_or_buffer=self.resource_id)


class OutgoingPreprocessResource(ABC):
    def __init__(self, resource: Any):
        self._resource = resource

    @property
    def resource(self) -> Any:
        return self._resource

    def export(self, path: Path):
        pass


class OutgoingPreprocessDataFrame(OutgoingPreprocessResource):
    def export(self, path: Path):
        rio.df_to_feather(df=self._resource, path=path)


class OutgoingPreprocessResourceNoExport(OutgoingPreprocessResource):
    def export(self, path: Path):
        raise NotExportable


@dataclass
class NewPreprocessResource(ABC):
    @abstractmethod
    def export(self, output_dir: Path) -> list[pr.ExportedPreprocessResource]:
        pass


@dataclass
class PureDataFrameContainer(NewPreprocessResource, ABC):
    def export(self, output_dir: Path) -> list[pr.ExportedPreprocessResource]:
        exported_resources = []

        for key, df in self.__dict__.items():
            rio.df_to_feather(df=df, path=output_dir / f"{key}.feather")
            exported_resources.append(
                pr.ExportedPreprocessResource(
                    path=output_dir, data_type=type(df).__name__
                )
            )
        return exported_resources


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
class NewAdmissionListBuilderResources(PureDataFrameContainer):
    icustay_bg_lab_vital: pd.DataFrame


@dataclass
class NewAdmissionListBuilderOutput(NewPreprocessResource):
    admission_list: list[NewFullAdmissionData]

    def export(self, output_dir: Path) -> list[pr.ExportedPreprocessResource]:
        pass


@dataclass
class NewFeatureBuilderResources(NewPreprocessResource):
    admission_list: list[NewFullAdmissionData]
    bg_lab_vital_summary_stats: pd.DataFrame

    def export(self, output_dir: Path) -> list[pr.ExportedPreprocessResource]:
        pass


@dataclass
class NewFeatureBuilderOutput(NewPreprocessResource):
    processed_admission_list: list[NewFullAdmissionData]

    def export(self, output_dir: Path) -> list[pr.ExportedPreprocessResource]:
        pass


@dataclass
class NewFeatureFinalizerResources(NewPreprocessResource):
    processed_admission_list: list[NewFullAdmissionData]

    def export(self, output_dir: Path) -> list[pr.ExportedPreprocessResource]:
        pass


@dataclass
class NewFeatureFinalizerOutput(NewPreprocessResource):
    measurement_col_names: tuple[str]
    measurement_data_list: list[np.ndarray]
    in_hospital_mortality_list: list[int]

    def export(self, output_dir: Path) -> list[pr.ExportedPreprocessResource]:
        pass


class NewPreprocessModule(ABC):
    def __init__(
        self,
        resources: dict[str, IncomingPreprocessResource],
        output_dir: Path,
        settings: dataclass,
    ):
        self._resources = resources
        self._output_dir = output_dir
        self._settings = settings
        self._resource_items = {
            key: value.item for key, value in self._resources.items()
        }

    @property
    def resources(self) -> dict[str, Any]:
        return self._resources

    @property
    def settings(self) -> dataclass:
        return self._settings

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    @property
    def resource_items(self) -> dict[str, Any]:
        return self._resource_items

    @abstractmethod
    def process(
        self,
    ) -> NewPreprocessResource | dict[str, OutgoingPreprocessResource]:
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

    def run_prefilter(
        self, prefilter_resources: dict[str, IncomingCSVDataFrame] = None
    ):
        instantiated_prefilter = self.prefilter(resources=prefilter_resources)
        prefilter_output = instantiated_prefilter.process()
        if self.save_checkpoints:
            for key, outgoing_resource in prefilter_output.items():
                outgoing_resource.export(
                    path=instantiated_prefilter.output_dir / f"{key}.feather"
                )
            # prefilter_output.export(
            #     output_dir=instantiated_prefilter.output_dir
            # )
        self.available_resources["prefiltered_bg"] = prefilter_output[
            "prefiltered_bg"
        ]
        self.available_resources["prefiltered_icustay"] = prefilter_output[
            "prefiltered_icustay"
        ]
        self.available_resources["prefiltered_lab"] = prefilter_output[
            "prefiltered_lab"
        ]
        self.available_resources["prefiltered_vital"] = prefilter_output[
            "prefiltered_vital"
        ]
        # self.available_resources["prefilter_output"] = prefilter_output
        # return prefilter_output

    def run_icustay_measurement_combiner(
        self,
        icustay_measurement_combiner_resources: dict[
            str, IncomingFeatherDataFrame
        ] = None,
    ):
        instantiated_measurement_combiner = self.icustay_measurement_combiner(
            resources=icustay_measurement_combiner_resources
        )
        measurement_combiner_output = (
            instantiated_measurement_combiner.process()
        )
        if self.save_checkpoints:
            for key, outgoing_resource in measurement_combiner_output.items():
                outgoing_resource.export(
                    path=instantiated_measurement_combiner.output_dir
                    / f"{key}.feather"
                )
            # measurement_combiner_output.export(
            #     output_dir=instantiated_measurement_combiner.output_dir
            # )
        self.available_resources["bg_lab_vital_summary_stats"] = (
            measurement_combiner_output["bg_lab_vital_summary_stats"]
        )
        self.available_resources["icustay_bg_lab_vital"] = (
            measurement_combiner_output["icustay_bg_lab_vital"]
        )

    def run_admission_list_builder(
        self,
        admission_list_builder_resources: dict[
            str, IncomingFeatherDataFrame
        ] = None,
    ):
        instantiated_admission_list_builder = self.admission_list_builder(
            resources=admission_list_builder_resources
        )
        admission_list_builder_output = (
            instantiated_admission_list_builder.process()
        )
        # Don't give ability to export b/c do not have non-pickle option
        # if self.save_checkpoints:
        #     admission_list_builder_output.export(
        #         output_dir=instantiated_admission_list_builder.output_dir
        #     )
        self.available_resources["full_admission_list"] = (
            admission_list_builder_output["full_admission_list"]
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
        # prefilter_resources = self.get_prefilter_resources()
        self.run_prefilter()
        self.run_icustay_measurement_combiner()
        self.run_admission_list_builder()
        #
        # feature_builder_resources = NewFeatureBuilderResources(
        #     admission_list=self.available_resources["admission_list"],
        #     bg_lab_vital_summary_stats=self.available_resources[
        #         "bg_lab_vital_summary_stats"
        #     ],
        # )
        # self.run_feature_builder(
        #     feature_builder_resources=feature_builder_resources
        # )
        #
        # feature_finalizer_resources = NewFeatureFinalizerResources(
        #     processed_admission_list=self.available_resources[
        #         "processed_admission_list"
        #     ]
        # )
        # self.run_feature_finalizer(
        #     feature_finalizer_resources=feature_finalizer_resources
        # )

        return self.available_resources
