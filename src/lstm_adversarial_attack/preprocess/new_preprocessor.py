from __future__ import annotations

import datetime

import msgspec
import numpy as np
import pandas as pd
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path, PosixPath
from typing import Any, Callable, TypeVar


sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.config_paths as cfp
import lstm_adversarial_attack.preprocess.preprocess_input_classes as pic
import lstm_adversarial_attack.preprocess.preprocess_resource as pr
import lstm_adversarial_attack.resource_io as rio


class NotExportable(Exception):
    def __init__(self, obj: object, msg: str = "Object is not exportable"):
        self._object = obj
        self._msg = msg

    def __str__(self):
        return f"{self._object} -> {self._msg}"


class NotImportable(Exception):
    def __init__(self, obj: object, msg: str = "Object is not importable"):
        self._object = obj
        self._msg = msg

    def __str__(self):
        return f"{self._object} -> {self._msg}"


_T = TypeVar("_T")


class IncomingPreprocessResource(ABC):
    def __init__(
        self,
        resource_id: str | Path,
        resource_pool: dict[str, OutgoingPreprocessResource] = None,
    ):
        if type(resource_id) == str:
            assert resource_pool is not None

        self._resource_id = resource_id
        self._resource_pool = resource_pool
        if type(self._resource_id) == str:
            self._item = self._resource_pool[self._resource_id].resource
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


class IncomingPreprocessResourceNoImport(IncomingPreprocessResource):
    def _import_object(self):
        raise NotImportable(self)


class IncomingPreprocessPickle(IncomingPreprocessResource):
    def _import_object(self) -> object:
        return rio.ResourceImporter().import_pickle_to_object(
            path=self.resource_id
        )


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
        raise NotExportable(self)


class OutgoingPreprocessPickle(OutgoingPreprocessResource):
    def export(self, path: Path):
        rio.ResourceExporter().export(resource=self._resource, path=path)


@dataclass
class NewPreprocessResource(ABC):
    @abstractmethod
    def export(self, output_dir: Path) -> list[pr.ExportedPreprocessResource]:
        pass


class NewFullAdmissionData(msgspec.Struct):
    """
    Container used as elements list build by FullAdmissionListBuilder
    """
    subject_id: int
    hadm_id: int
    icustay_id: int
    admittime: np.datetime64
    dischtime: np.datetime64
    hospital_expire_flag: int
    intime: np.datetime64
    outtime: np.datetime64
    time_series: pd.DataFrame


#  https://stackoverflow.com/a/65392400  (need this to work with dill)
NewFullAdmissionData.__module__ = __name__


class OutgoingFullAdmissionData(OutgoingPreprocessResource):
    def export(self, path: Path):
        rio.export_full_admission_data(obj=self._resource, path=path)


class OutgoingListOfArrays(OutgoingPreprocessResource):
    def export(self, path: Path):
        rio.list_of_np_to_json(resource=self.resource, path=path)


class JsonReadyOutput(OutgoingPreprocessResource):
    def export(self, path: Path):
        encoded_output = msgspec.json.encode(self.resource)
        with path.open(mode="wb") as out_file:
            out_file.write(encoded_output)


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
        if self.save_checkpoints:
            admission_list_builder_output["full_admission_list"].export(
                path=instantiated_admission_list_builder.output_dir
                / "full_admission_list.pickle"
            )
        self.available_resources["full_admission_list"] = (
            admission_list_builder_output["full_admission_list"]
        )

    def run_feature_builder(
        self,
        feature_builder_resources: dict[
            str, IncomingPreprocessResourceNoImport | IncomingFeatherDataFrame
        ],
    ):
        instantiated_feature_builder = self.feature_builder(
            resources=feature_builder_resources
        )
        feature_builder_output = instantiated_feature_builder.process()
        if self.save_checkpoints:
            feature_builder_output["processed_admission_list"].export(
                path=instantiated_feature_builder.output_dir
                / "processed_admission_list.pickle"
            )
        self.available_resources["processed_admission_list"] = (
            feature_builder_output["processed_admission_list"]
        )

    def run_feature_finalizer(
        self, feature_finalizer_resources: dict[str, IncomingPreprocessPickle]
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
        self.run_feature_builder(
            feature_builder_resources={
                "full_admission_list": IncomingPreprocessResourceNoImport(
                    resource_id="full_admission_list",
                    resource_pool=self.available_resources,
                ),
                "bg_lab_vital_summary_stats": IncomingFeatherDataFrame(
                    resource_id=cfp.STAY_MEASUREMENT_OUTPUT
                    / "bg_lab_vital_summary_stats.feather"
                ),
            }
        )

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
