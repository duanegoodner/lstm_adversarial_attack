from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, TypeVar

import pandas as pd
import lstm_adversarial_attack.config as config
import lstm_adversarial_attack.config_paths as cfp
import lstm_adversarial_attack.preprocess.encode_decode as edc
import lstm_adversarial_attack.preprocess.encode_decode_structs as eds
import lstm_adversarial_attack.resource_io as rio

_T = TypeVar("_T")


class OutgoingPreprocessResource(ABC):
    def __init__(self, resource: Any):
        self._resource = resource

    @property
    def resource(self) -> Any:
        return self._resource

    @property
    @abstractmethod
    def file_ext(self) -> str:
        pass

    @abstractmethod
    def export(self, path: Path):
        pass


class IncomingPreprocessResource(ABC):
    """
    Base class for resource used by a PreprocessModule.
    """

    def __init__(
        self,
        resource_id: str | Path,
        resource_pool: dict[str, OutgoingPreprocessResource] = None,
    ):
        """
        Initializes resource by assigning an object to self._item. If
        resource_id is a Path, object is obtained by importing from file. If
        resource_id is a string, resource pool must not be None, and resource_id
        must equal the key of an entry in resource_pool.
        :param resource_id:
        :param resource_pool:
        """
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


@dataclass
class SingleResourceInfo(ABC):
    @abstractmethod
    def build_resource(self, **kwargs):
        pass


@dataclass
class PoolResourceInfo(SingleResourceInfo):
    key: str
    constructor: Callable[..., IncomingPreprocessResource]

    def build_resource(
        self, resource_pool: dict[str, OutgoingPreprocessResource]
    ) -> dict[str, IncomingPreprocessResource]:
        return {
            self.key: self.constructor(
                resource_id=self.key, resource_pool=resource_pool
            )
        }


@dataclass
class FileResourceInfo(SingleResourceInfo):
    key: str
    path: Path
    constructor: Callable[..., IncomingPreprocessResource]

    def build_resource(
        self, **kwargs
    ) -> dict[str, IncomingPreprocessResource]:
        return {self.key: self.constructor(resource_id=self.path)}


@dataclass
class SingleOutputInfo:
    key: str
    constructor: Callable[..., OutgoingPreprocessResource]


class IncomingFeatherDataFrame(IncomingPreprocessResource):
    def _import_object(self) -> pd.DataFrame:
        return rio.feather_to_df(path=self.resource_id)


class IncomingCSVDataFrame(IncomingPreprocessResource):
    def _import_object(self) -> pd.DataFrame:
        return pd.read_csv(filepath_or_buffer=self.resource_id)


class IncomingFullAdmissionData(IncomingPreprocessResource):
    def _import_object(self) -> list[eds.FullAdmissionData]:
        return edc.import_admission_data_list(path=self.resource_id)


class OutgoingPreprocessDataFrame(OutgoingPreprocessResource):
    def export(self, path: Path):
        rio.df_to_feather(df=self._resource, path=path)

    @property
    def file_ext(self) -> str:
        return ".feather"


class OutgoingPreprocessPickle(OutgoingPreprocessResource):
    def export(self, path: Path):
        rio.ResourceExporter().export(resource=self._resource, path=path)

    @property
    def file_ext(self) -> str:
        return ".pickle"


class OutgoingFullAdmissionData(OutgoingPreprocessResource):
    def export(self, path: Path):
        edc.export_admission_data_list(
            admission_data_list=self._resource, path=path
        )

    @property
    def file_ext(self) -> str:
        return ".json"


class OutgoingFeaturesList(OutgoingPreprocessResource):
    def export(self, path: Path):
        edc.AdmissionDataWriter().export(obj=self.resource, path=path)
        # edc.export_feature_arrays(np_arrays=self.resource, path=path)

    @property
    def file_ext(self) -> str:
        return ".json"


class JsonReadyOutput(OutgoingPreprocessResource):
    def export(self, path: Path):
        edc.export_json_ready_object(obj=self.resource, path=path)

    @property
    def file_ext(self) -> str:
        return ".json"


class OutgoingFeatureArrays(OutgoingPreprocessResource):
    def export(self, path: Path):
        struct_for_export = eds.FeatureArrays(data=self.resource)
        edc.FeatureArraysWriter().export(obj=struct_for_export, path=path)

    @property
    def file_ext(self) -> str:
        return ".json"


class OutgoingClassLabels(OutgoingPreprocessResource):
    def export(self, path: Path):
        struct_for_export = eds.ClassLabels(data=self.resource)
        edc.ClassLabelsWriter().export(obj=struct_for_export, path=path)

    @property
    def file_ext(self) -> str:
        return ".json"


class OutgoingMeasurementColumnNames(OutgoingPreprocessResource):
    def export(self, path: Path):
        struct_for_export = eds.MeasurementColumnNames(data=self.resource)
        edc.MeasurementColumnNamesWriter().export(struct_for_export, path=path)

    @property
    def file_ext(self) -> str:
        return ".json"


@dataclass
class PreprocessModuleResources(ABC):
    module_name: str = None

    def __post_init__(self):
        config_reader = config.ConfigReader()
        for object_field in fields(self):
            if object_field.name != "name" and getattr(self, object_field.name) is None:
                value = config_reader.get_config_value(f"preprocess.{self.module_name}.resources.{object_field.name}")







@dataclass
class PrefilterResources:
    icustay: IncomingPreprocessResource = field(
        default_factory=lambda: IncomingCSVDataFrame(
            resource_id=cfp.PREFILTER_INPUT_FILES["icustay"]
        )
    )
    bg: IncomingPreprocessResource = field(
        default_factory=lambda: IncomingCSVDataFrame(
            resource_id=cfp.PREFILTER_INPUT_FILES["bg"]
        )
    )
    vital: IncomingPreprocessResource = field(
        default_factory=lambda: IncomingCSVDataFrame(
            resource_id=cfp.PREFILTER_INPUT_FILES["vital"]
        )
    )
    lab: IncomingPreprocessResource = field(
        default_factory=lambda: IncomingCSVDataFrame(
            resource_id=cfp.PREFILTER_INPUT_FILES["lab"]
        )
    )


@dataclass
class PrefilterOutputConstructors:
    prefiltered_icustay: Callable[..., OutgoingPreprocessResource] = (
        OutgoingPreprocessDataFrame
    )
    prefiltered_bg: Callable[..., OutgoingPreprocessResource] = (
        OutgoingPreprocessDataFrame
    )
    prefiltered_lab: Callable[..., OutgoingPreprocessResource] = (
        OutgoingPreprocessDataFrame
    )
    prefiltered_vital: Callable[..., OutgoingPreprocessResource] = (
        OutgoingPreprocessDataFrame
    )


@dataclass
class ICUStayMeasurementMergerResources:
    prefiltered_icustay: IncomingFeatherDataFrame = field(
        default_factory=lambda: IncomingFeatherDataFrame(
            resource_id=cfp.STAY_MEASUREMENT_INPUT_FILES["prefiltered_icustay"]
        )
    )
    prefiltered_bg: IncomingFeatherDataFrame = field(
        default_factory=lambda: IncomingFeatherDataFrame(
            resource_id=cfp.STAY_MEASUREMENT_INPUT_FILES["prefiltered_bg"]
        )
    )
    prefiltered_lab: IncomingFeatherDataFrame = field(
        default_factory=lambda: IncomingFeatherDataFrame(
            resource_id=cfp.STAY_MEASUREMENT_INPUT_FILES["prefiltered_lab"]
        )
    )
    prefiltered_vital: IncomingFeatherDataFrame = field(
        default_factory=lambda: IncomingFeatherDataFrame(
            resource_id=cfp.STAY_MEASUREMENT_INPUT_FILES["prefiltered_vital"]
        )
    )


@dataclass
class ICUStayMeasurementMergerOutputConstructors:
    icustay_bg_lab_vital: Callable[..., OutgoingPreprocessResource] = (
        OutgoingPreprocessDataFrame
    )
    bg_lab_vital_summary_stats: Callable[..., OutgoingPreprocessResource] = (
        OutgoingPreprocessDataFrame
    )


@dataclass
class AdmissionListBuilderResources:
    icustay_bg_lab_vital: IncomingFeatherDataFrame = field(
        default_factory=lambda: IncomingFeatherDataFrame(
            resource_id=cfp.FULL_ADMISSION_LIST_INPUT_FILES[
                "icustay_bg_lab_vital"
            ]
        )
    )


@dataclass
class AdmissionListBuilderOutputConstructors:
    full_admission_list: Callable[..., OutgoingPreprocessResource] = (
        OutgoingFullAdmissionData
    )


@dataclass
class FeatureBuilderResources:
    full_admission_list: IncomingPreprocessResource = field(
        default_factory=lambda: IncomingFullAdmissionData(
            resource_id=cfp.FEATURE_BUILDER_INPUT_FILES["full_admission_list"]
        )
    )
    bg_lab_vital_summary_stats: IncomingFeatherDataFrame = field(
        default_factory=lambda: IncomingFeatherDataFrame(
            resource_id=cfp.FEATURE_BUILDER_INPUT_FILES[
                "bg_lab_vital_summary_stats"
            ]
        )
    )


@dataclass
class FeatureBuilderOutputConstructors:
    processed_admission_list: Callable[..., OutgoingPreprocessResource] = (
        OutgoingFullAdmissionData
    )


@dataclass
class FeatureFinalizerResources:
    processed_admission_list: IncomingFullAdmissionData = field(
        default_factory=lambda: IncomingFullAdmissionData(
            resource_id=cfp.FEATURE_FINALIZER_INPUT_FILES[
                "processed_admission_list"
            ]
        )
    )


@dataclass
class FeatureFinalizerOutputConstructors:
    in_hospital_mortality_list: Callable[..., OutgoingPreprocessResource] = (
        OutgoingClassLabels
    )
    measurement_col_names: Callable[..., OutgoingPreprocessResource] = (
        OutgoingMeasurementColumnNames
    )
    measurement_data_list: Callable[..., OutgoingPreprocessResource] = (
        OutgoingFeatureArrays
    )
