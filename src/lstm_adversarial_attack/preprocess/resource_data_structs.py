from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields, Field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, TypeVar, NamedTuple

import pandas as pd
import lstm_adversarial_attack.config as config
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


class DataSourceType(Enum):
    FILE = auto()
    POOL = auto()


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
        self._resource_id = resource_id
        self._resource_pool = resource_pool

        if type(resource_id) == str:
            assert resource_pool is not None
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


class ResourceDetails(NamedTuple):
    data_source_type: DataSourceType
    constructor: Callable[..., IncomingPreprocessResource]


@dataclass
class ResourceInfoNew:
    module_name: str
    key: str
    constructor: Callable[..., IncomingPreprocessResource]
    data_source_type: DataSourceType
    config_path: Path = None

    def build_resource(
        self, resource_pool: dict[str, OutgoingPreprocessResource] = None
    ) -> IncomingPreprocessResource:
        config_reader = config.ConfigReader(config_path=self.config_path)

        if self.data_source_type == DataSourceType.POOL:
            assert resource_pool is not None and self.key in resource_pool
            return self.constructor(resource_id=self.key, resource_pool=resource_pool)
        if self.data_source_type == DataSourceType.FILE:
            path = Path(
                config_reader.read_path(
                    config_key=f"preprocess.{self.module_name}.resources.{self.key}"
                )
            )
            assert path.exists()
            return self.constructor(resource_id=path)


@dataclass
class ResourceInfo(ABC):
    @abstractmethod
    def build_resource(self, **kwargs):
        pass


@dataclass
class PoolResourceInfo(ResourceInfo):
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
class FileResourceInfo(ResourceInfo):
    key: str
    path: Path
    constructor: Callable[..., IncomingPreprocessResource]

    def build_resource(self, **kwargs) -> dict[str, IncomingPreprocessResource]:
        return {self.key: self.constructor(resource_id=self.path)}


@dataclass
class SingleOutputInfo:
    key: str
    constructor: Callable[..., OutgoingPreprocessResource]


class IncomingDataFrame(IncomingPreprocessResource):
    _import_dispatch = {".csv": pd.read_csv, ".feather": rio.feather_to_df}

    def _import_object(self) -> pd.DataFrame:
        file_extension = self._resource_id.suffix
        return self._import_dispatch[file_extension](self._resource_id)


class IncomingFullAdmissionData(IncomingPreprocessResource):
    def _import_object(self) -> list[eds.FullAdmissionData]:
        return edc.import_admission_data_list(path=self._resource_id)


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
        edc.export_admission_data_list(admission_data_list=self._resource, path=path)

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
    module_name: str
    default_data_source_type: DataSourceType
    config_file: Path = None
    resource_pool: dict[str, OutgoingPreprocessResource] = None

    @property
    def build_info(self) -> tuple[Field, ...]:
        return fields(PreprocessModuleResources)

    @property
    def resource_fields(self) -> tuple[Field, ...]:
        return tuple(
            [
                object_field
                for object_field in fields(self)
                if object_field not in self.build_info
            ]
        )

    @property
    def resources_dict(self) -> dict[str, IncomingPreprocessResource]:
        return {
            resource_field.name: getattr(self, resource_field.name)
            for resource_field in self.resource_fields
        }

    def __post_init__(self):
        config_reader = config.ConfigReader()

        for object_field in self.resource_fields:
            if getattr(self, object_field.name) is None:
                if self.default_data_source_type == DataSourceType.POOL:
                    setattr(
                        self,
                        object_field.name,
                        object_field.type(
                            resource_id=object_field.name,
                            resource_pool=self.resource_pool,
                        ),
                    )
                if self.default_data_source_type == DataSourceType.FILE:
                    path_str = config_reader.read_path(
                        f"preprocess.{self.module_name}.resources.{object_field.name}"
                    )
                    attr = object_field.type(resource_id=Path(path_str))
                    setattr(self, object_field.name, attr)


@dataclass
class PrefilterResources(PreprocessModuleResources):
    icustay: IncomingDataFrame = None
    bg: IncomingDataFrame = None
    lab: IncomingDataFrame = None
    vital: IncomingDataFrame = None

    # icustay: IncomingDataFrame = IncomingDataFrame(
    #     module_name="prefilter", resource_id="icustay"
    # )
    # bg: IncomingDataFrame = IncomingDataFrame(module_name="prefilter", resource_id="bg")
    # vital: IncomingDataFrame = IncomingDataFrame(
    #     module_name="prefilter", resource_id="vital"
    # )
    # lab: IncomingDataFrame = IncomingDataFrame(
    #     module_name="prefilter", resource_id="lab"
    # )

    # icustay: IncomingPreprocessResource = field(
    #     default_factory=lambda: IncomingDataFrame(
    #         resource_id=cfp.PREFILTER_INPUT_FILES["icustay"]
    #     )
    # )
    # bg: IncomingPreprocessResource = field(
    #     default_factory=lambda: IncomingDataFrame(
    #         resource_id=cfp.PREFILTER_INPUT_FILES["bg"]
    #     )
    # )
    # vital: IncomingPreprocessResource = field(
    #     default_factory=lambda: IncomingDataFrame(
    #         resource_id=cfp.PREFILTER_INPUT_FILES["vital"]
    #     )
    # )
    # lab: IncomingPreprocessResource = field(
    #     default_factory=lambda: IncomingDataFrame(
    #         resource_id=cfp.PREFILTER_INPUT_FILES["lab"]
    #     )
    # )


@dataclass
class PrefilterOutputConstructors:
    prefiltered_icustay: Callable[
        ..., OutgoingPreprocessResource
    ] = OutgoingPreprocessDataFrame
    prefiltered_bg: Callable[
        ..., OutgoingPreprocessResource
    ] = OutgoingPreprocessDataFrame
    prefiltered_lab: Callable[
        ..., OutgoingPreprocessResource
    ] = OutgoingPreprocessDataFrame
    prefiltered_vital: Callable[
        ..., OutgoingPreprocessResource
    ] = OutgoingPreprocessDataFrame


@dataclass
class ICUStayMeasurementMergerResources(PreprocessModuleResources):
    prefiltered_icustay: IncomingDataFrame = None
    prefiltered_bg: IncomingDataFrame = None
    prefiltered_lab: IncomingDataFrame = None
    prefiltered_vital: IncomingDataFrame = None

    # prefiltered_icustay: IncomingDataFrame = field(
    #     default_factory=lambda: IncomingDataFrame(
    #         resource_id=cfp.STAY_MEASUREMENT_INPUT_FILES["prefiltered_icustay"]
    #     )
    # )
    # prefiltered_bg: IncomingDataFrame = field(
    #     default_factory=lambda: IncomingDataFrame(
    #         resource_id=cfp.STAY_MEASUREMENT_INPUT_FILES["prefiltered_bg"]
    #     )
    # )
    # prefiltered_lab: IncomingDataFrame = field(
    #     default_factory=lambda: IncomingDataFrame(
    #         resource_id=cfp.STAY_MEASUREMENT_INPUT_FILES["prefiltered_lab"]
    #     )
    # )
    # prefiltered_vital: IncomingDataFrame = field(
    #     default_factory=lambda: IncomingDataFrame(
    #         resource_id=cfp.STAY_MEASUREMENT_INPUT_FILES["prefiltered_vital"]
    #     )
    # )


@dataclass
class ICUStayMeasurementMergerOutputConstructors:
    icustay_bg_lab_vital: Callable[
        ..., OutgoingPreprocessResource
    ] = OutgoingPreprocessDataFrame
    bg_lab_vital_summary_stats: Callable[
        ..., OutgoingPreprocessResource
    ] = OutgoingPreprocessDataFrame


@dataclass
class AdmissionListBuilderResources(PreprocessModuleResources):
    icustay_bg_lab_vital: IncomingDataFrame = None
    # icustay_bg_lab_vital: IncomingDataFrame = field(
    #     default_factory=lambda: IncomingDataFrame(
    #         resource_id=cfp.FULL_ADMISSION_LIST_INPUT_FILES[
    #             "icustay_bg_lab_vital"]
    #     )
    # )


@dataclass
class AdmissionListBuilderOutputConstructors:
    full_admission_list: Callable[
        ..., OutgoingPreprocessResource
    ] = OutgoingFullAdmissionData


@dataclass
class FeatureBuilderResources(PreprocessModuleResources):
    full_admission_list: IncomingFullAdmissionData = None
    bg_lab_vital_summary_stats: IncomingDataFrame = None
    # full_admission_list: IncomingPreprocessResource = field(
    #     default_factory=lambda: IncomingFullAdmissionData(
    #         resource_id=cfp.FEATURE_BUILDER_INPUT_FILES["full_admission_list"]
    #     )
    # )
    # bg_lab_vital_summary_stats: IncomingDataFrame = field(
    #     default_factory=lambda: IncomingDataFrame(
    #         resource_id=cfp.FEATURE_BUILDER_INPUT_FILES[
    #             "bg_lab_vital_summary_stats"]
    #     )
    # )


@dataclass
class FeatureBuilderOutputConstructors:
    processed_admission_list: Callable[
        ..., OutgoingPreprocessResource
    ] = OutgoingFullAdmissionData


@dataclass
class FeatureFinalizerResources(PreprocessModuleResources):
    processed_admission_list: IncomingFullAdmissionData = None
    # processed_admission_list: IncomingFullAdmissionData = field(
    #     default_factory=lambda: IncomingFullAdmissionData(
    #         resource_id=cfp.FEATURE_FINALIZER_INPUT_FILES[
    #             "processed_admission_list"]
    #     )
    # )


@dataclass
class FeatureFinalizerOutputConstructors:
    in_hospital_mortality_list: Callable[
        ..., OutgoingPreprocessResource
    ] = OutgoingClassLabels
    measurement_col_names: Callable[
        ..., OutgoingPreprocessResource
    ] = OutgoingMeasurementColumnNames
    measurement_data_list: Callable[
        ..., OutgoingPreprocessResource
    ] = OutgoingFeatureArrays
