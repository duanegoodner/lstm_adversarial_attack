from abc import ABC, abstractmethod
from dataclasses import Field, dataclass, fields
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, TypeVar

import pandas as pd

import lstm_adversarial_attack.preprocess.encode_decode as edc
import lstm_adversarial_attack.preprocess.encode_decode_structs as eds
import lstm_adversarial_attack.utils.resource_io as rio
from lstm_adversarial_attack.config import CONFIG_READER, PATH_CONFIG_READER

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
        edc.export_admission_data_list(
            admission_data_list=self._resource, path=path
        )

    @property
    def file_ext(self) -> str:
        return ".json"


class OutgoingFeaturesList(OutgoingPreprocessResource):
    def export(self, path: Path):
        edc.AdmissionDataWriter().export(obj=self.resource, path=path)

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
    collection_ids: dict[str, str]
    module_name: str
    default_data_source_type: DataSourceType
    resource_pool: dict[str, OutgoingPreprocessResource] = None

    @property
    def build_info(self) -> tuple[Field, ...]:
        """
        The object fields defined by parent class PreprocessModuleResources.
        """
        return fields(PreprocessModuleResources)

    @property
    def resource_fields(self) -> tuple[Field, ...]:
        """
        The object fields defined by the subclass.
        """
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

    @property
    def file_resource_names_dict(self) -> dict[str, dict[str, str]]:
        return PATH_CONFIG_READER.get_config_value(
            f"preprocess.{self.module_name}.resources"
        )

    def validate_file_resources(self):
        resource_keys = [
            key
            for inner_dict in self.file_resource_names_dict.values()
            for key in inner_dict.keys()
        ]
        assert (
            len(resource_keys)
            == len(set(resource_keys))
            == len(self.resource_fields)
        )
        assert all(
            [
                object_field.name in resource_keys
                for object_field in self.resource_fields
            ]
        )

    def get_file_resources(self):
        self.validate_file_resources()

        config_path_keys = {
            "db": "from_db",
            "preprocess": "from_other_preprocess_modules",
        }

        for short_name, long_name in config_path_keys.items():
            for object_field in self.resource_fields:
                if (
                    object_field.name
                    in self.file_resource_names_dict.get(long_name, {}).keys()
                ):
                    resource_path = (
                        Path(
                            PATH_CONFIG_READER.read_path(
                                f"{short_name}.output_root"
                            )
                        )
                        / self.collection_ids[short_name]
                        / self.file_resource_names_dict.get(long_name, {}).get(
                            object_field.name
                        )
                    )
                    attr = object_field.type(resource_id=resource_path)
                    setattr(self, object_field.name, attr)

    def get_pool_resources(self):
        for object_field in self.resource_fields:
            if getattr(self, object_field.name) is None:
                setattr(
                    self,
                    object_field.name,
                    object_field.type(
                        resource_id=object_field.name,
                        resource_pool=self.resource_pool,
                    ),
                )

    def __post_init__(self):

        if self.default_data_source_type == DataSourceType.FILE:
            self.get_file_resources()

        if self.default_data_source_type == DataSourceType.POOL:
            self.get_pool_resources()


@dataclass
class PrefilterResources(PreprocessModuleResources):
    icustay: IncomingDataFrame = None
    bg: IncomingDataFrame = None
    lab: IncomingDataFrame = None
    vital: IncomingDataFrame = None


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
class ICUStayMeasurementMergerResources(PreprocessModuleResources):
    prefiltered_icustay: IncomingDataFrame = None
    prefiltered_bg: IncomingDataFrame = None
    prefiltered_lab: IncomingDataFrame = None
    prefiltered_vital: IncomingDataFrame = None


@dataclass
class ICUStayMeasurementMergerOutputConstructors:
    icustay_bg_lab_vital: Callable[..., OutgoingPreprocessResource] = (
        OutgoingPreprocessDataFrame
    )
    bg_lab_vital_summary_stats: Callable[..., OutgoingPreprocessResource] = (
        OutgoingPreprocessDataFrame
    )


@dataclass
class AdmissionListBuilderResources(PreprocessModuleResources):
    icustay_bg_lab_vital: IncomingDataFrame = None


@dataclass
class AdmissionListBuilderOutputConstructors:
    full_admission_list: Callable[..., OutgoingPreprocessResource] = (
        OutgoingFullAdmissionData
    )


@dataclass
class FeatureBuilderResources(PreprocessModuleResources):
    full_admission_list: IncomingFullAdmissionData = None
    bg_lab_vital_summary_stats: IncomingDataFrame = None


@dataclass
class FeatureBuilderOutputConstructors:
    processed_admission_list: Callable[..., OutgoingPreprocessResource] = (
        OutgoingFullAdmissionData
    )


@dataclass
class FeatureFinalizerResources(PreprocessModuleResources):
    processed_admission_list: IncomingFullAdmissionData = None


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
