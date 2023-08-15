from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TypeVar

import pandas as pd

import lstm_adversarial_attack.config_paths as cfp
import lstm_adversarial_attack.preprocess.encode_decode as edc
import lstm_adversarial_attack.preprocess.preprocess_data_structures as pds
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
        return {
            self.key: self.constructor(resource_id=self.path)
        }


class IncomingFeatherDataFrame(IncomingPreprocessResource):
    def _import_object(self) -> pd.DataFrame:
        return rio.feather_to_df(path=self.resource_id)


class IncomingCSVDataFrame(IncomingPreprocessResource):
    def _import_object(self) -> pd.DataFrame:
        return pd.read_csv(filepath_or_buffer=self.resource_id)


class IncomingPreprocessPickle(IncomingPreprocessResource):
    def _import_object(self) -> object:
        return rio.ResourceImporter().import_pickle_to_object(
            path=self.resource_id
        )


class IncomingFullAdmissionData(IncomingPreprocessResource):
    def _import_object(self) -> list[pds.NewFullAdmissionData]:
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
        edc.export_admission_data_list(data_obj=self._resource, path=path)

    @property
    def file_ext(self) -> str:
        return ".json"


class OutgoingListOfArrays(OutgoingPreprocessResource):
    def export(self, path: Path):
        edc.export_list_of_numpy_arrays(np_arrays=self.resource, path=path)

    @property
    def file_ext(self) -> str:
        return ".json"


class JsonReadyOutput(OutgoingPreprocessResource):
    def export(self, path: Path):
        edc.export_json_ready_object(obj=self.resource, path=path)

    @property
    def file_ext(self) -> str:
        return ".json"


@dataclass
class NewPrefilterResources:
    icustay: IncomingCSVDataFrame = field(
        default_factory=lambda: IncomingCSVDataFrame(
            resource_id=cfp.PREFILTER_INPUT_FILES["icustay"]
        )
    )
    bg: IncomingCSVDataFrame = field(
        default_factory=lambda: IncomingCSVDataFrame(
            resource_id=cfp.PREFILTER_INPUT_FILES["bg"]
        )
    )
    vital: IncomingCSVDataFrame = field(
        default_factory=lambda: IncomingCSVDataFrame(
            resource_id=cfp.PREFILTER_INPUT_FILES["vital"]
        )
    )
    lab: IncomingCSVDataFrame = field(
        default_factory=lambda: IncomingCSVDataFrame(
            resource_id=cfp.PREFILTER_INPUT_FILES["lab"]
        )
    )


@dataclass
class NewICUStayMeasurementMergerResources:
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
class NewAdmissionListBuilderResources:
    icustay_bg_lab_vital: IncomingFeatherDataFrame = field(
        default_factory=lambda: IncomingFeatherDataFrame(
            resource_id=cfp.FULL_ADMISSION_LIST_INPUT_FILES[
                "icustay_bg_lab_vital"
            ]
        )
    )


@dataclass
class NewFeatureBuilderResources:
    full_admission_list: IncomingFullAdmissionData = field(
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
class NewFeatureFinalizerResources:
    processed_admission_list: IncomingFullAdmissionData = field(
        default_factory=lambda: IncomingFullAdmissionData(
            resource_id=cfp.FEATURE_FINALIZER_INPUT_FILES[
                "processed_admission_list"
            ]
        )
    )
