import datetime
from functools import cached_property

import msgspec
import pandas as pd


class NewFullAdmissionData(msgspec.Struct):
    """
    Container used as elements list build by FullAdmissionListBuilder
    """

    subject_id: int
    hadm_id: int
    icustay_id: int
    admittime: pd.Timestamp
    dischtime: pd.Timestamp
    hospital_expire_flag: int
    intime: pd.Timestamp
    outtime: pd.Timestamp
    time_series: pd.DataFrame


class DecomposedTimeSeries(msgspec.Struct):
    index: list[int]
    time_vals: list[pd.Timestamp]
    data: list[list[float]]


class NewFullAdmissionDataListHeader(msgspec.Struct):
    time_series_col_names: list[str]
    time_series_dtypes: list[str]

    @property
    def dtype_dict(self) -> dict[str, str]:
        return dict(zip(self.time_series_col_names, self.time_series_dtypes))
