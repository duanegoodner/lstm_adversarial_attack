from dataclasses import dataclass
import pandas as pd


@dataclass
class FullAdmissionData:
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
