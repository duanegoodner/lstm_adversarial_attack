import time

import msgspec
import numpy as np
import pandas as pd
import lstm_adversarial_attack.config_paths as cfp
import lstm_adversarial_attack.preprocess.new_preprocessor as pre
import lstm_adversarial_attack.resource_io as rio
from typing import Any


class FullAdmissionDataForJson(msgspec.Struct):
    subject_id: int
    hadm_id: int
    icustay_id: int
    admittime: str
    dischtime: str
    hospital_expire_flag: int
    intime: str
    outtime: str
    time_series: list[list[str | float]]

    @classmethod
    def from_in_app_object(cls, in_app_object: pre.NewFullAdmissionData):
        return cls(
            subject_id=int(in_app_object.subject_id),
            hadm_id=int(in_app_object.hadm_id),
            icusstay_id=int(in_app_object.icustay_id),
            admittime=str(in_app_object.admittime),
            dischtime=str(in_app_object.dischtime),
            hospital_expire_flag=int(in_app_object.hospital_expire_flag),
            intime=str(in_app_object.intime),
            outtime=str(in_app_object.outtime),
            time_series=in_app_object.time_series.values.tolist()
        )


def enc_hook(obj: Any) -> Any:
    if isinstance(obj, pre.NewFullAdmissionData):
        return FullAdmissionDataForJson(in_app_object=obj)
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    if pd.isna(obj):
        return None


encoder = msgspec.json.Encoder(enc_hook=enc_hook)


full_admission_list = rio.ResourceImporter().import_pickle_to_object(
    path=cfp.FULL_ADMISSION_LIST_OUTPUT / "full_admission_list.pickle"
)


# encoded_small_admission_list = encoder.encode(small_admission_list)

values_start = time.time()
time_series_list_a = [item.time_series.values.tolist() for item in full_admission_list]
values_end = time.time()

to_numpy_start = time.time()
time_series_list_b = [item.time_series.to_numpy().tolist() for item in full_admission_list]
to_numpy_end = time.time()

print(f"values time = {values_end - values_start}")
print(f"to_numpy time = {to_numpy_end - to_numpy_start}")

