import datetime

import msgspec
import numpy as np
import time
from typing import Any, Type

import pandas as pd

import lstm_adversarial_attack.config_paths as cfp
import lstm_adversarial_attack.preprocess.new_preprocessor as pre
import lstm_adversarial_attack.resource_io as rio


def enc_hook(obj: Any) -> Any:
    # if isinstance(obj, np.ndarray):
    #     return obj.tolist()

    if isinstance(obj, pd.Timestamp):
        return obj.to_pydatetime()
    if isinstance(obj, pd.DataFrame):
        return obj.to_numpy().tolist()
    if isinstance(obj, np.datetime64):
        return obj.astype(datetime.datetime)
    if pd.isna(obj):
        return None
    else:
        raise NotImplementedError(
            f"Encoder does not support objects of type {type(obj)}"
        )


def dec_hook(type: Type, obj: Any) -> Any:
    if type is np.ndarray:
        return np.array(obj)
    else:
        raise NotImplementedError(
            f"Decoder does not support objects of type {type(obj)}"
        )


np_capable_encoder = msgspec.json.Encoder(enc_hook=enc_hook)
new_full_admission_list_decoder = msgspec.json.Decoder(
    list[pre.NewFullAdmissionData], dec_hook=dec_hook
)

admission_list = rio.ResourceImporter().import_pickle_to_list(
    path=cfp.FULL_ADMISSION_LIST_OUTPUT / "full_admission_list.pickle"
)

encode_start = time.time()
encoded_full_admission_list = np_capable_encoder.encode(admission_list)
encode_end = time.time()
print(f"encoding time = {encode_end - encode_start}")

write_start = time.time()
with (cfp.FULL_ADMISSION_LIST_OUTPUT / "full_admission_list.json").open(
    mode="wb"
) as outfile:
    outfile.write(encoded_full_admission_list)
write_end = time.time()
print(f"write time = {write_end - write_start}")
