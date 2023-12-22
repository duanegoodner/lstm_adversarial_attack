import collections
import pickle
import msgspec.json
import time
import torch

import lstm_adversarial_attack.resource_io as rio
import lstm_adversarial_attack.config_paths as cfp
import lstm_adversarial_attack.attack.attack_data_structs as ads
import json
from collections import OrderedDict
from pathlib import PosixPath
from typing import Any


class AttackTunerDriverDictEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, PosixPath):
            return str(o)
        if isinstance(o, torch.device):
            return o.type
        if isinstance(o, ads.AttackTuningRanges):
            return o.__dict__
        return json.JSONEncoder.default(self, o)


class StateDictEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, OrderedDict):
            return {key: value for key, value in o.items()}
        if isinstance(o, torch.Tensor):
            return o.tolist()
        return json.JSONEncoder.default(self, o)


attack_tuner_driver_dict = rio.ResourceImporter().import_pickle_to_object(
    path=cfp.ATTACK_HYPERPARAMETER_TUNING
    / "2023-07-01_11_03_13.591090"
    / "attack_driver_dict.pickle"
)

state_dict_tensors = attack_tuner_driver_dict["target_model_checkpoint"][
    "state_dict"
]
state_dict_lists = {
    key: tensor.tolist() for key, tensor in state_dict_tensors.items()
}

dill_state_dict_path = (
    cfp.CV_ASSESSMENT_OUTPUT_DIR / "test_dill_state_dict.pickle"
)
standard_json_state_dict_path = (
    cfp.CV_ASSESSMENT_OUTPUT_DIR / "test_standard_state_dict.json"
)
standard_json_two_step_state_dict_path = (
    cfp.CV_ASSESSMENT_OUTPUT_DIR
    / "test_standard_json_two_step_state_dict.json"
)
msgspec_json_state_dict_path = (
    cfp.CV_ASSESSMENT_OUTPUT_DIR / "test_msgspec_state_dict.json"
)

dill_write_start = time.time()
with dill_state_dict_path.open(mode="wb") as out_file:
    pickle.dump(obj=state_dict_tensors, file=out_file)
dill_write_end = time.time()

dill_load_start = time.time()
with dill_state_dict_path.open(mode="rb") as in_file:
    dill_imported_state_dict = pickle.load(in_file)
dill_load_end = time.time()



standard_json_write_start = time.time()
with standard_json_state_dict_path.open(mode="w") as out_file:
    json.dump(obj=state_dict_lists, fp=out_file)
standard_json_write_end = time.time()

standard_json_load_start = time.time()
with standard_json_state_dict_path.open(mode="r") as in_file:
    standard_json_imported_state_dict = json.load(fp=in_file)
standard_json_load_end = time.time()



msgspec_write_start = time.time()
msgspec_encoder = msgspec.json.Encoder()
msgspec_encoded_state_dict = msgspec_encoder.encode(state_dict_lists)
with msgspec_json_state_dict_path.open(mode="wb") as out_file:
    out_file.write(msgspec_encoded_state_dict)
msgspec_write_end = time.time()

msgspec_load_start = time.time()
decoder = msgspec.json.Decoder()
with msgspec_json_state_dict_path.open(mode="rb") as in_file:
    encoded_data = in_file.read()
msgspec_imported_state_dict = decoder.decode(encoded_data)
msgspec_load_end = time.time()



print(f"dill write time = {dill_write_end - dill_write_start}")
print(f"dill load time = {dill_load_end - dill_load_start}")

print(
    "standard json write time ="
    f" {standard_json_write_end - standard_json_write_start}"
)
print(
    "standard json load time ="
    f" {standard_json_load_end - standard_json_load_start}"
)

print(f"msgspec write time = {msgspec_write_end - msgspec_write_start}")
print(f"msgspec load time = {msgspec_load_end - msgspec_load_start}")
#
# limited_checkpoint["converted_state_dict"] = {
#     key: tensor.tolist()
#     for key, tensor in limited_checkpoint["state_dict"].items()
# }
#
# limited_checkpoint_path = (
#     cfp.ATTACK_HYPERPARAMETER_TUNING / "limited_checkpoint.json"
# )
#
# with limited_checkpoint_path.open(mode="w") as out_file:
#     json.dump(obj=limited_checkpoint, fp=out_file, cls=StateDictEncoder)
#
# with limited_checkpoint_path.open(mode="r") as in_file:
#     re_imported_limited_checkpoint = json.load(fp=in_file)
#
# final_state_dict = collections.OrderedDict()
# for key, nested_list in re_imported_limited_checkpoint["state_dict"].items():
#     final_state_dict[key] = torch.tensor(nested_list, dtype=torch.float32)
