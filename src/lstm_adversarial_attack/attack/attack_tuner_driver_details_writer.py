import collections

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

non_checkpoint_info = {
    key: value
    for key, value in attack_tuner_driver_dict.items()
    if key != "target_model_checkpoint"
}

limited_checkpoint = {}
limited_checkpoint["epoch_num"] = attack_tuner_driver_dict[
    "target_model_checkpoint"
]["epoch_num"]
limited_checkpoint["state_dict"] = attack_tuner_driver_dict[
    "target_model_checkpoint"
]["state_dict"]


non_checkpoint_info_path = (
    cfp.ATTACK_HYPERPARAMETER_TUNING / "test_non_checkpoint_info.json"
)

with non_checkpoint_info_path.open(mode="w") as out_file:
    json.dump(
        obj=non_checkpoint_info,
        fp=out_file,
        cls=AttackTunerDriverDictEncoder,
    )

limited_checkpoint["converted_state_dict"] = {
    key: tensor.tolist()
    for key, tensor in limited_checkpoint["state_dict"].items()
}

limited_checkpoint_path = (
    cfp.ATTACK_HYPERPARAMETER_TUNING / "limited_checkpoint.json"
)

with limited_checkpoint_path.open(mode="w") as out_file:
    json.dump(obj=limited_checkpoint, fp=out_file, cls=StateDictEncoder)

with limited_checkpoint_path.open(mode="r") as in_file:
    re_imported_limited_checkpoint = json.load(fp=in_file)

final_state_dict = collections.OrderedDict()
for key, nested_list in re_imported_limited_checkpoint["state_dict"].items():
    final_state_dict[key] = torch.tensor(nested_list, dtype=torch.float32)
