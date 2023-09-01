from typing import Any, Type

import msgspec
import numpy as np
import torch
import lstm_adversarial_attack.config_paths as cfp



class OptimizerStateDict(msgspec.Struct):
    state: dict[int, dict[str, torch.Tensor]]
    param_groups: list[dict[str, Any]]


def enc_hook(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if isinstance(obj, np.float64):
        return float(obj)
    else:
        # Raise a NotImplementedError for other types
        raise NotImplementedError(
            f"Objects of type {type(obj)} are not supported")


def dec_hook(decode_type: Type, obj: Any) -> Any:
    if decode_type is torch.Tensor:
        return torch.tensor(obj)


encoder = msgspec.json.Encoder(enc_hook=enc_hook)
decoder = msgspec.json.Decoder(OptimizerStateDict, dec_hook=dec_hook)



full_checkpoint_pickle = (
    cfp.CV_ASSESSMENT_OUTPUT_DIR
    / "cv_training_20230831232347009480"
    / "checkpoints"
    / "fold_0"
    / "2023-08-31_23_25_12.647384.tar"
)
full_checkpoint = torch.load(full_checkpoint_pickle)
opt_state_dict = OptimizerStateDict(**full_checkpoint["optimizer_state_dict"])
encoded_opt_state_dict = encoder.encode(opt_state_dict)
decoded_opt_state_dict = decoder.decode(encoded_opt_state_dict)
