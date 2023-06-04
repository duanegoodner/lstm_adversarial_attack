import sys
import torch
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.resource_io as rio
from inferrer import StandardModelInferrer
from lstm_adversarial_attack.config_paths import TRAINING_OUTPUT_DIR
from lstm_adversarial_attack.x19_mort_general_dataset import (
    X19MGeneralDatasetWithIndex,
    x19m_with_index_collate_fn,
)

training_output_root = TRAINING_OUTPUT_DIR / "2023-05-30_22:02:07.515447"
model_path = training_output_root / "model.pickle"
checkpoint_path = (
    training_output_root / "checkpoints" / "2023-05-30_22:19:03.666893.tar"
)


def create_logit_no_dropout_model(
    original_model: torch.nn.Sequential,
) -> torch.nn.Sequential:
    new_module_list = [
        val if type(val) != torch.nn.Dropout else torch.nn.Dropout(0)
        for key, val in list(original_model._modules.items())[:-1]
    ]
    return torch.nn.Sequential(*new_module_list)


if __name__ == "__main__":
    if torch.cuda.is_available():
        cur_device = torch.device("cuda:0")
    else:
        cur_device = torch.device("cpu")

    model = rio.ResourceImporter().import_pickle_to_object(path=model_path)
    checkpoint = torch.load(checkpoint_path)

    logit_no_dropout_model = create_logit_no_dropout_model(
        original_model=model
    )

    logit_no_dropout_model.load_state_dict(
        checkpoint["state_dict"], strict=False
    )

    logit_no_dropout_model.load_state_dict(
        checkpoint["state_dict"], strict=False
    )

    my_inferrer = StandardModelInferrer(
        device=cur_device,
        model=logit_no_dropout_model,
        dataset=X19MGeneralDatasetWithIndex.from_feaure_finalizer_output(),
        collate_fn=x19m_with_index_collate_fn,
        batch_size=128,
    )

    inference_results = my_inferrer.infer()