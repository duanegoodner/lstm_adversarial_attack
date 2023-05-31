import sys
import torch
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.resource_io as rio
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

model = rio.ResourceImporter().import_pickle_to_object(path=model_path)
checkpoint = torch.load(checkpoint_path)

# drop the Sequential object
# module_list = list(model.modules())[1:]


def create_logit_no_dropout_model(
    original_model: torch.nn.Sequential,
) -> torch.nn.Sequential:
    module_list = list(original_model.modules())
    assert type(module_list[0]) == torch.nn.Sequential
    new_module_list = [
        item if type(item) != torch.nn.Dropout else torch.nn.Dropout(0)
        for item in module_list[1:-1]
    ]
    return torch.nn.Sequential(*new_module_list)


logit_no_dropout_model = create_logit_no_dropout_model(original_model=model)

dataset = X19MGeneralDatasetWithIndex.from_feaure_finalizer_output()
data_loader = DataLoader(
    dataset=dataset,
    batch_size=4,
    collate_fn=x19m_with_index_collate_fn,
    shuffle=False,
)
