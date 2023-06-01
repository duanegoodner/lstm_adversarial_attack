import sys
import torch
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parent.parent.parent))
from adv_attack_trainer import AdversarialAttackTrainer
import lstm_adversarial_attack.resource_io as rio
from attacker_helpers import AdversarialAttackerBuilder, AdversarialLoss
from lstm_adversarial_attack.config_paths import ATTACK_OUTPUT_DIR, TRAINING_OUTPUT_DIR
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

attacker_builder = AdversarialAttackerBuilder(
    full_model=model,
    state_dict=checkpoint["state_dict"],
    batch_size=4,
    input_size=19,
    max_sequence_length=48,
)

attacker = attacker_builder.build()

data_loader = DataLoader(
    dataset=X19MGeneralDatasetWithIndex.from_feaure_finalizer_output(),
    batch_size=4,
    shuffle=False,
    collate_fn=x19m_with_index_collate_fn
)

if torch.cuda.is_available():
    cur_device = torch.device("cuda:0")
else:
    cur_device = torch.device("cpu")

attack_trainer = AdversarialAttackTrainer(
    device=cur_device,
    attacker=attacker,
    loss_fn=AdversarialLoss(kappa=0.0),
    optimizer=torch.optim.Adam(
        params=attacker.parameters(), lr=1e-4
    ),
    data_loader=data_loader,
    output_dir=ATTACK_OUTPUT_DIR
)

attack_trainer.train_attacker()
