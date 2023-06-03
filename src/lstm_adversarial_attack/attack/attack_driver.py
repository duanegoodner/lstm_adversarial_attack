import sys
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from adv_attack_trainer import AdversarialAttackTrainer
from adversarial_attacker import AdversarialAttacker
import lstm_adversarial_attack.resource_io as rio
from attacker_helpers import AdversarialLoss
from lstm_adversarial_attack.config_paths import (
    ATTACK_OUTPUT_DIR,
    TRAINING_OUTPUT_DIR,
)
from lstm_adversarial_attack.config_settings import MAX_OBSERVATION_HOURS
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

# attacker_builder = AdversarialAttackerBuilder(
#     full_model=model,
#     state_dict=checkpoint["state_dict"],
#     dataset=X19MGeneralDatasetWithIndex.from_feaure_finalizer_output(),
#     batch_size=128,
#     input_size=19,
#     max_sequence_length=48,
# )
# attacker = attacker_builder.build()

attacker = AdversarialAttacker(
    full_model=model,
    state_dict=checkpoint["state_dict"],
    input_size=19,
    max_sequence_length=MAX_OBSERVATION_HOURS,
    batch_size=128
)

if torch.cuda.is_available():
    cur_device = torch.device("cuda:0")
else:
    cur_device = torch.device("cpu")

attack_trainer = AdversarialAttackTrainer(
    device=cur_device,
    attacker=attacker,
    loss_fn=AdversarialLoss(kappa=0.0),
    lambda_1=0,
    optimizer=torch.optim.Adam(params=attacker.parameters(), lr=1e-4),
    dataset=X19MGeneralDatasetWithIndex.from_feaure_finalizer_output(),
    collate_fn=x19m_with_index_collate_fn,
    output_dir=ATTACK_OUTPUT_DIR,
    inference_batch_size=128,
    use_weighted_data_loader=False
)

# orig_predictions = attack_trainer.get_orig_predictions()

attack_trainer.train_attacker()
