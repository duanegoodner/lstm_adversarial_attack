import sys
import torch
from torch.utils.data import Subset
from pathlib import Path
from typing import Callable

sys.path.append(str(Path(__file__).parent.parent.parent))
from adv_attack_trainer import AdversarialAttackTrainer
from attack_result_data_structs import (
    AttackSummary,
    TrainerResult,
    TrainerSuccessSummary,
)
import lstm_adversarial_attack.resource_io as rio
from lstm_adversarial_attack.config_paths import DEFAULT_ATTACK_TARGET_DIR
from lstm_adversarial_attack.x19_mort_general_dataset import (
    X19MGeneralDatasetWithIndex,
    x19m_with_index_collate_fn,
)


def run_adv_attack_trainer(
    device: torch.device,
    model_path: Path,
    checkpoint_path: Path,
    batch_size: int = 128,
    kappa: float = 0.0,
    lambda_1: float = 0.1,
    epochs_per_batch: int = 1000,
    optimizer_constructor: Callable = torch.optim.Adam,
    optimizer_constructor_kwargs: dict = None,
    dataset: X19MGeneralDatasetWithIndex = X19MGeneralDatasetWithIndex.from_feaure_finalizer_output(max_num_samples=256),
    collate_fn: Callable = x19m_with_index_collate_fn,
    inference_batch_size: int = 128,
    attack_misclassified_samples: bool = False,
    use_weighted_data_loader: bool = False,
    save_result: bool = False,
    return_full_trainer: bool = False,
) -> AdversarialAttackTrainer | TrainerResult:
    if optimizer_constructor_kwargs is None:
        optimizer_constructor_kwargs = {"lr": 1e-2}

    model = rio.ResourceImporter().import_pickle_to_object(path=model_path)
    checkpoint = torch.load(checkpoint_path)

    trainer = AdversarialAttackTrainer(
        device=device,
        model=model,
        state_dict=checkpoint["state_dict"],
        batch_size=batch_size,
        kappa=kappa,
        lambda_1=lambda_1,
        epochs_per_batch=epochs_per_batch,
        optimizer_constructor=optimizer_constructor,
        optimizer_constructor_kwargs=optimizer_constructor_kwargs,
        dataset=dataset,
        collate_fn=collate_fn,
        inference_batch_size=inference_batch_size,
        attack_misclassified_samples=attack_misclassified_samples,
        use_weighted_data_loader=use_weighted_data_loader,
        save_result=save_result,
    )

    train_result = trainer.train_attacker()

    if return_full_trainer:
        return trainer
    else:
        return train_result


if __name__ == "__main__":
    if torch.cuda.is_available():
        cur_device = torch.device("cuda:0")
    else:
        cur_device = torch.device("cpu")

    checkpoint_files = list(DEFAULT_ATTACK_TARGET_DIR.glob("*.tar"))
    assert len(checkpoint_files) == 1
    checkpoint_path = checkpoint_files[0]

    trainer_result = run_adv_attack_trainer(
        device=cur_device,
        model_path=DEFAULT_ATTACK_TARGET_DIR / "model.pickle",
        checkpoint_path=checkpoint_path,
        save_result=False,
    )

    success_summary = TrainerSuccessSummary(trainer_result=trainer_result)
