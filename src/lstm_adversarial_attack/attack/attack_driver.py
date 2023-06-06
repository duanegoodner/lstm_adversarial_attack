import sys
import torch
from pathlib import Path
from typing import Callable

sys.path.append(str(Path(__file__).parent.parent.parent))
from adv_attack_trainer import AdversarialAttackTrainer
from attack_result_data_structs import (
    TrainerResult,
    TrainerSuccessSummary,
)
import lstm_adversarial_attack.resource_io as rio
from lstm_adversarial_attack.config_paths import (
    DEFAULT_ATTACK_TARGET_DIR,
    ATTACK_OUTPUT_DIR,
)
from lstm_adversarial_attack.x19_mort_general_dataset import (
    X19MGeneralDatasetWithIndex,
    x19m_with_index_collate_fn,
)


class AttackDriver:
    def __init__(
        self,
        device: torch.device,
        model_path: Path,
        checkpoint_path: Path,
        batch_size: int = 128,
        epochs_per_batch: int = 100,
        kappa: float = 0.0,
        lambda_1: float = 3e-2,
        optimizer_constructor: Callable = torch.optim.SGD,
        optimizer_constructor_kwargs: dict = None,
        max_num_samples=None,
        sample_selection_seed=None,
        attack_misclassified_samples: bool = False,
        save_train_result: bool = False,
        save_full_trainer: bool = False,
        return_full_trainer: bool = False,
    ):
        self.device = device
        self.model_path = model_path
        self.checkpoint_path = checkpoint_path
        self.batch_size = batch_size
        self.epochs_per_batch = epochs_per_batch
        self.kappa = kappa
        self.lambda_1 = lambda_1
        self.optimizer_constructor = optimizer_constructor
        if optimizer_constructor_kwargs is None:
            optimizer_constructor_kwargs = {"lr": 1e-1}
        self.optimizer_constructor_kwargs = optimizer_constructor_kwargs
        self.max_num_samples = max_num_samples
        self.sample_selection_seed = sample_selection_seed
        self.dataset = (
            X19MGeneralDatasetWithIndex.from_feaure_finalizer_output(
                max_num_samples=max_num_samples,
                random_seed=sample_selection_seed,
            )
        )
        self.collate_fn = x19m_with_index_collate_fn
        self.attack_misclassified_samples = attack_misclassified_samples
        self.save_train_result = save_train_result
        self.save_full_trainer = save_full_trainer
        self.return_full_trainer = return_full_trainer

    def __call__(self) -> AdversarialAttackTrainer | TrainerResult:
        model = rio.ResourceImporter().import_pickle_to_object(
            path=self.model_path
        )
        checkpoint = torch.load(self.checkpoint_path)
        attack_trainer = AdversarialAttackTrainer(
            device=self.device,
            model=model,
            state_dict=checkpoint["state_dict"],
            batch_size=self.batch_size,
            kappa=self.kappa,
            lambda_1=self.lambda_1,
            epochs_per_batch=self.epochs_per_batch,
            optimizer_constructor=self.optimizer_constructor,
            optimizer_constructor_kwargs=self.optimizer_constructor_kwargs,
            dataset=self.dataset,
            collate_fn=self.collate_fn,
            attack_misclassified_samples=self.attack_misclassified_samples,
        )

        train_result = attack_trainer.train_attacker()

        if self.save_train_result or self.save_full_trainer:
            output_dir = rio.create_timestamped_dir(
                parent_path=ATTACK_OUTPUT_DIR
            )
            if self.save_train_result:
                train_result_output_path = output_dir / "train_result.pickle"
                rio.ResourceExporter().export(
                    resource=train_result, path=train_result_output_path
                )
            if self.save_full_trainer:
                trainer_output_path = output_dir / "trainer.pickle"
                rio.ResourceExporter().export(
                    resource=attack_trainer, path=trainer_output_path
                )

        if self.return_full_trainer:
            return attack_trainer
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

    attack_driver = AttackDriver(
        device=cur_device,
        batch_size=1,
        epochs_per_batch=100,
        model_path=DEFAULT_ATTACK_TARGET_DIR / "model.pickle",
        checkpoint_path=checkpoint_path,
        max_num_samples=29,
        sample_selection_seed=1357,
        save_train_result=False,
    )

    trainer_result = attack_driver()
    success_summary = TrainerSuccessSummary(trainer_result=trainer_result)
