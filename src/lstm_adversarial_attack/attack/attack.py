import sys
import torch
from pathlib import Path
from typing import Callable

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.attack.adv_attack_trainer as aat
import lstm_adversarial_attack.attack.attack_data_structs as ads
import lstm_adversarial_attack.attack.attack_result_data_structs as ards
import lstm_adversarial_attack.path_searches as ps
import lstm_adversarial_attack.resource_io as rio
import lstm_adversarial_attack.config_paths as cfg_paths

from lstm_adversarial_attack.x19_mort_general_dataset import (
    X19MGeneralDatasetWithIndex,
    x19m_with_index_collate_fn,
)


class AttackDriver:
    def __init__(
        self,
        device: torch.device,
        model_path: Path,
        checkpoint: dict,
        batch_size: int,
        epochs_per_batch: int,
        kappa: float,
        lambda_1: float,
        optimizer_constructor: Callable,
        optimizer_constructor_kwargs: dict = None,
        max_num_samples=None,
        sample_selection_seed=None,
        attack_misclassified_samples: bool = False,
        output_dir: Path = None,
        result_file_prefix: str = "",
        save_attack_driver: bool = False,
        checkpoint_interval: int = None,
    ):
        self.device = device
        self.model_path = model_path
        self.checkpoint = checkpoint
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
        if self.sample_selection_seed is not None:
            torch.manual_seed(self.sample_selection_seed)
        self.dataset = (
            X19MGeneralDatasetWithIndex.from_feature_finalizer_output(
                max_num_samples=max_num_samples,
                random_seed=sample_selection_seed,
            )
        )
        self.collate_fn = x19m_with_index_collate_fn
        self.attack_misclassified_samples = attack_misclassified_samples
        self.result_file_prefix = result_file_prefix
        self.save_attack_driver = save_attack_driver
        self.output_dir = self.initialize_output_dir(output_dir=output_dir)
        self.checkpoint_interval = checkpoint_interval

    def initialize_output_dir(self, output_dir: Path | None):
        if output_dir is None:
            output_dir = rio.create_timestamped_dir(
                parent_path=cfg_paths.FROZEN_HYPERPARAMETER_ATTACK
            )
        if self.save_attack_driver:
            rio.ResourceExporter().export(
                resource=self, path=output_dir / "attack_driver.pickle"
            )
        return output_dir

    @classmethod
    def from_attack_hyperparameter_settings(
        cls,
        device: torch.device,
        model_path: Path,
        checkpoint: dict,
        settings: ads.AttackHyperParameterSettings,
        epochs_per_batch: int = None,
        max_num_samples: int = None,
        sample_selection_seed: int = None,
        attack_misclassified_samples: bool = False,
        output_dir: Path = None,
        save_attack_driver: bool = False,
        checkpoint_interval: int = None,
    ):
        return cls(
            device=device,
            model_path=model_path,
            checkpoint=checkpoint,
            batch_size=2**settings.log_batch_size,
            kappa=settings.kappa,
            lambda_1=settings.lambda_1,
            optimizer_constructor=getattr(
                torch.optim, settings.optimizer_name
            ),
            optimizer_constructor_kwargs={"lr": settings.learning_rate},
            max_num_samples=max_num_samples,
            sample_selection_seed=sample_selection_seed,
            attack_misclassified_samples=attack_misclassified_samples,
            output_dir=output_dir,
            epochs_per_batch=epochs_per_batch,
            save_attack_driver=save_attack_driver,
            checkpoint_interval=checkpoint_interval,
        )

    @classmethod
    def from_attack_hyperparameter_tuning(
        cls,
        device: torch.device,
        tuning_output_dir: Path = None,
        max_num_samples: int = None,
        epochs_per_batch: int = None,
        sample_selection_seed: int = None,
        save_attack_driver: bool = True,
        checkpoint_interval: int = None,
    ):
        if tuning_output_dir is None:
            tuning_output_dir = ps.subdir_with_latest_content_modification(
                root_path=cfg_paths.ATTACK_HYPERPARAMETER_TUNING
            )
        optuna_study = rio.ResourceImporter().import_pickle_to_object(
            path=tuning_output_dir / "optuna_study.pickle"
        )
        tuner_driver = rio.ResourceImporter().import_pickle_to_object(
            path=tuning_output_dir / "attack_tuner_driver.pickle"
        )
        if epochs_per_batch is None:
            epochs_per_batch = tuner_driver.epochs_per_batch

        return cls.from_attack_hyperparameter_settings(
            device=device,
            model_path=tuner_driver.target_model_path,
            checkpoint=tuner_driver.target_model_checkpoint,
            settings=ads.AttackHyperParameterSettings(
                **optuna_study.best_params
            ),
            epochs_per_batch=epochs_per_batch,
            max_num_samples=max_num_samples,
            sample_selection_seed=sample_selection_seed,
            save_attack_driver=save_attack_driver,
            checkpoint_interval=checkpoint_interval,
        )

    def __call__(self) -> aat.AdversarialAttackTrainer | ards.TrainerResult:
        model = rio.ResourceImporter().import_pickle_to_object(
            path=self.model_path
        )
        attack_trainer = aat.AdversarialAttackTrainer(
            device=self.device,
            model=model,
            state_dict=self.checkpoint["state_dict"],
            batch_size=self.batch_size,
            kappa=self.kappa,
            lambda_1=self.lambda_1,
            epochs_per_batch=self.epochs_per_batch,
            optimizer_constructor=self.optimizer_constructor,
            optimizer_constructor_kwargs=self.optimizer_constructor_kwargs,
            dataset=self.dataset,
            collate_fn=self.collate_fn,
            attack_misclassified_samples=self.attack_misclassified_samples,
            output_dir=self.output_dir,
            checkpoint_interval=self.checkpoint_interval,
        )

        train_result = attack_trainer.train_attacker()

        train_result_output_path = rio.create_timestamped_filepath(
            parent_path=self.output_dir,
            file_extension="pickle",
            suffix=f"{self.result_file_prefix}_final_attack_result",
        )

        rio.ResourceExporter().export(
            resource=train_result, path=train_result_output_path
        )
        return train_result


def attack_with_tuned_params(sample_selection_seed: int = 2023, checkpoint_interval: int = 50):
    if torch.cuda.is_available():
        cur_device = torch.device("cuda:0")
    else:
        cur_device = torch.device("cpu")

    attack_driver = AttackDriver.from_attack_hyperparameter_tuning(
        device=cur_device,
        sample_selection_seed=2023,
        checkpoint_interval=50,
    )
    trainer_result = attack_driver()
    success_summary = ards.TrainerSuccessSummary(trainer_result=trainer_result)

    return success_summary


if __name__ == "__main__":
    cur_success_summary = attack_with_tuned_params()

