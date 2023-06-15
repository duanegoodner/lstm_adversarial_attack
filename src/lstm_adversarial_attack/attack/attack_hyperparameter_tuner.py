import optuna
import torch
from dataclasses import dataclass
from optuna.pruners import BasePruner, MedianPruner
from optuna.samplers import BaseSampler, TPESampler
from pathlib import Path

from lstm_adversarial_attack.attack.attack import AttackDriver
from lstm_adversarial_attack.attack.attack_result_data_structs import TrainerSuccessSummary
from lstm_adversarial_attack.config_paths import ATTACK_HYPERPARAMETER_TUNING
import lstm_adversarial_attack.resource_io as rio
import lstm_adversarial_attack.attack.best_checkpoint_retriever as bcr


@dataclass
class AttackTuningRanges:
    kappa: tuple[float, float]
    lambda_1: tuple[float, float]
    optimizer_name: tuple[str, ...]
    learning_rate: tuple[float, float]
    log_batch_size: tuple[int, int]


@dataclass
class AttackHyperParameterSettings:
    kappa: float
    lambda_1: float
    optimizer_name: str
    learning_rate: float
    log_batch_size: int

    @classmethod
    def from_optuna_active_trial(
        cls, trial: optuna.Trial, tuning_ranges: AttackTuningRanges
    ):
        return cls(
            kappa=trial.suggest_float(
                "kappa", *tuning_ranges.kappa, log=False
            ),
            lambda_1=trial.suggest_float(
                "lambda_1", *tuning_ranges.lambda_1, log=True
            ),
            optimizer_name=trial.suggest_categorical(
                "optimizer_name", list(tuning_ranges.optimizer_name)
            ),
            learning_rate=trial.suggest_float(
                "learning_rate", *tuning_ranges.learning_rate, log=True
            ),
            log_batch_size=trial.suggest_int(
                "log_batch_size", *tuning_ranges.log_batch_size
            ),
        )


class AttackHyperParameterTuner:
    def __init__(
        self,
        device: torch.device,
        model_path: Path,
        # checkpoint_path: Path,
        checkpoint: dict,
        epochs_per_batch: int,
        max_num_samples: int,
        tuning_ranges: AttackTuningRanges,
        sample_selection_seed: int = 13579,
        pruner: BasePruner = MedianPruner(),
        hyperparameter_sampler: BaseSampler = TPESampler(),
        save_trial_info: bool = True,
    ):
        self.device = device
        self.model_path = model_path
        # self.checkpoint_path = checkpoint_path
        self.checkpoint = checkpoint
        self.epoch_per_batch = epochs_per_batch
        self.max_num_samples = max_num_samples
        self.tuning_ranges = tuning_ranges
        self.sample_selection_seed = (sample_selection_seed,)
        self.pruner = pruner
        self.hyperparameter_sampler = hyperparameter_sampler
        self.save_trial_info = save_trial_info
        self.output_dir = rio.create_timestamped_dir(
            parent_path=ATTACK_HYPERPARAMETER_TUNING
        )

    def build_attack_driver(self, trial: optuna.Trial) -> AttackDriver:
        settings = AttackHyperParameterSettings.from_optuna_active_trial(
            trial=trial, tuning_ranges=self.tuning_ranges
        )
        attack_driver = AttackDriver(
            device=self.device,
            model_path=self.model_path,
            # checkpoint_path=self.checkpoint_path,
            checkpoint=self.checkpoint,
            epochs_per_batch=self.epoch_per_batch,
            batch_size=2**settings.log_batch_size,
            kappa=settings.kappa,
            lambda_1=settings.lambda_1,
            optimizer_constructor=getattr(
                torch.optim, settings.optimizer_name
            ),
            optimizer_constructor_kwargs={"lr": settings.learning_rate},
            max_num_samples=self.max_num_samples,
            sample_selection_seed=self.sample_selection_seed,
            save_train_result=True,
            output_dir=self.output_dir,
        )

        return attack_driver

    def objective_fn(self, trial) -> float:
        attack_driver = self.build_attack_driver(trial=trial)
        trainer_result = attack_driver()
        success_summary = TrainerSuccessSummary(trainer_result=trainer_result)

        if len(success_summary.best_perts_summary.sparse_small_scores) == 0:
            return 0.0
        else:
            return torch.sum(
                success_summary.best_perts_summary.sparse_small_scores
            ).item()

    def export_study(self, study: optuna.Study):
        study_filename = "optuna_study.pickle"
        study_export_path = self.output_dir / study_filename
        rio.ResourceExporter().export(resource=study, path=study_export_path)

    def tune(
        self, num_trials: int, timeout: int | None = None
    ) -> optuna.Study:
        study = optuna.create_study(
            direction="maximize", sampler=self.hyperparameter_sampler
        )
        study.optimize(
            func=self.objective_fn, n_trials=num_trials, timeout=timeout
        )
        self.export_study(study=study)
        return study



