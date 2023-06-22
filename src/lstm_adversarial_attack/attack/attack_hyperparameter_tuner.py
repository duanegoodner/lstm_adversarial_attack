import optuna
import torch
from optuna.pruners import BasePruner, MedianPruner
from optuna.samplers import BaseSampler, TPESampler
from pathlib import Path

import lstm_adversarial_attack.attack.attack as atk
import lstm_adversarial_attack.attack.attack_data_structs as ads
import lstm_adversarial_attack.attack.attack_result_data_structs as ards
import lstm_adversarial_attack.config_paths as cfg_paths
import lstm_adversarial_attack.resource_io as rio


class AttackHyperParameterTuner:
    def __init__(
        self,
        device: torch.device,
        model_path: Path,
        checkpoint: dict,
        epochs_per_batch: int,
        max_num_samples: int,
        tuning_ranges: ads.AttackTuningRanges,
        sample_selection_seed: int = 13579,
        pruner: BasePruner = MedianPruner(),
        hyperparameter_sampler: BaseSampler = TPESampler(),
        output_dir: Path = None,
    ):
        self.device = device
        self.model_path = model_path
        self.checkpoint = checkpoint
        self.epoch_per_batch = epochs_per_batch
        self.max_num_samples = max_num_samples
        self.tuning_ranges = tuning_ranges
        self.sample_selection_seed = (sample_selection_seed,)
        self.pruner = pruner
        self.hyperparameter_sampler = hyperparameter_sampler
        self.output_dir, self.attack_results_dir = self.initialize_output_dir(
            output_dir=output_dir
        )

    def initialize_output_dir(
        self, output_dir: Path = None
    ) -> tuple[Path, Path]:
        if output_dir is None:
            initialized_output_dir = rio.create_timestamped_dir(
                parent_path=cfg_paths.ATTACK_HYPERPARAMETER_TUNING
            )
        else:
            initialized_output_dir = output_dir
            if not initialized_output_dir.exists():
                initialized_output_dir.mkdir()

        rio.ResourceExporter().export(
            resource=self,
            path=initialized_output_dir / "attack_hyperparameter_tuner.pickle",
        )

        attack_results_dir = initialized_output_dir / "attack_trial_results"
        if not attack_results_dir.exists():
            attack_results_dir.mkdir()

        return initialized_output_dir, attack_results_dir

    def build_attack_driver(self, trial: optuna.Trial) -> atk.AttackDriver:
        settings = ads.AttackHyperParameterSettings.from_optuna_active_trial(
            trial=trial, tuning_ranges=self.tuning_ranges
        )
        attack_driver = atk.AttackDriver(
            device=self.device,
            model_path=self.model_path,
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
            output_dir=self.attack_results_dir,
            result_file_prefix=f"trial_{trial.number}",
        )

        return attack_driver

    def objective_fn(self, trial) -> float:
        attack_driver = self.build_attack_driver(trial=trial)
        trainer_result = attack_driver()
        success_summary = ards.TrainerSuccessSummary(
            trainer_result=trainer_result
        )

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
        for trial_num in range(num_trials):
            study.optimize(func=self.objective_fn, n_trials=1, timeout=timeout)
            self.export_study(study=study)
        return study
