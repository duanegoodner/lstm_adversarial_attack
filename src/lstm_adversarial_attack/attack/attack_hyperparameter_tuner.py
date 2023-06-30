import numpy as np
import optuna
import torch
from optuna.pruners import BasePruner, MedianPruner
from optuna.samplers import BaseSampler, TPESampler
from pathlib import Path
from typing import Callable

import lstm_adversarial_attack.attack.attack as atk
import lstm_adversarial_attack.attack.attack_data_structs as ads
import lstm_adversarial_attack.attack.attack_result_data_structs as ards
import lstm_adversarial_attack.config_paths as cfg_paths
import lstm_adversarial_attack.resource_io as rio


class AttackTunerObjectivesBuilder:
    """
    Each method returns a Callable that can be to calculate the return value of
    an AttackHyperParameterTuner's .objective_fn method.
    """

    @staticmethod
    def sparse_small() -> Callable[[ards.TrainerSuccessSummary], float]:
        def objective(success_summary: ards.TrainerSuccessSummary) -> float:
            if (
                len(success_summary.examples_summary_best.sparse_small_scores)
                == 0
            ):
                return 0.0
            else:
                return np.sum(
                    success_summary.examples_summary_best.sparse_small_scores
                ).item()

        return objective

    @staticmethod
    def sparsity() -> Callable[[ards.TrainerSuccessSummary], float]:

        def objective(success_summary: ards.TrainerSuccessSummary) -> float:
            if len(success_summary.examples_summary_best.sparsity) == 0:
                return 0.0
            else:
                return np.sum(
                    success_summary.examples_summary_best.sparsity
                ).item()

        return objective

    @staticmethod
    def max_num_nonzero_perts(
        max_perts: int,
    ) -> Callable[[ards.TrainerSuccessSummary], float]:
        def objective(success_summary: ards.TrainerSuccessSummary) -> float:
            return success_summary.examples_summary_best.num_examples_with_num_nonzero_less_than(
                cutoff=(max_perts + 1)
            )
        return objective


class AttackHyperParameterTuner:
    def __init__(
        self,
        device: torch.device,
        model_path: Path,
        checkpoint: dict,
        epochs_per_batch: int,
        max_num_samples: int,
        tuning_ranges: ads.AttackTuningRanges,
        objective: Callable[[ards.TrainerSuccessSummary], float],
        sample_selection_seed: int = 13579,
        pruner: BasePruner = MedianPruner(),
        hyperparameter_sampler: BaseSampler = TPESampler(),
        output_dir: Path = None,
        continue_study_path: Path = None,
    ):
        self.device = device
        self.model_path = model_path
        self.checkpoint = checkpoint
        self.epoch_per_batch = epochs_per_batch
        self.max_num_samples = max_num_samples
        self.tuning_ranges = tuning_ranges
        self.objective = objective
        self.sample_selection_seed = sample_selection_seed
        self.pruner = pruner
        self.hyperparameter_sampler = hyperparameter_sampler
        self.output_dir, self.attack_results_dir = self.initialize_output_dir(
            output_dir=output_dir
        )
        self.continue_study_path = continue_study_path

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
            result_file_prefix=f"_trial_{trial.number}",
        )

        return attack_driver

    def objective_fn(self, trial) -> float:
        attack_driver = self.build_attack_driver(trial=trial)
        trainer_result = attack_driver()
        success_summary = ards.TrainerSuccessSummary(
            trainer_result=trainer_result
        )

        return self.objective(success_summary)

        # return success_summary.examples_summary_best.num_examples_with_num_nonzero_less_than(
        #     cutoff=2
        # )

    def export_study(self, study: optuna.Study):
        study_filename = "optuna_study.pickle"
        study_export_path = self.output_dir / study_filename
        rio.ResourceExporter().export(resource=study, path=study_export_path)

    def tune(
        self, num_trials: int, timeout: int | None = None
    ) -> optuna.Study:
        if self.continue_study_path is not None:
            study = rio.ResourceImporter().import_pickle_to_object(
                path=self.continue_study_path
            )
        else:
            study = optuna.create_study(
                direction="maximize", sampler=self.hyperparameter_sampler
            )
        for trial_num in range(num_trials):
            study.optimize(func=self.objective_fn, n_trials=1, timeout=timeout)
            self.export_study(study=study)
        return study
