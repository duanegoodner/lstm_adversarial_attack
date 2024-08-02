from pathlib import Path
from typing import Any

import numpy as np
import optuna
import torch
import torch.nn as nn

import lstm_adversarial_attack.attack.adv_attack_trainer as ata
# import lstm_adversarial_attack.attack.attack_driver as ad
import lstm_adversarial_attack.attack.attack_data_structs as ads
import lstm_adversarial_attack.attack.attack_result_data_structs as ards
import lstm_adversarial_attack.config_paths as cfg_paths
import lstm_adversarial_attack.utils.resource_io as rio
# import lstm_adversarial_attack.data_structures as ds
import lstm_adversarial_attack.model.model_data_structs as mds
import lstm_adversarial_attack.model.tuner_helpers as tuh
from lstm_adversarial_attack.dataset.x19_mort_general_dataset import (
    X19MGeneralDatasetWithIndex,
    x19m_with_index_collate_fn,
)


class AttackTunerObjectives:
    """
    Contains methods available to HyperParameterAttackTuner for calculating
    return value of tuner objective_fn from a TrainerSuccessSummary. Each method
    uses different attribute of a TrainerSuccessSummary for its calc.
    """

    @staticmethod
    def sparse_small(success_summary: ards.TrainerSuccessSummary) -> float:
        """
        Uses TrainerSuccessSummary object's sparse_small_scores
        :param success_summary: a TrainerSuccessSummary object obtained from
        attacks from a particular tuning trial
        :return: sum of sparse_small scores of adversarial perturbations of
        all successful attacks
        """

        if len(success_summary.perts_summary_best.sparse_small_scores) == 0:
            return 0.0
        else:
            return np.sum(
                success_summary.perts_summary_best.sparse_small_scores
            ).item()

    @staticmethod
    def sparse_small_max(success_summary: ards.TrainerSuccessSummary) -> float:
        """
        Uses TrainerSuccessSummary object's sparse_small_max_scores
        :param success_summary: a TrainerSuccessSummary object obtained from
        attacks from a particular tuning trial
        :return: sum of spars_small_max_scores of adversarial perturbations for
        all successful attacks
        """
        if (
            len(success_summary.perts_summary_best.sparse_small_max_scores)
            == 0
        ):
            return 0.0
        else:
            return np.sum(
                success_summary.perts_summary_best.sparse_small_max_scores
            ).item()

    @staticmethod
    def sparsity(success_summary: ards.TrainerSuccessSummary) -> float:
        """
        Uses TrainerSuccessSummary object's sparsity attribute
        :param success_summary: a TrainerSuccessSummary object obtained from
        attacks from a particular tuning trial
        :return: sum of sparsity of adversarial perturbations for
        all successful attacks
        """
        if len(success_summary.perts_summary_best.sparsity) == 0:
            return 0.0
        else:
            return np.sum(success_summary.perts_summary_best.sparsity).item()

    @staticmethod
    def max_num_nonzero_perts(
        success_summary: ards.TrainerSuccessSummary, max_num_perts: int
    ) -> float:
        """
        Returns number of adversarial perturbations with total number of
        non-zero elements less than or equal to max_num_non_perts.
        :param success_summary: a TrainerSuccessSummary object obtained from
        attacks from a particular tuning trial
        :param max_num_perts:
        :return: Number of adversarial perturbations with total number of
        non-zero elements less than or equal to max_num_non_perts.
        """
        return success_summary.perts_summary_best.num_examples_with_num_nonzero_less_than(
            cutoff=(max_num_perts + 1)
        )


class AttackTuner:
    """
    Tunes hyperparameters of AdversarialAttackTrainers and their associated
    Adversarial Attackers. (Each test of a new set of params instantiates new
    AttackDriver, AdversarialAttackTrainer and model)
    """

    def __init__(
        self,
        device: torch.device,
        model_hyperparameters: tuh.X19LSTMHyperParameterSettings,
        dataset: X19MGeneralDatasetWithIndex,
        model: nn.Module,
        checkpoint: mds.TrainingCheckpoint,
        epochs_per_batch: int,
        max_num_samples: int,
        tuning_ranges: ads.AttackTuningRanges,
        objective_name: str,
        sample_selection_seed: int,
        study: optuna.Study,
        objective_extra_kwargs: dict[str, Any] = None,
        output_dir: Path = None,
        continue_study_path: Path = None,
        attack_misclassifiied_samples: bool = False,
    ):
        """
        :param device: device to run on
        :param checkpoint: params (from prev training) to load into model
        :param epochs_per_batch: number of times to run attack algo (run by
        AdversarialAttackTrainer / Adversarial Attacker) runs on each batch
        :param max_num_samples: Number candidate samples to take from a dataset
        for attack. Default behavior of AdversarialAttackTrainer is to not
        attack samples misclassified by target model, so not all candidat
        samples get attacked.
        :param tuning_ranges: parameter ranges to explore during tuning
        :param sample_selection_seed: random seed set
        :param output_dir: (optional) Directory where trial results are saved.
        New directory automatically created if not specified.
        :param continue_study_path: (optional) pickle of pre-existing study to
        add on to
        """
        self.device = device
        self.model_hyperparameters = model_hyperparameters
        self.model = model
        self.checkpoint = checkpoint
        self.epoch_per_batch = epochs_per_batch
        self.max_num_samples = max_num_samples
        self.tuning_ranges = tuning_ranges
        self.objective_name = objective_name
        self.objective = getattr(AttackTunerObjectives, self.objective_name)
        if objective_extra_kwargs is None:
            objective_extra_kwargs = {}
        self.objective_extra_kwargs = objective_extra_kwargs
        self.sample_selection_seed = sample_selection_seed
        self.dataset = dataset
        self.study = study
        self.output_dir, self.attack_results_dir = self.initialize_output_dir(
            output_dir=output_dir
        )
        self.continue_study_path = continue_study_path
        self.attack_misclassified_samples = attack_misclassifiied_samples

    # TODO Simplify this method with `parents` and `exist_ok` args of mkdir().
    def initialize_output_dir(
        self, output_dir: Path = None
    ) -> tuple[Path, Path]:
        """
        :param output_dir: Directory where tuning results are saved. If None,
        a new directory gets created
        :return: Tuple of the output directory and its subdirectory. Subdir
        will hold trial results. Parent store .pickles of TunerDriver and
        Optuna Study.
        """
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

    def create_attack_trainer(
        self, trial: optuna.Trial
    ) -> ata.AdversarialAttackTrainer:

        settings = ads.BuildAttackHyperParameterSettings.from_optuna_trial(
            trial=trial, tuning_ranges=self.tuning_ranges
        )

        attack_trainer = ata.AdversarialAttackTrainer(
            device=self.device,
            model=self.model,
            attack_hyperparameters=settings,
            state_dict=self.checkpoint.state_dict,
            epochs_per_batch=self.epoch_per_batch,
            dataset=self.dataset,
            collate_fn=x19m_with_index_collate_fn,
            attack_misclassified_samples=self.attack_misclassified_samples,
            output_dir=self.output_dir,
        )

        return attack_trainer

    def objective_fn(self, trial: optuna.Trial) -> float:
        """
        Runs attack for a single trial. Result gets converted to scalar that
        Optuna uses to quantify effectiveness of attack.
        :param trial: current Optuna trial.
        :return: value quantifying success of attack
        """

        attack_trainer = self.create_attack_trainer(trial=trial)
        attack_trainer_result = attack_trainer.train_attacker()
        success_summary = ards.TrainerSuccessSummary(
            attack_trainer_result=attack_trainer_result
        )

        return self.objective(
            success_summary=success_summary, **self.objective_extra_kwargs
        )

    def tune(
        self, num_trials: int, timeout: int | None = None
    ) -> optuna.Study:
        """
        Runs an optuna Study consisting of Trials (one set of hyperparameters
        per trial)
        :param num_trials: number of trials to run
        :param timeout: max time for study (default of None means no limit)
        :return: an optuna Study object
        """
        for trial_num in range(num_trials):
            self.study.optimize(
                func=self.objective_fn, n_trials=1, timeout=timeout
            )
        return self.study
