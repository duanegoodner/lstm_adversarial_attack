import sys
from pathlib import Path
from typing import Any

import optuna
import torch

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.attack.attack_data_structs as ads
import lstm_adversarial_attack.attack.attack_hyperparameter_tuner as aht
import lstm_adversarial_attack.attack.model_retriever as amr
import lstm_adversarial_attack.config_paths as cfg_paths
import lstm_adversarial_attack.config_settings as cfg_settings
import lstm_adversarial_attack.data_provenance as dpr
import lstm_adversarial_attack.data_structures as ds
import lstm_adversarial_attack.preprocess.encode_decode as edc
import lstm_adversarial_attack.resource_io as rio
import lstm_adversarial_attack.tune_train.cross_validation_summarizer as cvs
import lstm_adversarial_attack.tune_train.tuner_helpers as tuh


class AttackTunerDriver(dpr.HasDataProvenance):
    """
    Instantiates and runs (or re-starts) an AttackHyperParameterTuner
    """

    def __init__(
        self,
        device: torch.device,
        hyperparameters_path: Path,
        # target_model_path: Path,
        objective_name: str,
        target_model_checkpoint: ds.TrainingCheckpoint,
        objective_extra_kwargs: dict[str, Any] = None,
        tuning_ranges: ads.AttackTuningRanges = None,
        output_dir: Path = None,
        epochs_per_batch: int = cfg_settings.ATTACK_TUNING_EPOCHS,
        max_num_samples: int = cfg_settings.ATTACK_TUNING_MAX_NUM_SAMPLES,
        sample_selection_seed: int = cfg_settings.ATTACK_SAMPLE_SELECTION_SEED,
        target_fold_index: int = None,
    ):
        """
        :param device: the device to run on
        :param target_model_path: path to .pickle file w/ model to attack
        :param objective_name: name of method in AttackTunerObjectives to user
        for computation of Optuna tuner objective_fn return value
        :param target_model_checkpoint: checkpoint file w/ params to load into
        model under attack
        :param tuning_ranges: hyperparamter tuning ranges (for use by Optuna)
        :param output_dir: directory where results will be saved. If not
        specified, default is timestamped dir under
        data/attack/attack_hyperparamter_tuning
        """
        self.device = device
        self.hyperparameters_path = hyperparameters_path
        # self.target_model_path = target_model_path
        self.objective_name = objective_name
        self.objective_extra_kwargs = objective_extra_kwargs
        self.target_model_checkpoint = target_model_checkpoint
        if tuning_ranges is None:
            tuning_ranges = ads.AttackTuningRanges()
        self.tuning_ranges = tuning_ranges
        if output_dir is None:
            output_dir = rio.create_timestamped_dir(
                parent_path=cfg_paths.ATTACK_HYPERPARAMETER_TUNING
            )
        self.epochs_per_batch = epochs_per_batch
        self.max_num_samples = max_num_samples
        self.output_dir = output_dir
        self.sample_selection_seed = sample_selection_seed
        self.target_fold_index = target_fold_index
        # self.write_provenance()
        self.export(filename="attack_tuner_driver_dict.pickle")

    @property
    def provenance_info(self) -> dpr.ProvenanceInfo:
        return dpr.ProvenanceInfo(
            category_name="attack_tuner_driver",
            new_items={
                "objective_name": self.objective_name,
                "objective_extra_kwargs": self.objective_extra_kwargs,
                "hyperparameters_path": self.hyperparameters_path,
                "target_fold_index": self.target_fold_index,
                "target_model_trained_to_epoch": (
                    self.target_model_checkpoint.epoch_num
                ),
            },
            output_dir=self.output_dir,
        )

    def run(self, num_trials: int) -> optuna.Study:
        """
        Instantiates and runs an AttackHyperParameterTuner
        :param num_trials:
        :return: an Optuna Study object (this also gets saved in .output_dir)
        """

        hyperparameters = (
            edc.X19LSTMHyperParameterSettingsReader().import_struct(
                path=self.hyperparameters_path
            )
        )

        model = tuh.X19LSTMBuilder(settings=hyperparameters).build()
        # model.load_state_dict(state_dict=)

        tuner = aht.AttackHyperParameterTuner(
            device=self.device,
            model=model,
            # model_path=self.target_model_path,
            checkpoint=self.target_model_checkpoint,
            epochs_per_batch=cfg_settings.ATTACK_TUNING_EPOCHS,
            max_num_samples=cfg_settings.ATTACK_TUNING_MAX_NUM_SAMPLES,
            tuning_ranges=self.tuning_ranges,
            output_dir=self.output_dir,
            objective_name=self.objective_name,
            sample_selection_seed=self.sample_selection_seed,
        )

        return tuner.tune(num_trials=num_trials)

    def restart(self, output_dir: Path, num_trials: int) -> optuna.Study:
        """
        Restarts tuning using params of self. Uses existing AttackDriver.
        Creates new AttackHyperParamterTuner
        :param output_dir: directory containing previous output and where new
        output will be written.
        :param num_trials: max number of trials to run (OK to stop early with
        CTRL-C since results get saved after each trial)
        :return: Optuna Study object
        """
        tuner = aht.AttackHyperParameterTuner(
            device=self.device,
            model_path=self.target_model_path,
            checkpoint=self.target_model_checkpoint,
            epochs_per_batch=self.epochs_per_batch,
            max_num_samples=self.max_num_samples,
            tuning_ranges=self.tuning_ranges,
            continue_study_path=output_dir / "optuna_study.pickle",
            output_dir=output_dir,
            objective_name=self.objective_name,
            sample_selection_seed=self.sample_selection_seed,
        )

        return tuner.tune(num_trials=num_trials)
