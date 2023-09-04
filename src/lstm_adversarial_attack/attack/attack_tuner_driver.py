import sys
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Any

import optuna
import torch

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.attack.attack_data_structs as ads
import lstm_adversarial_attack.attack.attack_hyperparameter_tuner as aht
import lstm_adversarial_attack.config_paths as cfg_paths
import lstm_adversarial_attack.config_settings as cfg_settings
import lstm_adversarial_attack.data_provenance as dpr
import lstm_adversarial_attack.data_structures as ds
import lstm_adversarial_attack.preprocess.encode_decode as edc
import lstm_adversarial_attack.tune_train.model_retriever as tmr
import lstm_adversarial_attack.tune_train.tuner_helpers as tuh
import lstm_adversarial_attack.tuning_db.tuning_studies_database as tsd
import lstm_adversarial_attack.preprocess.encode_decode_structs as eds


class AttackTunerDriver(dpr.HasDataProvenance):
    """
    Instantiates and runs (or re-starts) an AttackHyperParameterTuner
    """

    def __init__(
        self,
        device: torch.device,
        hyperparameters_path: Path | str,
        objective_name: str,
        objective_extra_kwargs: dict[str, Any] = None,
        db_env_var_name: str = "ATTACK_TUNING_DB_NAME",
        study_name: str = None,
        tuning_ranges: ads.AttackTuningRanges = None,
        epochs_per_batch: int = cfg_settings.ATTACK_TUNING_EPOCHS,
        max_num_samples: int = cfg_settings.ATTACK_TUNING_MAX_NUM_SAMPLES,
        sample_selection_seed: int = cfg_settings.ATTACK_SAMPLE_SELECTION_SEED,
        training_result_dir: Path | str = None,
        pruner_name: str = "MedianPruner",
        pruner_kwargs: dict[str, Any] = None,
        sampler_name: str = "TPESampler",
        sampler_kwargs: dict[str, Any] = None,
    ):
        """
        :param device: the device to run on
        :param objective_name: name of method in AttackTunerObjectives to user
        for computation of Optuna tuner objective_fn return value
        :param tuning_ranges: hyperparamter tuning ranges (for use by Optuna)
        specified, default is timestamped dir under
        data/attack/attack_hyperparamter_tuning
        """
        self.device = device
        self.hyperparameters_path = Path(hyperparameters_path)
        self.objective_name = objective_name
        self.objective_extra_kwargs = objective_extra_kwargs
        self.db_env_var_name = db_env_var_name
        if study_name is None:
            study_name = self.build_study_name()
        self.study_name = study_name
        self.output_dir = cfg_paths.ATTACK_HYPERPARAMETER_TUNING / study_name
        self.has_pre_existing_local_output = self.output_dir.exists()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if tuning_ranges is None:
            tuning_ranges = ads.AttackTuningRanges()
        self.tuning_ranges = tuning_ranges
        self.epochs_per_batch = epochs_per_batch
        self.max_num_samples = max_num_samples
        self.sample_selection_seed = sample_selection_seed
        self.training_result_dir = (
            Path(training_result_dir)
            if training_result_dir is not None
            else None
        )
        if pruner_kwargs is None:
            pruner_kwargs = {}
        self.pruner_kwargs = pruner_kwargs
        self.pruner = self.get_pruner(pruner_name=pruner_name)
        if sampler_kwargs is None:
            sampler_kwargs = {}
        self.sampler_kwargs = sampler_kwargs
        self.hyperparameter_sampler = self.get_sampler(
            sampler_name=sampler_name
        )
        # self.write_provenance()
        self.export(filename="attack_tuner_driver_dict.pickle")

    @property
    def summary(self) -> eds.AttackTunerDriverSummary:
        return eds.AttackTunerDriverSummary(
            hyperparameters_path=str(self.hyperparameters_path),
            objective_name=self.objective_name,
            objective_extra_kwargs=self.objective_extra_kwargs,
            db_env_var_name=self.db_env_var_name,
            study_name=self.study_name,
            is_continuation=self.has_pre_existing_local_output,
            tuning_ranges=self.tuning_ranges,
            epochs_per_batch=self.epochs_per_batch,
            max_num_samples=self.max_num_samples,
            sample_selection_seed=self.sample_selection_seed,
            training_result_dir=str(self.training_result_dir),
            pruner_name=self.pruner.__class__.__name__,
            pruner_kwargs=self.pruner_kwargs,
            sampler_name=self.hyperparameter_sampler.__class__.__name__,
            sampler_kwargs=self.sampler_kwargs,
        )

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

    @cached_property
    def db(self) -> tsd.OptunaDatabase:
        db_dotenv_info = tsd.get_db_dotenv_info(
            db_name_var=self.db_env_var_name
        )
        return tsd.OptunaDatabase(**db_dotenv_info)

    @cached_property
    def target_fold_checkpoint_pair(self) -> tmr.FoldCheckpointPair:
        return tmr.ModelRetriever(
            training_output_dir=self.training_result_dir
        ).get_representative_checkpoint()

    @property
    def target_fold_index(self) -> int:
        return self.target_fold_checkpoint_pair.fold

    @property
    def target_model_checkpoint(self) -> ds.TrainingCheckpoint:
        return self.target_fold_checkpoint_pair.checkpoint

    @staticmethod
    def build_study_name() -> str:
        timestamp = "".join(
            char for char in str(datetime.now()) if char.isdigit()
        )
        return f"attack_tuning_{timestamp}"

    def get_pruner(self, pruner_name: str) -> optuna.pruners.BasePruner:
        return getattr(optuna.pruners, pruner_name)(**self.pruner_kwargs)

    def get_sampler(self, sampler_name: str) -> optuna.samplers.BaseSampler:
        return getattr(optuna.samplers, sampler_name)(**self.sampler_kwargs)

    def run(self, num_trials: int) -> optuna.Study:
        """
        Instantiates and runs an AttackHyperParameterTuner
        :param num_trials:
        :return: an Optuna Study object (this also gets saved in .output_dir)
        """

        if not self.summary.is_continuation:
            timestamp = "".join(
                char for char in str(datetime.now()) if char.isdigit()
            )
            summary_output_path = (
                self.output_dir / f"attack_tuner_driver_summary_{timestamp}.json"
            )
            # summary_output_path = Path(f"attack_tuner_driver_summary_{timestamp}")

            edc.AttackTunerDriverSummaryWriter().export(
                obj=self.summary, path=summary_output_path
            )

        hyperparameters = (
            edc.X19LSTMHyperParameterSettingsReader().import_struct(
                path=self.hyperparameters_path
            )
        )

        model = tuh.X19LSTMBuilder(settings=hyperparameters).build()
        # TODO check really need to load state dict here (loading if not
        #  actually necessary unlikely to be a problem)
        model.load_state_dict(
            state_dict=self.target_model_checkpoint.state_dict
        )

        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.db.storage,
            load_if_exists=True,
            # TODO: don't hardcode direction
            direction="maximize",
            sampler=self.hyperparameter_sampler,
            pruner=self.pruner,
        )

        tuner = aht.AttackHyperParameterTuner(
            device=self.device,
            model=model,
            checkpoint=self.target_model_checkpoint,
            epochs_per_batch=cfg_settings.ATTACK_TUNING_EPOCHS,
            max_num_samples=cfg_settings.ATTACK_TUNING_MAX_NUM_SAMPLES,
            tuning_ranges=self.tuning_ranges,
            output_dir=self.output_dir,
            objective_name=self.objective_name,
            sample_selection_seed=self.sample_selection_seed,
            study=study,
        )

        return tuner.tune(num_trials=num_trials)