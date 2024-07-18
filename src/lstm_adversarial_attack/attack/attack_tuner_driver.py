import sys
from datetime import datetime
from functools import cached_property
from pathlib import Path

import optuna
import torch

sys.path.append(str(Path(__file__).parent.parent.parent))
# AttackTunerDriverSettings and AttackTunerDriverPaths moved to attack_data_structs when fixing (de)serialization
import lstm_adversarial_attack.attack.attack_data_structs as ads
import lstm_adversarial_attack.attack.attack_tuner as atn
import lstm_adversarial_attack.data_structures as ds
import lstm_adversarial_attack.model.cross_validation_summarizer as cvs
import lstm_adversarial_attack.model.model_retriever as tmr
import lstm_adversarial_attack.model.tuner_helpers as tuh
import lstm_adversarial_attack.preprocess.encode_decode as edc
import lstm_adversarial_attack.preprocess.encode_decode_structs as eds
import lstm_adversarial_attack.tuning_db.tuning_studies_database as tsd


class AttackTunerDriver:
    """
    Instantiates and runs (or re-starts) an AttackTuner
    """

    def __init__(
        self,
        device: torch.device,
        settings: ads.AttackTunerDriverSettings,
        paths: ads.AttackTunerDriverPaths,
        study_name: str = None,
        tuning_ranges: ads.AttackTuningRanges = None,
        model_training_result_dir: Path | str = None,
    ):
        """
        :param device: the device to run on
        :param tuning_ranges: hyperparamter tuning ranges (for use by Optuna)
        specified, default is timestamped dir under
        data/attack/attack_hyperparamter_tuning
        """
        self.device = device
        self.settings = settings
        self.paths = paths
        # self.hyperparameters_path = Path(hyperparameters_path)
        self.objective_extra_kwargs = (
            {"max_perts": settings.max_perts}
            if self.settings.objective_name == "max_num_nonzero_perts"
            else {}
        )
        if study_name is None:
            study_name = self.build_study_name()
        self.study_name = study_name
        self.output_dir = Path(self.paths.output_dir) / study_name
        self.has_pre_existing_local_output = self.output_dir.exists()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if tuning_ranges is None:
            tuning_ranges = ads.AttackTuningRanges()
        self.tuning_ranges = tuning_ranges
        self.model_training_result_dir = (
            Path(model_training_result_dir)
            if model_training_result_dir is not None
            else None
        )
        self.pruner_kwargs = self.settings.pruner_kwargs
        self.pruner = self.get_pruner(pruner_name=self.settings.pruner_name)
        self.sampler_kwargs = self.settings.sampler_kwargs
        self.hyperparameter_sampler = self.get_sampler(
            sampler_name=self.settings.sampler_name
        )
        self.save_model_hyperparameters()

    @property
    def hyperparameters_path(self) -> Path:
        return self.model_training_result_dir / "hyperparameters.json"

    def save_model_hyperparameters(self):
        model_hyperparameters = (
            edc.X19LSTMHyperParameterSettingsReader().import_struct(
                path=self.hyperparameters_path
            )
        )
        edc.X19LSTMHyperParameterSettingsWriter().export(
            obj=model_hyperparameters,
            path=self.output_dir / "model_hyperparameters.json",
        )

    @property
    def summary(self) -> eds.AttackTunerDriverSummary:
        return eds.AttackTunerDriverSummary(
            settings=self.settings.__dict__,
            paths=self.paths.__dict__,
            study_name=self.study_name,
            is_continuation=self.has_pre_existing_local_output,
            tuning_ranges=self.tuning_ranges,
            model_training_result_dir=str(self.model_training_result_dir),
        )

    @cached_property
    def db(self) -> tsd.OptunaDatabase:
        db_dotenv_info = tsd.get_db_dotenv_info(
            db_name_var=self.settings.db_env_var_name
        )
        return tsd.OptunaDatabase(**db_dotenv_info)

    # TODO Move this to be part of __init__ and self.summary
    @cached_property
    def target_checkpoint_info(self) -> cvs.CheckpointInfo:
        return tmr.ModelRetriever(
            training_output_dir=self.model_training_result_dir
        ).get_representative_checkpoint()

    @property
    def target_fold_index(self) -> int:
        return self.target_checkpoint_info.fold

    @property
    def target_model_checkpoint(self) -> ds.TrainingCheckpoint:
        return self.target_checkpoint_info.checkpoint

    @property
    def target_model_checkpoint_path(self) -> Path:
        return self.target_checkpoint_info.save_path

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

    def run(self) -> optuna.Study:
        """
        Instantiates and runs an AttackTuner
        :return: an Optuna Study object (this also gets saved in .output_dir)
        """

        if not self.summary.is_continuation:
            timestamp = "".join(
                char for char in str(datetime.now()) if char.isdigit()
            )
            summary_output_path = (
                self.output_dir
                / f"attack_tuner_driver_summary_{timestamp}.json"
            )
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

        tuner = atn.AttackTuner(
            device=self.device,
            model_hyperparameters=hyperparameters,
            model=model,
            checkpoint=self.target_model_checkpoint,
            epochs_per_batch=self.settings.epochs_per_batch,
            max_num_samples=self.settings.max_num_samples,
            tuning_ranges=self.tuning_ranges,
            output_dir=self.output_dir,
            objective_name=self.settings.objective_name,
            sample_selection_seed=self.settings.sample_selection_seed,
            study=study,
        )

        return tuner.tune(num_trials=self.settings.num_trials)
