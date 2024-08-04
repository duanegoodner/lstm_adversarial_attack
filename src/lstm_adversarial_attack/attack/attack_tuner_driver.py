import pprint
import sys
from functools import cached_property
from pathlib import Path

import optuna
import torch

sys.path.append(str(Path(__file__).parent.parent.parent))
# AttackTunerDriverSettings and AttackTunerDriverPaths moved to attack_data_structs when fixing (de)serialization
import lstm_adversarial_attack.attack.attack_data_structs as ads
import lstm_adversarial_attack.attack.attack_tuner as atn
import lstm_adversarial_attack.model.cross_validation_summarizer as cvs
import lstm_adversarial_attack.model.model_data_structs as mds
import lstm_adversarial_attack.model.model_retriever as tmr
import lstm_adversarial_attack.model.tuner_helpers as tuh
import lstm_adversarial_attack.tuning_db.tuning_studies_database as tsd
import lstm_adversarial_attack.utils.redirect_output as rdo
from lstm_adversarial_attack.config import CONFIG_READER
from lstm_adversarial_attack.dataset.x19_mort_general_dataset import (
    X19MGeneralDatasetWithIndex,
)


class AttackTunerDriver:
    """
    Instantiates and runs (or re-starts) an AttackTuner
    """

    def __init__(
        self,
        device: torch.device,
        cv_training_id: str,
        attack_tuning_id: str,
        redirect_terminal_output: bool,
    ):
        """
        :param device: the device to run on
        :param tuning_ranges: hyperparamter tuning ranges (for use by Optuna)
        specified, default is timestamped dir under
        data/attack/attack_hyperparamter_tuning
        """
        self.device = device
        self.cv_training_id = cv_training_id
        self.attack_tuning_id = attack_tuning_id
        self.redirect_terminal_output = redirect_terminal_output
        self.settings = ads.AttackTunerDriverSettings.from_config()
        self.paths = ads.AttackTunerDriverPaths.from_config()
        self.objective_extra_kwargs = (
            {"max_perts": self.settings.max_perts}
            if self.settings.objective_name == "max_num_nonzero_perts"
            else {}
        )
        self.output_dir = Path(self.paths.output_dir) / self.attack_tuning_id
        self.has_pre_existing_local_output = self.output_dir.exists()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tuning_ranges = ads.AttackTuningRanges()
        self.pruner_kwargs = self.settings.pruner_kwargs
        self.pruner = self.get_pruner(pruner_name=self.settings.pruner_name)
        self.sampler_kwargs = self.settings.sampler_kwargs
        self.hyperparameter_sampler = self.get_sampler(
            sampler_name=self.settings.sampler_name
        )

    @property
    def model_training_result_dir(self) -> Path:
        cv_output_root = Path(
            CONFIG_READER.read_path("model.cv_driver.output_dir")
        )
        return cv_output_root / self.cv_training_id

    @property
    def cross_validator_driver_summary(
        self,
    ) -> mds.CrossValidatorDriverSummary:
        cv_driver_summary_path = (
            self.model_training_result_dir
            / f"cross_validator_driver_summary_{self.cv_training_id}.json"
        )

        return mds.CROSS_VALIDATOR_DRIVER_SUMMARY_IO.import_to_struct(
            path=cv_driver_summary_path
        )

    @property
    def preprocess_id(self) -> str:
        return self.cross_validator_driver_summary.preprocess_id

    @property
    def model_tuning_id(self) -> str:
        return self.cross_validator_driver_summary.model_tuning_id

    @property
    def model_hyperparameters(self) -> tuh.X19LSTMHyperParameterSettings:
        return self.cross_validator_driver_summary.model_hyperparameters

    @classmethod
    def from_attack_tuning_id(
        cls, attack_tuning_id: str, device: torch.device, redirect_terminal_output: bool
    ):
        attack_tuning_output_root = Path(
            CONFIG_READER.read_path("attack.tune.output_dir")
        )
        attack_tuner_driver_summary = (
            ads.ATTACK_TUNER_DRIVER_SUMMARY_IO.import_to_struct(
                path=attack_tuning_output_root
                / attack_tuning_id
                / f"attack_tuner_driver_summary_{attack_tuning_id}.json"
            )
        )
        return cls(
            device=device,
            cv_training_id=attack_tuner_driver_summary.cv_training_id,
            attack_tuning_id=attack_tuner_driver_summary.attack_tuning_id,
            redirect_terminal_output=redirect_terminal_output,
        )

    @property
    def summary(self) -> ads.AttackTunerDriverSummary:
        return ads.AttackTunerDriverSummary(
            preprocess_id=self.preprocess_id,
            model_tuning_id=self.model_tuning_id,
            cv_training_id=self.cv_training_id,
            attack_tuning_id=self.attack_tuning_id,
            model_hyperparameters=self.model_hyperparameters,
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
    def target_model_checkpoint(self) -> mds.TrainingCheckpoint:
        return self.target_checkpoint_info.checkpoint

    @property
    def target_model_checkpoint_path(self) -> Path:
        return self.target_checkpoint_info.save_path

    @property
    def study_name(self) -> str:
        return f"attack_tuning_{self.attack_tuning_id}"

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
            summary_output_path = (
                self.output_dir
                / f"attack_tuner_driver_summary_{self.attack_tuning_id}.json"
            )
            mds.TUNER_DRIVER_SUMMARY_IO.export(
                obj=self.summary, path=summary_output_path
            )
            print(
                f"Starting new Attack Hyperparameter Tuning session "
                f"{self.attack_tuning_id}.\nUsing trained model from CV training "
                f"session ID {self.cv_training_id} as the attack target.\n"
            )
        if self.summary.is_continuation:
            print(
                f"Resuming Attack Hyperparameter Tuning session "
                f"{self.attack_tuning_id}.\n Using model from CV Training "
                f"session {self.cv_training_id} as the attack target."
            )

        print("Attack tuner driver settings:")
        pprint.pprint(self.settings.__dict__)
        print()

        model = tuh.X19LSTMBuilder(settings=self.model_hyperparameters).build()
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
            model_hyperparameters=self.model_hyperparameters,
            dataset=X19MGeneralDatasetWithIndex.from_feature_finalizer_output(
                preprocess_id=self.preprocess_id,
                max_num_samples=self.settings.max_num_samples,
            ),
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

        log_file_fid = None
        if self.redirect_terminal_output:
            log_file_path = (
                Path(
                    CONFIG_READER.read_path("redirected_output.attack_tuning")
                )
                / f"{self.attack_tuning_id}.log"
            )
            log_file_fid = rdo.set_redirection(
                log_file_path=log_file_path, include_optuna=True
            )
        completed_study = tuner.tune(num_trials=self.settings.num_trials)
        if self.redirect_terminal_output:
            log_file_fid.close()

        return completed_study

    def __call__(self) -> optuna.Study:

        completed_study = self.run()

        return completed_study
