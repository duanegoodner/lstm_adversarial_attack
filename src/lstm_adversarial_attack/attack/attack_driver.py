import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.attack.adv_attack_trainer as aat
import lstm_adversarial_attack.attack.attack_data_structs as ads
import lstm_adversarial_attack.attack.attack_result_data_structs as ards
import lstm_adversarial_attack.model.cross_validation_summarizer as cvs
import lstm_adversarial_attack.model.model_retriever as tmr
import lstm_adversarial_attack.model.tuner_helpers as tuh
import lstm_adversarial_attack.tuning_db.tuning_studies_database as tsd
from lstm_adversarial_attack.config import CONFIG_READER
import lstm_adversarial_attack.dataset.x19_mort_general_dataset as xmd
from lstm_adversarial_attack.dataset.x19_mort_general_dataset import (
    X19MGeneralDatasetWithIndex,
    x19m_with_index_collate_fn,
)


class AttackDriver:
    """
    Instantiates and runs an AdversarialAttackTrainer
    """

    def __init__(
        self,
        device: torch.device,
        attack_tuning_id: str,
        attack_id: str,
        checkpoint_interval: int = None,
    ):
        """
        :param device: device to run on
        :param checkpoint_interval: number of batches per checkpoint
        """
        self.device = device
        self.attack_tuning_id = attack_tuning_id
        self.attack_id = attack_id
        self.settings = ads.AttackDriverSettings.from_config()
        if self.settings.sample_selection_seed is not None:
            torch.manual_seed(self.settings.sample_selection_seed)
        self.paths = ads.AttackDriverPaths.from_config()
        self.collate_fn = x19m_with_index_collate_fn
        self.checkpoint_interval = checkpoint_interval
        self.initialize_output_dir()

    @property
    def attack_tuner_driver_summary(self) -> ads.AttackTunerDriverSummary:
        return ads.ATTACK_TUNER_DRIVER_SUMMARY_IO.import_to_struct(
            path=Path(CONFIG_READER.read_path("attack.tune.output_dir"))
            / self.attack_tuning_id
            / f"attack_tuner_driver_summary_{self.attack_tuning_id}.json"
        )

    @property
    def attack_tuning_study_name(self) -> str:
        return f"attack_tuning_{self.attack_tuning_id}"

    @property
    def model_hyperparameters(self) -> tuh.X19LSTMHyperParameterSettings:
        return self.attack_tuner_driver_summary.model_hyperparameters

    @property
    def attack_hyperparameters(self) -> ads.AttackHyperParameterSettings:
        attack_hyperparameters_dict = tsd.ATTACK_TUNING_DB.get_best_params(
            study_name=self.attack_tuning_study_name
        )
        return ads.AttackHyperParameterSettings(**attack_hyperparameters_dict)

    @property
    def target_model_checkpoint_info(self) -> cvs.CheckpointInfo:
        return tmr.ModelRetriever(
            training_output_dir=Path(
                self.attack_tuner_driver_summary.model_training_result_dir
            )
        ).get_representative_checkpoint()

    @property
    def preprocess_id(self) -> str:
        return self.attack_tuner_driver_summary.preprocess_id

    @property
    def output_dir(self) -> Path:
        return Path(self.paths.output_dir) / self.attack_id

    @property
    def summary(self) -> ads.AttackDriverSummary:
        return ads.AttackDriverSummary(
            preprocess_id=self.preprocess_id,
            model_tuning_id=self.attack_tuner_driver_summary.model_tuning_id,
            cv_training_id=self.attack_tuner_driver_summary.cv_training_id,
            attack_tuning_id=self.attack_tuning_id,
            attack_id=self.attack_id,
            settings=self.settings,
            paths=self.paths,
            model_hyperparameters=self.model_hyperparameters,
            attack_hyperparameters=self.attack_hyperparameters,
        )

    def initialize_output_dir(self):
        """
        Creates directory where results of attacked will be saved
        :return: path to output dir (either same as output_dir in arg, or
        path to newly create directory)
        """
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def __call__(
        self,
    ) -> aat.AdversarialAttackTrainer | ards.AttackTrainerResult:
        """
        Imports model to attack, then trains and runs attack driver
        :return: AttackTrainerResult (dataclass with attack results)
        """

        ads.ATTACK_DRIVER_SUMMARY_IO.export(
            obj=self.summary,
            path=self.output_dir
            / f"attack_driver_summary_{self.attack_id}.json",
        )

        checkpoint = self.target_model_checkpoint_info.checkpoint
        model = tuh.X19LSTMBuilder(settings=self.model_hyperparameters).build()
        model.load_state_dict(state_dict=checkpoint.state_dict)

        attack_trainer = aat.AdversarialAttackTrainer(
            device=self.device,
            model=model,
            state_dict=checkpoint.state_dict,
            attack_hyperparameters=self.attack_hyperparameters,
            epochs_per_batch=self.settings.epochs_per_batch,
            dataset=(
                X19MGeneralDatasetWithIndex.from_feature_finalizer_output(
                    preprocess_id=self.preprocess_id,
                    max_num_samples=self.settings.max_num_samples,
                    random_seed=self.settings.sample_selection_seed,
                )
            ),
            collate_fn=self.collate_fn,
            attack_misclassified_samples=self.settings.attack_misclassified_samples,
            output_dir=self.output_dir,
            checkpoint_interval=self.checkpoint_interval,
        )

        attack_trainer_result = attack_trainer.train_attacker()

        attack_trainer_result_dto = ards.AttackTrainerResultDTO(
            dataset_info=xmd.X19MGeneralDatasetInfo(
                preprocess_id=self.preprocess_id,
                max_num_samples=self.settings.max_num_samples,
                random_seed=self.settings.sample_selection_seed,
            ),
            dataset_indices=attack_trainer_result.dataset_indices,
            epochs_run=attack_trainer_result.epochs_run,
            input_seq_lengths=attack_trainer_result.input_seq_lengths,
            first_examples=attack_trainer_result.first_examples,
            best_examples=attack_trainer_result.best_examples,
        )

        train_result_output_path = (
            self.output_dir / f"final_attack_result_{self.attack_id}.json"
        )
        ards.ATTACK_TRAINER_RESULT_IO.export(
            obj=attack_trainer_result_dto, path=train_result_output_path
        )

        return attack_trainer_result
