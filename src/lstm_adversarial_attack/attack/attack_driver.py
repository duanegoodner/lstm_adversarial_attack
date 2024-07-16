import sys
from functools import cached_property
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.attack.adv_attack_trainer as aat
import lstm_adversarial_attack.attack.attack_data_structs as ads
import lstm_adversarial_attack.attack.attack_result_data_structs as ards
import lstm_adversarial_attack.attack.attack_tuner_driver as atd
import lstm_adversarial_attack.config_paths as cfg_paths
import lstm_adversarial_attack.config_settings as cfg_settings
import lstm_adversarial_attack.path_searches as ps
import lstm_adversarial_attack.preprocess.encode_decode as edc
import lstm_adversarial_attack.preprocess.encode_decode_structs as eds
import lstm_adversarial_attack.resource_io as rio
from lstm_adversarial_attack.x19_mort_general_dataset import (
    X19MGeneralDatasetWithIndex,
    x19m_with_index_collate_fn,
)
import lstm_adversarial_attack.data_structures as ds
import lstm_adversarial_attack.model.tuner_helpers as tuh
import lstm_adversarial_attack.model.cross_validation_summarizer as cvs
import lstm_adversarial_attack.model.model_retriever as tmr


class AttackDriver:
    """
    Instantiates and runs an AdversarialAttackTrainer
    """

    def __init__(
        self,
        attack_tuner_driver: atd.AttackTunerDriver,
        device: torch.device,
        # model_hyperparameters_path: Path | str,
        attack_tuning_study_name: str,
        model_hyperparameters: tuh.X19LSTMHyperParameterSettings,
        attack_hyperparameters: ads.AttackHyperParameterSettings,
        epochs_per_batch: int = cfg_settings.ATTACK_TUNING_EPOCHS,
        db_env_var_name: str = "ATTACK_TUNING_DB_NAME",
        max_num_samples=cfg_settings.ATTACK_TUNING_MAX_NUM_SAMPLES,
        sample_selection_seed=cfg_settings.ATTACK_SAMPLE_SELECTION_SEED,
        attack_misclassified_samples: bool = False,
        output_dir: Path = None,
        result_file_prefix: str = "",
        save_attack_driver: bool = False,
        checkpoint_interval: int = None,
        hyperparameter_tuning_result_dir: Path = None,
    ):
        """
        :param device: device to run on
        # :param checkpoint: Info saved during training classifier. Contents
        include model params.
        :param epochs_per_batch: number of attack iterations per batch
        (https://arxiv.org/abs/1802.04822). Defines a margin by which alternate
        class logit value needs to exceed original class logit value in order
        to reduce loss function.
        searching for adversarial example_data
        constructor
        :param max_num_samples: Number candidate samples to take from a dataset
        for attack. Default behavior of AdversarialAttackTrainer is to not
        attack samples misclassified by target model, so not all candidate
        samples get attacked.
        :param sample_selection_seed: random seed to use when selecting subset
        of samples from original dataset
        :param attack_misclassified_samples: whether to run attacks on samples
        that original model misclassifies
        :param output_dir: directory where attack results are saved
        :param result_file_prefix: prefix to use in result file output
        :param save_attack_driver: whether to save AttackDriver .pickle
        :param checkpoint_interval: number of batches per checkpoint
        """
        self.attack_tuner_driver = attack_tuner_driver
        self.device = device
        self.attack_tuning_study_name = attack_tuning_study_name
        self.db_env_var_name = db_env_var_name
        # self.model_hyperparameters_path = Path(model_hyperparameters_path)
        self.model_hyperparameters = model_hyperparameters
        # self.checkpoint = checkpoint
        self.attack_hyperparameters = attack_hyperparameters
        self.epochs_per_batch = epochs_per_batch
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
        self.checkpoint_interval = checkpoint_interval
        self.output_dir = self.initialize_output_dir(output_dir=output_dir)
        self.hyperparameter_tuning_results_dir = (
            hyperparameter_tuning_result_dir
        )

    @staticmethod
    def initialize_output_dir(output_dir: Path | None):
        """
        Initializes directory where results of attacked will be saved
        :param output_dir: Path of output directory. If None, a directory
        will be created
        :return: path to output dir (either same as output_dir in arg, or
        path to newly create directory)
        """
        if output_dir is None:
            output_dir = rio.create_timestamped_dir(
                parent_path=cfg_paths.FROZEN_HYPERPARAMETER_ATTACK
            )
        return output_dir

    # @classmethod
    # def from_attack_hyperparameter_tuning(
    #     cls,
    #     device: torch.device,
    #     tuning_result_dir: Path = None,
    #     max_num_samples: int = None,
    #     epochs_per_batch: int = None,
    #     sample_selection_seed: int = None,
    #     save_attack_driver: bool = True,
    #     checkpoint_interval: int = None,
    # ):
    #     """
    #     Creates AttackDriver using output from previous hyperparameter tuning
    #     :param device: device to run on
    #     :param tuning_result_dir: directory where tuning data is saved
    #     :param max_num_samples: number of candidate samples for attack
    #     :param epochs_per_batch: num attack iterations per batch
    #     :param sample_selection_seed: random seed to use when selecting subset
    #     of samples from original dataset
    #     :param save_attack_driver: whether to save AttackDriver .pickle
    #     :param checkpoint_interval: number of batches per checkpoint
    #     :return:
    #     """
    #     if tuning_result_dir is None:
    #         tuning_result_dir = ps.latest_modified_file_with_name_condition(
    #             component_string="optuna_study.pickle",
    #             root_dir=cfg_paths.ATTACK_HYPERPARAMETER_TUNING,
    #         ).parent
    #     optuna_study = rio.ResourceImporter().import_pickle_to_object(
    #         path=tuning_result_dir / "optuna_study.pickle"
    #     )
    #     tuner_driver_dict = rio.ResourceImporter().import_pickle_to_object(
    #         path=tuning_result_dir / "attack_tuner_driver_dict.pickle"
    #     )
    #     tuner_driver = atd.AttackTunerDriver(**tuner_driver_dict)
    #
    #     # attack can have different # epochs per batch than tuner if specified
    #     if epochs_per_batch is None:
    #         epochs_per_batch = tuner_driver.epochs_per_batch
    #
    #     return cls(
    #         device=device,
    #         # model_path=tuner_driver.target_model_path,
    #         checkpoint=tuner_driver.target_model_checkpoint,
    #         attack_hyperparameters=ads.AttackHyperParameterSettings(
    #             **optuna_study.best_params
    #         ),
    #         epochs_per_batch=epochs_per_batch,
    #         max_num_samples=max_num_samples,
    #         sample_selection_seed=sample_selection_seed,
    #         save_attack_driver=save_attack_driver,
    #         checkpoint_interval=checkpoint_interval,
    #         hyperparameter_tuning_result_dir=tuning_result_dir,
    #     )

    def __call__(self) -> aat.AdversarialAttackTrainer | ards.TrainerResult:
        """
        Imports model to attack, then trains and runs attack driver
        :return: TrainerResult (dataclass with attack results)
        """

        # model_hyperparameters = (
        #     edc.X19LSTMHyperParameterSettingsReader().import_struct(
        #         path=self.model_hyperparameters_path
        #     )
        # )

        checkpoint = self.attack_tuner_driver.target_model_checkpoint

        model = tuh.X19LSTMBuilder(settings=self.model_hyperparameters).build()
        model.load_state_dict(state_dict=checkpoint.state_dict)

        attack_trainer = aat.AdversarialAttackTrainer(
            device=self.device,
            model=model,
            state_dict=checkpoint.state_dict,
            attack_hyperparameters=self.attack_hyperparameters,
            epochs_per_batch=self.epochs_per_batch,
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
