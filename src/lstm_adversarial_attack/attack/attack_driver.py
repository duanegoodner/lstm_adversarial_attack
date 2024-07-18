import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.attack.adv_attack_trainer as aat
import lstm_adversarial_attack.attack.attack_data_structs as ads
import lstm_adversarial_attack.attack.attack_result_data_structs as ards
import lstm_adversarial_attack.config_paths as cfg_paths
import lstm_adversarial_attack.config_settings as cfg_settings
import lstm_adversarial_attack.model.cross_validation_summarizer as cvs
import lstm_adversarial_attack.model.tuner_helpers as tuh
import lstm_adversarial_attack.resource_io as rio
from lstm_adversarial_attack.x19_mort_general_dataset import (
    X19MGeneralDatasetWithIndex, x19m_with_index_collate_fn)


class AttackDriver:
    """
    Instantiates and runs an AdversarialAttackTrainer
    """

    def __init__(
        self,
        target_model_checkpoint_info: cvs.CheckpointInfo,
        device: torch.device,
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
        self.target_model_checkpoint_info = target_model_checkpoint_info
        self.device = device
        self.attack_tuning_study_name = attack_tuning_study_name
        self.db_env_var_name = db_env_var_name
        self.model_hyperparameters = model_hyperparameters
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

    def __call__(self) -> aat.AdversarialAttackTrainer | ards.TrainerResult:
        """
        Imports model to attack, then trains and runs attack driver
        :return: TrainerResult (dataclass with attack results)
        """

        checkpoint = self.target_model_checkpoint_info.checkpoint
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
