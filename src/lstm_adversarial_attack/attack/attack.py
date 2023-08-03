import argparse
import sys
import torch
from pathlib import Path
from typing import Callable

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.attack.adv_attack_trainer as aat
import lstm_adversarial_attack.attack.attack_data_structs as ads
import lstm_adversarial_attack.attack.attack_result_data_structs as ards
import lstm_adversarial_attack.attack.attack_tuner_driver as atd
import lstm_adversarial_attack.path_searches as ps
import lstm_adversarial_attack.resource_io as rio
import lstm_adversarial_attack.config_paths as cfg_paths
import lstm_adversarial_attack.config_settings as cfg_settings
import lstm_adversarial_attack.data_provenance as dpr

from lstm_adversarial_attack.x19_mort_general_dataset import (
    X19MGeneralDatasetWithIndex,
    x19m_with_index_collate_fn,
)


class AttackDriver(dpr.HasDataProvenance):
    """
    Instantiates and runs an AdversarialAttackTrainer
    """

    def __init__(
        self,
        device: torch.device,
        model_path: Path,
        checkpoint: dict,
        batch_size: int,
        epochs_per_batch: int,
        kappa: float,
        lambda_1: float,
        optimizer_constructor: Callable,
        optimizer_constructor_kwargs: dict = None,
        max_num_samples=None,
        sample_selection_seed=None,
        attack_misclassified_samples: bool = False,
        output_dir: Path = None,
        result_file_prefix: str = "",
        save_attack_driver: bool = False,
        checkpoint_interval: int = None,
        attack_hyperparameter_settings: ads.AttackHyperParameterSettings = None,
        hyperparameter_tuning_result_dir: Path = None,
    ):
        """
        :param device: device to run on
        :param model_path: path to pickle file with model to be attacked
        :param checkpoint: Info saved during training classifier. Contents
        include model params.
        :param batch_size: num samples per batch
        :param epochs_per_batch: number of attack iterations per batch
        :param kappa: Parameter from Equation 1 in Sun et al
        (https://arxiv.org/abs/1802.04822). Defines a margin by which alternate
        class logit value needs to exceed original class logit value in order
        to reduce loss function.
        :param lambda_1: L1 regularization constant applied to perturbations
        :param optimizer_constructor: constructor of optimizer used when
        searching for adversarial example_data
        :param optimizer_constructor_kwargs: kwargs passed to optimizer
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
        self.device = device
        self.model_path = model_path
        self.checkpoint = checkpoint
        self.batch_size = batch_size
        self.epochs_per_batch = epochs_per_batch
        self.kappa = kappa
        self.lambda_1 = lambda_1
        self.optimizer_constructor = optimizer_constructor
        if optimizer_constructor_kwargs is None:
            optimizer_constructor_kwargs = {"lr": 1e-1}
        self.optimizer_constructor_kwargs = optimizer_constructor_kwargs
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
        self.attack_hyperparameter_settings = attack_hyperparameter_settings
        self.hyperparameter_tuning_results_dir = (
            hyperparameter_tuning_result_dir
        )
        self.export(filename="attack_driver_dict.pickle")

    @property
    def provenance_info(self) -> dpr.ProvenanceInfo:
        return dpr.ProvenanceInfo(
            previous_info=(
                self.hyperparameter_tuning_results_dir / "provenance.pickle"
                if self.hyperparameter_tuning_results_dir is not None
                and (
                    self.hyperparameter_tuning_results_dir
                    / "provenance.pickle"
                ).exists()
                else None
            ),
            category_name="attack_driver",
            new_items={
                "epochs_per_batch": self.epochs_per_batch,
                "attack_hyperparameter_settings": (
                    self.attack_hyperparameter_settings
                ),
                "hyperparameter_tuning_result_dir": (
                    self.hyperparameter_tuning_results_dir
                ),
            },
            output_dir=self.output_dir,
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

    @classmethod
    def from_attack_hyperparameter_settings(
        cls,
        device: torch.device,
        model_path: Path,
        checkpoint: dict,
        settings: ads.AttackHyperParameterSettings,
        tuning_result_dir: Path = None,
        epochs_per_batch: int = None,
        max_num_samples: int = None,
        sample_selection_seed: int = None,
        attack_misclassified_samples: bool = False,
        output_dir: Path = None,
        save_attack_driver: bool = False,
        checkpoint_interval: int = None,
    ):
        """
        Creates AttackDriver from AttackHyperParameterSettings object
        :param device: device to run on
        :param model_path: path to pickle with model to attack
        :param checkpoint: Info saved during training classifier. Contents
        include model params.
        :param settings: hyperparameters to be used by AttackTrainer
        :param tuning_result_dir: directory containing attack hyperparameter
        tuning results. Will be None
        :param epochs_per_batch: num attack iterations per batch
        :param max_num_samples: number of candidate samples for attack
        :param sample_selection_seed: random seed to use when selecting subset
        of samples from original dataset
        :param attack_misclassified_samples: whether to run attacks on samples
        that original model misclassifies
        :param output_dir: directory where attack results will be saved
        :param save_attack_driver: whether to save AttackDriver .pickle
        :param checkpoint_interval: number of batches per checkpoint
        :return:
        """

        return cls(
            device=device,
            model_path=model_path,
            checkpoint=checkpoint,
            batch_size=2**settings.log_batch_size,
            kappa=settings.kappa,
            lambda_1=settings.lambda_1,
            optimizer_constructor=getattr(
                torch.optim, settings.optimizer_name
            ),
            optimizer_constructor_kwargs={"lr": settings.learning_rate},
            max_num_samples=max_num_samples,
            sample_selection_seed=sample_selection_seed,
            attack_misclassified_samples=attack_misclassified_samples,
            output_dir=output_dir,
            epochs_per_batch=epochs_per_batch,
            save_attack_driver=save_attack_driver,
            checkpoint_interval=checkpoint_interval,
            attack_hyperparameter_settings=settings,
            hyperparameter_tuning_result_dir=tuning_result_dir,
        )

    @classmethod
    def from_attack_hyperparameter_tuning(
        cls,
        device: torch.device,
        tuning_result_dir: Path = None,
        max_num_samples: int = None,
        epochs_per_batch: int = None,
        sample_selection_seed: int = None,
        save_attack_driver: bool = True,
        checkpoint_interval: int = None,
    ):
        """
        Creates AttackDriver using output from previous hyperparameter tuning
        :param device: device to run on
        :param tuning_result_dir: directory where tuning data is saved
        :param max_num_samples: number of candidate samples for attack
        :param epochs_per_batch: num attack iterations per batch
        :param sample_selection_seed: random seed to use when selecting subset
        of samples from original dataset
        :param save_attack_driver: whether to save AttackDriver .pickle
        :param checkpoint_interval: number of batches per checkpoint
        :return:
        """
        if tuning_result_dir is None:
            tuning_result_dir = ps.latest_modified_file_with_name_condition(
                component_string="optuna_study.pickle",
                root_dir=cfg_paths.ATTACK_HYPERPARAMETER_TUNING,
            ).parent
        optuna_study = rio.ResourceImporter().import_pickle_to_object(
            path=tuning_result_dir / "optuna_study.pickle"
        )
        tuner_driver_dict = rio.ResourceImporter().import_pickle_to_object(
            path=tuning_result_dir / "attack_tuner_driver_dict.pickle"
        )
        tuner_driver = atd.AttackTunerDriver(**tuner_driver_dict)

        # attack can have different # epochs per batch than tuner if specified
        if epochs_per_batch is None:
            epochs_per_batch = tuner_driver.epochs_per_batch

        return cls.from_attack_hyperparameter_settings(
            device=device,
            model_path=tuner_driver.target_model_path,
            checkpoint=tuner_driver.target_model_checkpoint,
            settings=ads.AttackHyperParameterSettings(
                **optuna_study.best_params
            ),
            epochs_per_batch=epochs_per_batch,
            max_num_samples=max_num_samples,
            sample_selection_seed=sample_selection_seed,
            save_attack_driver=save_attack_driver,
            checkpoint_interval=checkpoint_interval,
            tuning_result_dir=tuning_result_dir,
        )

    def __call__(self) -> aat.AdversarialAttackTrainer | ards.TrainerResult:
        """
        Imports model to attack, then trains and runs attack driver
        :return: TrainerResult (dataclass with attack results)
        """
        model = rio.ResourceImporter().import_pickle_to_object(
            path=self.model_path
        )
        attack_trainer = aat.AdversarialAttackTrainer(
            device=self.device,
            model=model,
            state_dict=self.checkpoint["state_dict"],
            batch_size=self.batch_size,
            kappa=self.kappa,
            lambda_1=self.lambda_1,
            epochs_per_batch=self.epochs_per_batch,
            optimizer_constructor=self.optimizer_constructor,
            optimizer_constructor_kwargs=self.optimizer_constructor_kwargs,
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


def main(tuning_result_dir: str) -> ards.TrainerSuccessSummary:
    """
    Runs attack on dataset
    :param tuning_result_dir: path to directory containing attack tuning
    result to use for full attack
    """
    if torch.cuda.is_available():
        cur_device = torch.device("cuda:0")
    else:
        cur_device = torch.device("cpu")

    tuning_result_dir_path = (
        Path(tuning_result_dir) if tuning_result_dir is not None else None
    )

    attack_driver = AttackDriver.from_attack_hyperparameter_tuning(
        device=cur_device,
        sample_selection_seed=cfg_settings.ATTACK_SAMPLE_SELECTION_SEED,
        checkpoint_interval=cfg_settings.ATTACK_CHECKPOINT_INTERVAL,
        tuning_result_dir=tuning_result_dir_path,
    )
    trainer_result = attack_driver()
    success_summary = ards.TrainerSuccessSummary(trainer_result=trainer_result)

    return success_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--tuning_result_dir",
        action="store",
        nargs="?",
        help=(
            "Path to directory containing attack tuning results to use for "
            "attack on full dataset. Defaults to most recent tuning "
            "result."
        ),
    )
    args_namespace = parser.parse_args()
    # args_namespace.tuning_result_dir = str(
    #     cfg_paths.ATTACK_HYPERPARAMETER_TUNING / "2023-07-01_11_03_13.591090"
    # )
    cur_success_summary = main(**args_namespace.__dict__)
