import sys
import torch
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, Subset, random_split
from typing import Callable

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.config_paths as cfg_paths
import lstm_adversarial_attack.config_settings as lcs
import lstm_adversarial_attack.resource_io as rio
import lstm_adversarial_attack.tune_train.trainer_driver as td
import lstm_adversarial_attack.tune_train.tuner_helpers as tuh
import lstm_adversarial_attack.x19_mort_general_dataset as xmd


class CrossValidator:
    """
    Runs K-fold cross-validation
    """

    def __init__(
        self,
        device: torch.device,
        dataset: Dataset,
        hyperparameter_settings: tuh.X19LSTMHyperParameterSettings,
        num_folds: int,
        epochs_per_fold: int,
        eval_interval: int,
        evals_per_checkpoint: int,
        collate_fn: Callable = xmd.x19m_collate_fn,
        fold_class: Callable = StratifiedKFold,
        kfold_random_seed: int = lcs.CV_ASSESSMENT_RANDOM_SEED,
        output_root_dir: Path = None,
        save_fold_info: bool = True,
    ):
        self.device = device
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.hyperparameter_settings = hyperparameter_settings
        self.num_folds = num_folds
        self.epochs_per_fold = epochs_per_fold
        self.eval_interval = eval_interval
        self.evals_per_checkpoint = evals_per_checkpoint
        self.fold_class = fold_class
        self.kfold_random_seed = kfold_random_seed
        self.cv_datasets = self.create_datasets()
        if output_root_dir is None:
            output_root_dir = rio.create_timestamped_dir(
                parent_path=cfg_paths.CV_ASSESSMENT_OUTPUT_DIR
            )
        self.output_root_dir = output_root_dir
        self.save_fold_info = save_fold_info

    def create_single_fold_dataset(
        self, eval_fraction: float = lcs.CV_DRIVER_SINGLE_FOLD_EVAL_FRACTION
    ) -> list[tuh.TrainEvalDatasetPair]:
        train_dataset_size = int(
            len(self.dataset) * (1 - eval_fraction)
        )
        test_dataset_size = len(self.dataset) - train_dataset_size
        train_dataset, test_dataset = random_split(
            dataset=self.dataset,
            lengths=(train_dataset_size, test_dataset_size),
        )
        train_eval_pair = tuh.TrainEvalDatasetPair(
            train=train_dataset, validation=test_dataset
        )
        return [train_eval_pair]


    def create_multi_fold_datasets(self) -> list[tuh.TrainEvalDatasetPair]:
        """
        Generates train/eval datasets (subsets of full dataset) for each fold
        :return: K pairs of train/eval datasets
        """
        fold_generator_builder = self.fold_class(
            n_splits=self.num_folds,
            shuffle=True,
            random_state=self.kfold_random_seed,
        )
        fold_generator = fold_generator_builder.split(
            self.dataset[:][0], self.dataset[:][1]
        )

        all_train_eval_pairs = []

        for fold_idx, (train_indices, validation_indices) in enumerate(
            fold_generator
        ):
            train_dataset = Subset(dataset=self.dataset, indices=train_indices)
            validation_dataset = Subset(
                dataset=self.dataset, indices=validation_indices
            )
            all_train_eval_pairs.append(
                tuh.TrainEvalDatasetPair(
                    train=train_dataset, validation=validation_dataset
                )
            )

        return all_train_eval_pairs

    def create_datasets(self) -> list[tuh.TrainEvalDatasetPair]:
        """
        Generates train/eval datasets (subsets of full dataset) for each fold
        :return: K pairs of train/eval datasets
        """
        if self.num_folds == 1:
            return self.create_single_fold_dataset()
        else:
            return self.create_multi_fold_datasets()


    def run_fold(
        self, fold_idx: int, train_eval_pair: tuh.TrainEvalDatasetPair
    ):
        """
        Runs train/eval sequences for a single fold.

        Results get saved under self.root_output_dir
        :param fold_idx: index of fold
        :param train_eval_pair: train/eval dataset pair for current fold
        """
        trainer_driver = td.TrainerDriver(
            device=self.device,
            hyperparameter_settings=self.hyperparameter_settings,
            model=tuh.X19LSTMBuilder(
                settings=self.hyperparameter_settings
            ).build(),
            train_eval_dataset_pair=train_eval_pair,
            output_root_dir=self.output_root_dir,
            tensorboard_output_dir=self.output_root_dir / "tensorboard",
            checkpoint_output_dir=self.output_root_dir
            / "checkpoints"
            / f"fold_{fold_idx}",
            summary_writer_subgroup=f"fold_{fold_idx}",
            summary_writer_add_graph=fold_idx == 0,
        )

        trainer_driver.run(
            num_epochs=self.epochs_per_fold,
            eval_interval=self.eval_interval,
            evals_per_checkpoint=1,
            save_checkpoints=True,
        )

    def run_all_folds(self):
        """
        Runs train/eval sequence on all folds.

        Results saved under self.root_output_dir
        """
        for fold_idx, train_eval_pair in enumerate(self.cv_datasets):
            self.run_fold(fold_idx=fold_idx, train_eval_pair=train_eval_pair)


if __name__ == "__main__":
    if torch.cuda.is_available():
        cur_device = torch.device("cuda:0")
    else:
        cur_device = torch.device("cpu")

    my_dataset = xmd.X19MGeneralDataset.from_feature_finalizer_output()

    study_path = cfg_paths.ONGOING_TUNING_STUDY_PICKLE
    study = rio.ResourceImporter().import_pickle_to_object(path=study_path)
    my_hyperparameters = tuh.X19LSTMHyperParameterSettings(**study.best_params)

    cross_validator = CrossValidator(
        device=cur_device,
        dataset=my_dataset,
        hyperparameter_settings=my_hyperparameters,
        num_folds=5,
        epochs_per_fold=20,
        eval_interval=5,
        evals_per_checkpoint=1,
    )

    cross_validator.run_all_folds()
