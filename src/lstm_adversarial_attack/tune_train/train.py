import argparse
import sys
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.config_paths as cfg_paths
import lstm_adversarial_attack.config_settings as cfg_set
import lstm_adversarial_attack.gpu_helpers as gh
import lstm_adversarial_attack.path_searches as ps
import lstm_adversarial_attack.tune_train.cross_validator_driver as cvd
import lstm_adversarial_attack.x19_mort_general_dataset as xmd


def main(
    study_path: str = None,
    num_folds: int = None,
    epochs_per_fold: int = None
):

    #  Do this instead of default args for easy argparse compatibility
    if study_path is None:
        study_path = ps.most_recently_modified_file_named(
            target_filename="optuna_study.pickle",
            root_dir=cfg_paths.HYPERPARAMETER_OUTPUT_DIR
        )
    if num_folds is None:
        num_folds = cfg_set.CV_DRIVER_NUM_FOLDS
    if epochs_per_fold is None:
        epochs_per_fold = cfg_set.CV_DRIVER_EPOCHS_PER_FOLD

    assert num_folds > 0

    cur_device = gh.get_device()
    # if torch.cuda.is_available():
    #     cur_device = torch.device("cuda:0")
    # else:
    #     cur_device = torch.device("cpu")

    # if num_folds > 1:
    cv_driver = cvd.CrossValidatorDriver.from_study_path(
        device=cur_device,
        dataset=xmd.X19MGeneralDataset.from_feature_finalizer_output(),
        study_path=Path(study_path),
        num_folds=num_folds,
        epochs_per_fold=epochs_per_fold
    )
    cv_driver.run()
    # else:
    #     single_fold_trainer = sft.SingleFoldTrainer(
    #         device=cur_device,
    #         dataset=xmd.X19MGeneralDataset.from_feature_finalizer_output(),
    #         train_dataset_fraction=1 - 1 / cfg_set.CV_DRIVER_NUM_FOLDS,
    #         study_path=study_path,
    #         num_epochs=epochs_per_fold
    #     )
    #     single_fold_trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--study_path",
        type=str,
        action="store",
        nargs="?",
        help="Path to an optuna.Study object .pickle file. Study will be "
             "imported, and model training will use its .best_params. If "
             "not specified, the most recently modified file named "
             "'optuna_study.pickle' under path specified by "
             "config_paths.HYPERPARAMETER_OUTPUT_DIR will be used."
    )
    parser.add_argument(
        "-f",
        "--num_folds",
        type=int,
        action="store",
        nargs="?",
        help="Number of cross-validation folds. Defaults to "
             "config_settings.CV_DRIVER_NUM_FOLDS"
    )

    parser.add_argument(
        "-e",
        "--epochs_per_fold",
        type=int,
        action="store",
        nargs="?",
        help="Number of epochs per fold. Defaults to "
             "config_settings.CV_DRIVER_EPOCHS_PER_FOLD"
    )

    args_namespace = parser.parse_args()


    main(**args_namespace.__dict__)



