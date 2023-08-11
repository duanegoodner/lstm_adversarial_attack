import argparse
import sys
from pathlib import Path

import optuna

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.attack.attack_tuner_driver as atd
import lstm_adversarial_attack.config_paths as cfg_paths
import lstm_adversarial_attack.config_settings as cfg_settings
import lstm_adversarial_attack.path_searches as ps
import lstm_adversarial_attack.resource_io as rio


def resume_tuning(
    num_trials: int, ongoing_tuning_dir: str | Path = None
) -> optuna.Study:
    """
    Resumes training using params of a previously used AttackTunerDriver and
    its associated Optuna Study. Default behavior saves new results to
    same directory as results of previous runs.
    :param num_trials: max # of trials to run
    :param ongoing_tuning_dir: directory where previous run data is saved
    and (under default settings) where new data will be saved.
    :return: an Optuna Study object (which also gets saved as .pickle)
    """

    if num_trials is None:
        num_trials = cfg_settings.ATTACK_TUNING_DEFAULT_NUM_TRIALS

    if ongoing_tuning_dir is None:
        ongoing_tuning_dir = ps.latest_modified_file_with_name_condition(
            component_string="optuna_study.pickle",
            root_dir=cfg_paths.ATTACK_HYPERPARAMETER_TUNING,
        ).parent

    # function accepts str or Path, but we need Path from here on
    ongoing_tuning_dir = Path(ongoing_tuning_dir)

    reloaded_attack_tuner_driver_dict = (
        rio.ResourceImporter().import_pickle_to_object(
            path=ongoing_tuning_dir / "attack_tuner_driver_dict.pickle"
        )
    )
    reloaded_tuner_driver = atd.AttackTunerDriver(
        **reloaded_attack_tuner_driver_dict
    )

    print(
        "Resuming Attack Hyperparameter Tuning study data in:\n"
        f"{reloaded_tuner_driver.output_dir}\n"
    )

    return reloaded_tuner_driver.restart(
        output_dir=ongoing_tuning_dir, num_trials=num_trials
    )


def main(
    existing_study_dir: str = None,
    num_trials: int = None,
) -> optuna.Study:
    """
    Tunes hyperparameters of an AdversarialAttackTrainer and its
    AdversarialAttacker. Can accept target_model_dir OR existing_study_dir or
    neither, but not both. If no args provided, starts new study using most
    recent cross-validation training results to build target model.

    :param existing_study_dir: directory containing an existing optuna.Study
    :param num_trials: number of trials to run. defaults to
    config_settings.ATTACK_TUNING_DEFAULT_NUM_TRIALS
    :return: an optuna.Study object with results of completed trials
    """

    continued_study = resume_tuning(
        num_trials=num_trials, ongoing_tuning_dir=existing_study_dir
    )
    return continued_study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Runs hyperparameter tuning on the attack algorithm. "
            "If no args passed, will start a new Optuna study "
            "using the model data from the most recent data saved in "
            "data/tune_train/cross_validation (uses fold with "
            "median or near median best performance)"
        )
    )
    parser.add_argument(
        "-n",
        "--num_trials",
        type=int,
        action="store",
        nargs="?",
        help="Number of trials to run",
    )

    parser.add_argument(
        "-s",
        "--existing_study_dir",
        type=str,
        action="store",
        nargs="?",
        help=(
            "Path to parent directory of existing Optuna study to use as "
            "starting point for continued (ongoing) training."
        ),
    )

    args_namespace = parser.parse_args()
    args_namespace.existing_study_dir = str(
        cfg_paths.ATTACK_HYPERPARAMETER_TUNING / "2023-06-28_12_11_46.874267"
    )
    main(**args_namespace.__dict__)
