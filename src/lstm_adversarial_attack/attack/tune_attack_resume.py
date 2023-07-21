import argparse
import sys
import optuna
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.config_paths as cfg_paths
import lstm_adversarial_attack.config_settings as cfg_settings
import lstm_adversarial_attack.attack.attack_tuner_driver as atd


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

    if num_trials is None:
        num_trials = cfg_settings.ATTACK_TUNING_DEFAULT_NUM_TRIALS
    continued_study = atd.resume_tuning(
        num_trials=num_trials, ongoing_tuning_dir=Path(existing_study_dir)
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
            "starting point for continued (ongoing) training. If value is "
            "provided for this arg, cannot use the -m (target_model_dir)"
            " option."
        ),
    )

    args_namespace = parser.parse_args()
    args_namespace.existing_study_dir = str(
        cfg_paths.ATTACK_HYPERPARAMETER_TUNING / "2023-06-28_12_11_46.874267"
    )
    main(**args_namespace.__dict__)
