import argparse
import sys
from pathlib import Path

import optuna

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.attack.attack_tuner_driver as atd
import lstm_adversarial_attack.config_paths as cfg_paths
import lstm_adversarial_attack.config_settings as cfg_settings
import lstm_adversarial_attack.gpu_helpers as gh
import lstm_adversarial_attack.path_searches as ps
import lstm_adversarial_attack.preprocess.encode_decode as edc
import lstm_adversarial_attack.tuning_db.tuning_studies_database as tsd


def resume_tuning(
    study_name: str = None,
    num_trials: int = None,
) -> optuna.Study:
    """
    Resumes training using params of a previously used AttackTunerDriver and
    its associated Optuna Study. Default behavior saves new results to
    same directory as results of previous runs.
    :param study_name: name of tuning study (as saved in RDB)
    :param num_trials: max # of trials to run
    :return: an Optuna Study object (which also gets saved as .pickle)
    """

    if study_name is None:
        study_name = tsd.ATTACK_TUNING_DB.get_latest_study().study_name

    study_dir = cfg_paths.ATTACK_HYPERPARAMETER_TUNING / study_name

    if num_trials is None:
        num_trials = cfg_settings.ATTACK_TUNING_DEFAULT_NUM_TRIALS

    cur_device = gh.get_device()

    attack_tuner_driver_summary_path = (
        ps.latest_modified_file_with_name_condition(
            component_string="attack_tuner_driver_summary_",
            root_dir=study_dir,
            comparison_type=ps.StringComparisonType.PREFIX,
        )
    )
    attack_tuner_driver_summary = (
        edc.AttackTunerDriverSummaryReader().import_struct(
            path=attack_tuner_driver_summary_path
        )
    )

    partial_constructor_kwargs = {
        key: val
        for key, val in attack_tuner_driver_summary.to_dict().items()
        if key not in ["is_continuation"]
    }

    constructor_kwargs = {
        **{"device": cur_device, **partial_constructor_kwargs}
    }

    attack_tuner_driver = atd.AttackTunerDriver(**constructor_kwargs)
    return attack_tuner_driver.run(num_trials=num_trials)


def main(
    study_name: str = None,
    num_trials: int = None,
) -> optuna.Study:
    """
    Tunes hyperparameters of an AdversarialAttackTrainer and its
    AdversarialAttacker. Can accept target_model_dir OR existing_study_dir or
    neither, but not both. If no args provided, starts new study using most
    recent cross-validation training results to build target model.

    :param study_name: name of tuning study (in RDB)
    :param num_trials: number of trials to run. defaults to
    config_settings.ATTACK_TUNING_DEFAULT_NUM_TRIALS
    :return: an optuna.Study object with results of completed trials
    """

    continued_study = resume_tuning(
        num_trials=num_trials, study_name=study_name
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
        "--study_name",
        type=str,
        action="store",
        nargs="?",
        help=(
            "Name of tuning study in RDB"
        ),
    )

    args_namespace = parser.parse_args()
    main(**args_namespace.__dict__)
