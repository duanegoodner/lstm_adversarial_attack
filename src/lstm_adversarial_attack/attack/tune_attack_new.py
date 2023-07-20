import argparse
import sys
import optuna
import torch
from os import PathLike
from pathlib import Path
from typing import Callable, Any

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.path_searches as ps
import lstm_adversarial_attack.attack.attack_tuner_driver as atd
import lstm_adversarial_attack.attack.attack_data_structs as ads
import lstm_adversarial_attack.attack.attack_result_data_structs as ards
import lstm_adversarial_attack.attack.attack_hyperparameter_tuner as aht
import lstm_adversarial_attack.config_paths as cfg_paths
import lstm_adversarial_attack.config_settings as cfg_settings
import lstm_adversarial_attack.resource_io as rio
import lstm_adversarial_attack.tune_train.cross_validation_summarizer as cvs
import lstm_adversarial_attack.attack.model_retriever as amr


def main(
    target_model_dir: str = None,
    num_trials: int = None,
    objective_name: str = None,
    max_perts: int = None,
) -> optuna.Study:
    if target_model_dir is None:
        target_model_dir = str(
            ps.most_recently_modified_subdir(
                root_path=cfg_paths.CV_ASSESSMENT_OUTPUT_DIR
            )
        )
    if num_trials is None:
        num_trials = cfg_settings.ATTACK_TUNING_DEFAULT_NUM_TRIALS
    if objective_name is None:
        objective_name = cfg_settings.ATTACK_TUNING_DEFAULT_OBJECTIVE
    if objective_name == "max_num_nonzero_perts":
        assert max_perts is not None
    objective_extra_kwargs = (
        {"max_perts": max_perts} if max_perts is not None else {}
    )

    study = atd.start_new_tuning(
        num_trials=num_trials,
        objective=getattr(aht.AttackTunerObjectives, objective_name),
        objective_extra_kwargs=objective_extra_kwargs,
        target_model_dir=Path(target_model_dir),
    )

    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--target_model_dir",
        type=str,
        action="store",
        nargs="?",
        help=(
            "Directory containing training results of model to attack. Default"
            " is most recently modified training output directory"
        ),
    )
    parser.add_argument(
        "-n",
        "--num_trials",
        type=int,
        action="store",
        nargs="?",
        help=(
            "Number of tuning trials to run. Default is value of"
            " config_settings.ATTACK_TUNING_DEFAULT_NUM_TRIALS"
        ),
    )
    parser.add_argument(
        "-o",
        "--objective_name",
        type=str,
        action="store",
        nargs="?",
        help=(
            "Name of method from AttackTunerObjectives to use to calculate "
            "return value at end of optuna objective_fn. Strongly affects the"
            " type of perturbation that tuning will favor. Defaults to value "
            "of cfg_settings.ATTACK_TUNING_DEFAULT_OBJECTIVE"
        ),
    )
    parser.add_argument(
        "-p",
        "--max_perts",
        type=str,
        action="store",
        nargs="?",
        help=(
            "Value to use for the max_perts argument when using "
            "'max_num_nonzero_perts' as the objective."
        ),
    )

    args_namespace = parser.parse_args()
    completed_study = main(**args_namespace.__dict__)
