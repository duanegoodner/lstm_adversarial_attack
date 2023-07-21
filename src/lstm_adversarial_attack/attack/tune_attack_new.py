import argparse
import sys
import optuna
from pathlib import Path
from typing import Any, Callable

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.attack.attack_tuner_driver as atd
import lstm_adversarial_attack.attack.attack_hyperparameter_tuner as aht
import lstm_adversarial_attack.config_paths as cfg_paths
import lstm_adversarial_attack.config_settings as cfg_settings
import lstm_adversarial_attack.gpu_helpers as gh
import lstm_adversarial_attack.tune_train.cross_validation_summarizer as cvs

def start_new_tuning(
    num_trials: int,
    objective: Callable[..., float],
    objective_extra_kwargs: dict[str, Any] = None,
    target_model_dir: Path = None,
) -> optuna.Study:
    """
    Creates a new AttackTunerDriver. Causes new Optuna Study to be created via
    AttackHyperParamteterTuner that the driver creates.
    :param num_trials: max num Optuna trials to run
    :param objective: method for calculating return val of tuner objective_fn
    from an attack TrainerResult
    :param target_model_dir: directory containing model and params
    files for model to be attacked.
    :return: an Optuna study object (which also get saved as pickle)
    """
    device = gh.get_device()

    tuner_driver = atd.AttackTunerDriver.from_model_assessment(
        device=device,
        selection_metric=cvs.EvalMetric.VALIDATION_LOSS,
        optimize_direction=cvs.OptimizeDirection.MIN,
        training_output_dir=target_model_dir,
        objective=objective,
        objective_extra_kwargs=objective_extra_kwargs,
    )

    print(
        "Starting new Attack Hyperparameter Tuning study using trained"
        f" predictive model in:\n {tuner_driver.target_model_path}\n\n"
        f"Tuning results will be saved in: {tuner_driver.output_dir}\n"
    )

    return tuner_driver.run(num_trials=num_trials)


def main(
    target_model_dir: str = None,
    num_trials: int = None,
    objective_name: str = None,
    max_perts: int = None,
) -> optuna.Study:
    if target_model_dir is None:
        target_model_dir = str(cfg_paths.ATTACK_DEFAULT_TARGET_MODEL_DIR)
    if num_trials is None:
        num_trials = cfg_settings.ATTACK_TUNING_DEFAULT_NUM_TRIALS
    if objective_name is None:
        objective_name = cfg_settings.ATTACK_TUNING_DEFAULT_OBJECTIVE
    if objective_name == "max_num_nonzero_perts":
        assert max_perts is not None
    objective_extra_kwargs = (
        {"max_perts": max_perts} if max_perts is not None else {}
    )

    study = start_new_tuning(
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
            " is value saved in config_paths.ATTACK_DEFAULT_TARGET_MODEL_DIR"
            " (cast from Path to string)"
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
