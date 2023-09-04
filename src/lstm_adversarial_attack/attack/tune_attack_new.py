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
import lstm_adversarial_attack.tune_train.model_retriever as tmr


def start_new_tuning(
    num_trials: int,
    objective_name: str,
    max_perts: int = None,
    training_result_dir: str = None,
    hyperparameters_path: str = None,
) -> optuna.Study:
    """
    Creates a new AttackTunerDriver. Causes new Optuna Study to be created via
    AttackHyperParamteterTuner that the driver creates.
    :param num_trials: max num Optuna trials to run
    :param objective_name: name of method from AttackTunerObjectives to use for
    calculating return val of tuner objective_fn from an attack TrainerResult
    :param max_perts: parameter needed for 'max_num_nonzero_perts' objective.
    Not needed when using other objectives.
    :param training_result_dir: directory containing model and params
    files for model to be attacked.
    :param hyperparameters_path: path to file containing predictive model
    hyperparameters
    :return: an Optuna study object (which also get saved as pickle)
    """
    device = gh.get_device()

    if training_result_dir is None:
        training_result_dir = str(
            ps.latest_modified_file_with_name_condition(
                component_string=".json",
                root_dir=cfg_paths.CV_ASSESSMENT_OUTPUT_DIR,
                comparison_type=ps.StringComparisonType.SUFFIX,
            ).parent.parent.parent
        )
    if hyperparameters_path is None:
        hyperparameters_path = str(
            Path(training_result_dir) / "hyperparameters.json"
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

    target_fold_checkpoint_info_pair = tmr.ModelRetriever(
        training_output_dir=Path(training_result_dir)
    ).get_representative_checkpoint()

    tuner_driver = atd.AttackTunerDriver(
        device=device,
        hyperparameters_path=Path(hyperparameters_path),
        objective_name=objective_name,
        objective_extra_kwargs=objective_extra_kwargs,
        training_result_dir=Path(training_result_dir),
        target_checkpoint=target_fold_checkpoint_info_pair.checkpoint_info.checkpoint,
        target_checkpoint_path=target_fold_checkpoint_info_pair.checkpoint_info.save_path,
        target_fold=target_fold_checkpoint_info_pair.fold
    )

    print(
        "Starting new Attack Hyperparameter Tuning study using"
        f" hyperparameters in:\n {tuner_driver.hyperparameters_path} and"
        f" checkpoint {tuner_driver.target_model_checkpoint}\n\nTuning results"
        f" will be saved in: {tuner_driver.output_dir}\n"
    )

    return tuner_driver.run(num_trials=num_trials)


def main(
    training_result_dir: str = None,
    num_trials: int = None,
    objective_name: str = None,
    max_perts: int = None,
) -> optuna.Study:
    """
    Takes arguments in format provided by command line interface and uses them
    to call start_new_tuning().
    :param training_result_dir: directory containing model and params
    files for model to be attacked
    :param num_trials: max num Optuna trials to run
    :param objective_name: name of method from AttackTunerObjectives to use for
    calculating return val of tuner objective_fn from an attack TrainerResult
    :param max_perts: Specifies the maximum number of nonzero elements an
    adversarial perturbation can have and still be counted as a success by the
    tuning objective function.
    :return: an optuna Study with results of trials
    """
    study = start_new_tuning(
        num_trials=num_trials,
        objective_name=objective_name,
        max_perts=max_perts,
        training_result_dir=training_result_dir,
    )

    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Starts a new study for tuning attack hyperparameters."
    )
    parser.add_argument(
        "-m",
        "--training_result_dir",
        type=str,
        action="store",
        nargs="?",
        help=(
            "Directory containing training results of model to attack. Default"
            " is value saved in"
            " config_paths.ATTACK_DEFAULT_training_result_dir (cast from Path"
            " to string)"
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
            "return value at end of optuna objective_fn. Available methods "
            "are: sparse_small, sparse_small_max, sparsity, and "
            "max_num_nonzero_perts. Default is value of "
            "cfg_settings.ATTACK_TUNING_DEFAULT_OBJECTIVE"
        ),
    )
    parser.add_argument(
        "-p",
        "--max_perts",
        type=str,
        action="store",
        nargs="?",
        help=(
            "Required only when using max_num_nonzero_perts as the objective. "
            "Specifies the maximum number of nonzero elements an adversarial "
            "perturbation can have and still be counted as a success by the "
            "tuning objective function."
        ),
    )

    args_namespace = parser.parse_args()
    # args_namespace.training_result_dir = str(
    #     cfg_paths.CV_ASSESSMENT_OUTPUT_DIR / "2023-06-17_23_57_23.366142"
    # )
    completed_study = main(**args_namespace.__dict__)
