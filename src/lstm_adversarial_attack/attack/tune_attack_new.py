import argparse
import sys
from pathlib import Path

import optuna

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.attack.attack_tuner_driver as atd
import lstm_adversarial_attack.config as config
import lstm_adversarial_attack.gpu_helpers as gh
import lstm_adversarial_attack.model.model_retriever as tmr
import lstm_adversarial_attack.path_searches as ps


def start_new_tuning(
    # num_trials: int,
    objective_name: str,
    max_perts: int = None,
    training_result_dir: str = None,
    hyperparameters_path: str = None,
    # custom_config: Path = None,
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
    :param custom_config: path to custom config.toml file
    :return: an Optuna study object (which also get saved as pickle)
    """
    device = gh.get_device()
    config_reader = config.ConfigReader()

    if training_result_dir is None:
        training_result_dir = str(
            ps.latest_modified_file_with_name_condition(
                component_string=".json",
                root_dir=Path(config_reader.read_path("model.cv_driver.output_dir")),
                comparison_type=ps.StringComparisonType.SUFFIX,
            ).parent.parent.parent
        )
    if hyperparameters_path is None:
        hyperparameters_path = str(Path(training_result_dir) / "hyperparameters.json")
    if objective_name is None:
        objective_name = config_reader.get_config_value("attack.tune.objective_name")
    if objective_name == "max_num_nonzero_perts":
        assert max_perts is not None
    objective_extra_kwargs = {"max_perts": max_perts} if max_perts is not None else {}

    tuner_driver = atd.AttackTunerDriver(
        device=device,
        settings=atd.AttackTunerDriverSettings.from_config(),
        paths=atd.AttackTunerDriverPaths.from_config(),
        hyperparameters_path=Path(hyperparameters_path),
        objective_name=objective_name,
        objective_extra_kwargs=objective_extra_kwargs,
        training_result_dir=Path(training_result_dir),
    )

    print(
        "Starting new Attack Hyperparameter Tuning study using"
        f" hyperparameters in:\n {tuner_driver.hyperparameters_path} and"
        f" checkpoint {tuner_driver.target_model_checkpoint}\n\nTuning results"
        f" will be saved in: {tuner_driver.output_dir}\n"
    )

    return tuner_driver.run()


def main(
    training_result_dir: str = None,
    objective_name: str = None,
    max_perts: int = None,
    custom_config: Path = None,
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
    :param custom_config: path of config file (needs to be a .toml file)
    :return: an optuna Study with results of trials
    """
    study = start_new_tuning(
        objective_name=objective_name,
        max_perts=max_perts,
        training_result_dir=training_result_dir,
        # custom_config=custom_config,
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
    # parser.add_argument(
    #     "-n",
    #     "--num_trials",
    #     type=int,
    #     action="store",
    #     nargs="?",
    #     help=(
    #         "Number of tuning trials to run. Default is value of"
    #         " config_settings.ATTACK_TUNING_DEFAULT_NUM_TRIALS"
    #     ),
    # )
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
    completed_study = main(**args_namespace.__dict__)
