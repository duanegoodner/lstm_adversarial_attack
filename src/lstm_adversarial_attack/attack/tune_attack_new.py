import argparse
import sys
from pathlib import Path

import optuna

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.attack.attack_tuner_driver as atd
import lstm_adversarial_attack.config as config
import lstm_adversarial_attack.gpu_helpers as gh
import lstm_adversarial_attack.path_searches as ps
import lstm_adversarial_attack.attack.attack_data_structs as ads


def start_new_tuning(
    model_training_result_dir: str = None,
) -> optuna.Study:
    """
    Creates a new AttackTunerDriver. Causes new Optuna Study to be created via
    AttackHyperParamteterTuner that the driver creates.
    :param model_training_result_dir: directory containing model and params
    files for model to be attacked.
    :return: an Optuna study object (which also get saved as pickle)
    """
    device = gh.get_device()
    config_reader = config.ConfigReader()

    if model_training_result_dir is None:
        model_training_result_dir = str(
            ps.latest_modified_file_with_name_condition(
                component_string=".json",
                root_dir=Path(config_reader.read_path("model.cv_driver.output_dir")),
                comparison_type=ps.StringComparisonType.SUFFIX,
            ).parent.parent.parent
        )

    tuner_driver = atd.AttackTunerDriver(
        device=device,
        settings=ads.AttackTunerDriverSettings.from_config(),
        paths=ads.AttackTunerDriverPaths.from_config(),
        model_training_result_dir=Path(model_training_result_dir),
    )

    print(
        "Starting new Attack Hyperparameter Tuning study using"
        f" hyperparameters in:\n {tuner_driver.hyperparameters_path} and"
        f" checkpoint {tuner_driver.target_model_checkpoint}\n\nTuning results"
        f" will be saved in: {tuner_driver.output_dir}\n"
    )

    return tuner_driver.run()


def main(
    model_training_result_dir: str = None,
) -> optuna.Study:
    """
    Takes arguments in format provided by command line interface and uses them
    to call start_new_tuning().
    :param model_training_result_dir: directory containing model and params
    files for model to be attacked
    """
    study = start_new_tuning(
        model_training_result_dir=model_training_result_dir,
    )

    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Starts a new study for tuning attack hyperparameters."
    )
    parser.add_argument(
        "-m",
        "--model_training_result_dir",
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

    args_namespace = parser.parse_args()
    completed_study = main(**args_namespace.__dict__)
