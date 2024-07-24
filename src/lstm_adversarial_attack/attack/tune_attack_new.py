import argparse
import sys
from datetime import datetime
from pathlib import Path

import optuna

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.attack.attack_data_structs as ads
import lstm_adversarial_attack.attack.attack_tuner_driver as atd
import lstm_adversarial_attack.gpu_helpers as gh
import lstm_adversarial_attack.path_searches as ps
import lstm_adversarial_attack.preprocess.encode_decode as edc
from lstm_adversarial_attack.config import CONFIG_READER


def start_new_tuning(
    # model_training_result_dir: str = None,
    cv_training_id: str = None,
) -> optuna.Study:
    """
    Creates a new AttackTunerDriver. Causes new Optuna Study to be created via
    AttackHyperParamteterTuner that the driver creates.
    :param model_training_result_dir: directory containing model and params
    files for model to be attacked.
    :return: an Optuna study object (which also get saved as pickle)
    """
    device = gh.get_device()
    attack_tuning_id = "".join(
        char for char in str(datetime.now()) if char.isdigit()
    )

    cv_output_root = Path(
        CONFIG_READER.read_path("model.cv_driver.output_dir")
    )

    if cv_training_id is None:
        cv_training_id = ps.get_latest_sequential_child_dirname(
            root_dir=cv_output_root
        )

    cv_driver_summary_path = (
        cv_output_root
        / cv_training_id
        / f"cross_validator_driver_summary_{cv_training_id}.json"
    )

    cv_driver_summary = edc.CrossValidatorSummaryReader().import_struct(
        path=cv_driver_summary_path
    )

    tuner_driver = atd.AttackTunerDriver(
        device=device,
        preprocess_id=cv_driver_summary.preprocess_id,
        attack_tuning_id=attack_tuning_id,
        model_hyperparameters=cv_driver_summary.model_hyperparameters,
        settings=ads.AttackTunerDriverSettings.from_config(),
        paths=ads.AttackTunerDriverPaths.from_config(),
        model_training_result_dir=cv_output_root / cv_training_id,
    )

    print(
        "Starting new Attack Hyperparameter Tuning study using trained model from"
        f" {str(cv_output_root / cv_training_id)} as the attack target. \n\nTuning results"
        f" will be saved in: {tuner_driver.output_dir}\n"
    )

    return tuner_driver.run()


def main(
    cv_training_id: str = None,
) -> optuna.Study:
    """
    Takes arguments in format provided by command line interface and uses them
    to call start_new_tuning().
    :param model_training_result_dir: directory containing model and params
    files for model to be attacked
    """
    study = start_new_tuning(
        cv_training_id=cv_training_id,
    )

    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Starts a new study for tuning attack hyperparameters."
    )
    parser.add_argument(
        "-t",
        "--cv_training_id",
        type=str,
        action="store",
        nargs="?",
        help=(
            "ID of cross validation training session providing trained model for attack."
        ),
    )

    args_namespace = parser.parse_args()
    completed_study = main(**args_namespace.__dict__)
