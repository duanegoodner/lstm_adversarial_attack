import argparse
import sys
from pathlib import Path

import optuna

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.attack.attack_tuner_driver as atd
import lstm_adversarial_attack.gpu_helpers as gh
import lstm_adversarial_attack.attack.attack_data_structs as ads
from lstm_adversarial_attack.config import CONFIG_READER
import lstm_adversarial_attack.preprocess.encode_decode as edc


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

    if model_training_result_dir is None:
        cv_output_parent = Path(
            CONFIG_READER.read_path("model.cv_driver.output_dir")
        )
        all_training_dirs = [
            item
            for item in list(cv_output_parent.iterdir())
            if item.name.startswith("cv_training_")
        ]
        model_training_result_dir = str(max(all_training_dirs))

    cv_driver_summary_paths = [
        item
        for item in list(Path(model_training_result_dir).iterdir())
        if item.name.startswith("cross_validator_driver_summary_")
    ]
    assert len(cv_driver_summary_paths) == 1
    cv_driver_summary = edc.CrossValidatorSummaryReader().import_struct(
        path=cv_driver_summary_paths[0]
    )

    tuner_driver = atd.AttackTunerDriver(
        device=device,
        preprocess_id=cv_driver_summary.preprocess_id,
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
            "Directory containing training results of model to attack. Defaults to most recently "
            "generated training result."
        ),
    )

    args_namespace = parser.parse_args()
    completed_study = main(**args_namespace.__dict__)
