import argparse
import pprint
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.tuning_db.tuning_studies_database as tsd
import lstm_adversarial_attack.utils.path_searches as ps
from lstm_adversarial_attack.config.read_write import PATH_CONFIG_READER


def main(model_tuning_id: str = None, model_tuning_trial_number: int = None):

    if model_tuning_id is None:
        model_tuning_output_root = PATH_CONFIG_READER.read_path(
            config_key="model.tuner_driver.output_dir"
        )
        model_tuning_id = ps.get_latest_sequential_child_dirname(
            root_dir=Path(model_tuning_output_root)
        )

    model_tuning_study = tsd.MODEL_TUNING_DB.get_study(
        study_name=f"model_tuning_{model_tuning_id}"
    )

    if model_tuning_trial_number is None:
        model_tuning_trial_number = model_tuning_study.best_trial.number

    hyperparams_dict = model_tuning_study.trials[
        model_tuning_trial_number
    ].params

    print(
        f"Summary of model tuning session {model_tuning_id}, trial "
        f"number {model_tuning_trial_number}:"
    )

    print(
        f"Objective function value = "
        f"{model_tuning_study.trials[model_tuning_trial_number].value}"
    )
    print("Hyperparameters = ")
    pprint.pprint(hyperparams_dict)
    print(
        f"\nThe best trial from tuning session {model_tuning_id}:\n"
        f"Trial number {model_tuning_study.best_trial.number} with objective "
        f"function value = {model_tuning_study.best_trial.value}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retrieves and displays hyperparameters tested during a "
        "model tuning session."
    )
    parser.add_argument(
        "-t",
        "--model_tuning_id",
        type=str,
        action="store",
        nargs="?",
        help="ID of model tuning session. Defaults to most recently created "
        "session.",
    )
    parser.add_argument(
        "--model_tuning_trial_number",
        "-n",
        type=int,
        action="store",
        help="Optional integer specifying trial number (within study)."
        "Defaults to best trial from study.",
    )
    args_namespace = parser.parse_args()
    main(
        model_tuning_id=args_namespace.model_tuning_id,
        model_tuning_trial_number=args_namespace.model_tuning_trial_number,
    )
