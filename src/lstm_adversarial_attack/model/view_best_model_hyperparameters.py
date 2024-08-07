import argparse
import pprint
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.utils.path_searches as ps
from lstm_adversarial_attack.config.read_write import PATH_CONFIG_READER
from lstm_adversarial_attack.tuning_db.tuning_studies_database import (
    MODEL_TUNING_DB,
)


def main(model_tuning_id: str = None):
    if model_tuning_id is None:
        model_tuning_output_root = PATH_CONFIG_READER.read_path(
            config_key="model.tuner_driver.output_dir"
        )
        model_tuning_id = ps.get_latest_sequential_child_dirname(
            root_dir=Path(model_tuning_output_root)
        )

    study_name = f"model_tuning_{model_tuning_id}"
    best_hyperparameters = MODEL_TUNING_DB.get_best_params(
        study_name=study_name
    )
    print(
        f"Best performing set of model hyperparameters tested during model tuning "
        f"session {model_tuning_id}:\n"
    )
    pprint.pprint(best_hyperparameters)


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
    args_namespace = parser.parse_args()
    main(model_tuning_id=args_namespace.model_tuning_id)
