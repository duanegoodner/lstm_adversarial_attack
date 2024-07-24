import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.data_structures as ds
import lstm_adversarial_attack.gpu_helpers as gh
import lstm_adversarial_attack.model.cross_validator_driver as cvd
import lstm_adversarial_attack.path_searches as ps
from lstm_adversarial_attack.config import CONFIG_READER


def main(model_tuning_id: str = None) -> dict[int, ds.TrainEvalLogPair]:
    cv_training_id = "".join(
        char for char in str(datetime.now()) if char.isdigit()
    )

    model_tuning_output_root = Path(
        CONFIG_READER.read_path("model.tuner_driver.output_dir")
    )

    if model_tuning_id is None:
        model_tuning_id = ps.get_latest_sequential_child_dirname(
            root_dir=model_tuning_output_root
        )

    cv_driver = cvd.CrossValidatorDriver.from_model_tuning_id(
        model_tuning_id=model_tuning_id,
        cv_training_id=cv_training_id,
        device=gh.get_device(),
    )

    return cv_driver.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--model_tuning_id",
        type=str,
        action="store",
        nargs="?",
        help="ID of model tuning session to obtain hyperparameters from",
    )

    args_namespace = parser.parse_args()

    all_fold_results = main(**args_namespace.__dict__)
