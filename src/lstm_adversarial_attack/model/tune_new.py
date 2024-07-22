import sys
from pathlib import Path

import optuna
import torch

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.model.model_tuner_driver as td
from lstm_adversarial_attack.config import CONFIG_READER

def main(preprocess_id: str = None) -> optuna.Study:
    """
    Runs a new optuna study consisting of multiple trials to find
    optimized hyperparameters for use when generating a model with a
    X19LSTMBuilder and training it with a StandardModelTrainer. Results will be
    saved in a newly created director under
    data/model/hyperparameter_tuning/. If overall study is killed early,
    data from completed trials is still saved.
    :return: an optuna.Study object with results of completed trials.
    """

    num_trials = CONFIG_READER.get_config_value(
        "model.tuner_driver.num_trials")

    if torch.cuda.is_available():
        cur_device = torch.device("cuda:0")
    else:
        cur_device = torch.device("cpu")

    if preprocess_id is None:
        preprocess_output_root = Path(
            CONFIG_READER.read_path("preprocess.output_root")
        )
        preprocess_id = str(
            max(
                [
                    int(item.name)
                    for item in list(preprocess_output_root.iterdir())
                    if (item.is_dir())
                ]
            )
        )

    tuner_driver = td.ModelTunerDriver(
        device=cur_device,
        settings=td.ModelTunerDriverSettings.from_config(),
        paths=td.ModelTunerDriverPaths.from_config(),
        preprocess_id=preprocess_id,
    )

    study = tuner_driver(num_trials=num_trials)

    return study


if __name__ == "__main__":
    completed_study = main()
