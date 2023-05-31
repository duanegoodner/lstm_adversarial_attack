import optuna
import sys
import time
import torch
from pathlib import Path
from typing import Callable

sys.path.append(str(Path(__file__).parent.parent.parent))
from hyperparameter_tuner import HyperParameterTuner
from tuner_helpers import X19MLSTMTuningRanges

from lstm_adversarial_attack.x19_mort_general_dataset import (
    x19m_collate_fn,
    X19MGeneralDataset,
)
from lstm_adversarial_attack.config_settings import (
    TUNING_LOG_LSTM_HIDDEN_SIZE,
    TUNING_LSTM_ACT_OPTIONS,
    TUNING_DROPOUT,
    TUNING_LOG_FC_HIDDEN_SIZE,
    TUNING_FC_ACT_OPTIONS,
    TUNING_OPTIMIZER_OPTIONS,
    TUNING_LEARNING_RATE,
    TUNING_LOG_BATCH_SIZE,
)


class TunerDriver:
    def __init__(
        self,
        device: torch.device,
        tuning_ranges: X19MLSTMTuningRanges = X19MLSTMTuningRanges(
            log_lstm_hidden_size=TUNING_LOG_LSTM_HIDDEN_SIZE,
            lstm_act_options=TUNING_LSTM_ACT_OPTIONS,
            dropout=TUNING_DROPOUT,
            log_fc_hidden_size=TUNING_LOG_FC_HIDDEN_SIZE,
            fc_act_options=TUNING_FC_ACT_OPTIONS,
            optimizer_options=TUNING_OPTIMIZER_OPTIONS,
            learning_rate=TUNING_LEARNING_RATE,
            log_batch_size=TUNING_LOG_BATCH_SIZE,
        ),
        dataset: X19MGeneralDataset = X19MGeneralDataset.from_feaure_finalizer_output(),
        collate_fn: Callable = x19m_collate_fn,
    ):
        self.tuner = HyperParameterTuner(
            device=device,
            dataset=dataset,
            collate_fn=collate_fn,
            tuning_ranges=tuning_ranges,
        )

    def __call__(
        self, num_trials: int
    ) -> tuple[HyperParameterTuner, optuna.Study]:
        completed_study = self.tuner.tune(num_trials=num_trials)
        return self.tuner, completed_study


if __name__ == "__main__":
    if torch.cuda.is_available():
        cur_device = torch.device("cuda:0")
    else:
        cur_device = torch.device("cpu")

    tuner_driver = TunerDriver(device=cur_device)
    tuner_driver(num_trials=5)