import optuna
import sys
import torch
from pathlib import Path
from typing import Callable

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.tune_train.hyperparameter_tuner as htu
import lstm_adversarial_attack.tune_train.tuner_helpers as tuh
import lstm_adversarial_attack.x19_mort_general_dataset as xmd
import lstm_adversarial_attack.config_settings as lcs



class TunerDriver:
    def __init__(
        self,
        device: torch.device,
        tuning_ranges: tuh.X19MLSTMTuningRanges = tuh.X19MLSTMTuningRanges(
            log_lstm_hidden_size=lcs.TUNING_LOG_LSTM_HIDDEN_SIZE,
            lstm_act_options=lcs.TUNING_LSTM_ACT_OPTIONS,
            dropout=lcs.TUNING_DROPOUT,
            log_fc_hidden_size=lcs.TUNING_LOG_FC_HIDDEN_SIZE,
            fc_act_options=lcs.TUNING_FC_ACT_OPTIONS,
            optimizer_options=lcs.TUNING_OPTIMIZER_OPTIONS,
            learning_rate=lcs.TUNING_LEARNING_RATE,
            log_batch_size=lcs.TUNING_LOG_BATCH_SIZE,
        ),
        dataset: xmd.X19MGeneralDataset = xmd.X19MGeneralDataset.from_feaure_finalizer_output(),
        collate_fn: Callable = xmd.x19m_collate_fn,
    ):
        self.tuner = htu.HyperParameterTuner(
            device=device,
            dataset=dataset,
            collate_fn=collate_fn,
            tuning_ranges=tuning_ranges,
        )

    def __call__(
        self, num_trials: int
    ) -> tuple[htu.HyperParameterTuner, optuna.Study]:
        completed_study = self.tuner.tune(num_trials=num_trials)
        return self.tuner, completed_study
