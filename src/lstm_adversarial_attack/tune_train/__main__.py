import sys
import time
import torch
from optuna.pruners import MedianPruner
from pathlib import Path

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


def main():
    if torch.cuda.is_available():
        cur_device = torch.device("cuda:0")
    else:
        cur_device = torch.device("cpu")

    tuning_ranges = X19MLSTMTuningRanges(
        log_lstm_hidden_size=TUNING_LOG_LSTM_HIDDEN_SIZE,
        lstm_act_options=TUNING_LSTM_ACT_OPTIONS,
        dropout=TUNING_DROPOUT,
        log_fc_hidden_size=TUNING_LOG_FC_HIDDEN_SIZE,
        fc_act_options=TUNING_FC_ACT_OPTIONS,
        optimizer_options=TUNING_OPTIMIZER_OPTIONS,
        learning_rate=TUNING_LEARNING_RATE,
        log_batch_size=TUNING_LOG_BATCH_SIZE,
    )

    tuner = HyperParameterTuner(
        device=cur_device,
        dataset=X19MGeneralDataset.from_feaure_finalizer_output(),
        collate_fn=x19m_collate_fn,
        num_folds=5,
        num_cv_epochs=20,
        epochs_per_fold=5,
        tuning_ranges=tuning_ranges,
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )
    start = time.time()
    completed_study = tuner.tune(num_trials=30)
    end = time.time()
    print(f"total time = {end - start}")

    return tuner, completed_study


if __name__ == "__main__":
    my_tuner, my_completed_study = main()
