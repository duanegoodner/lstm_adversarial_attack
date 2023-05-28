import time
import torch
from optuna.pruners import MedianPruner
from x19_mort_general_dataset import x19m_collate_fn, X19MGeneralDataset
from hyperparameter_tuner import (
    HyperParameterTuner,
    X19MLSTMTuningRanges,
)


my_tuning_ranges = X19MLSTMTuningRanges(
    log_lstm_hidden_size=(5, 7),
    lstm_act_options=("ReLU", "Tanh"),
    dropout=(0, 0.5),
    log_fc_hidden_size=(4, 8),
    fc_act_options=("ReLU", "Tanh"),
    optimizer_options=("Adam", "RMSprop", "SGD"),
    learning_rate=(1e-5, 1e-1),
    log_batch_size=(5, 8),
)


def main():
    if torch.cuda.is_available():
        cur_device = torch.device("cuda:0")
    else:
        cur_device = torch.device("cpu")

    tuner = HyperParameterTuner(
        device=cur_device,
        dataset=X19MGeneralDataset.from_feaure_finalizer_output(),
        collate_fn=x19m_collate_fn,
        num_folds=5,
        num_cv_epochs=20,
        epochs_per_fold=5,
        tuning_ranges=my_tuning_ranges,
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    )

    completed_study = tuner.tune(num_trials=30)

    return tuner, completed_study


if __name__ == "__main__":
    start = time.time()
    my_tuner, my_completed_study = main()
    end = time.time()
    print(f"total time = {end - start}")


