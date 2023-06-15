import sys
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.attack.attack_data_structs as ads
import lstm_adversarial_attack.attack.attack_hyperparameter_tuner as aht
import lstm_adversarial_attack.attack.best_checkpoint_retriever as bcr
import lstm_adversarial_attack.config_paths as lcp


if __name__ == "__main__":
    if torch.cuda.is_available():
        cur_device = torch.device("cuda:0")
    else:
        cur_device = torch.device("cpu")

    tuning_ranges = ads.AttackTuningRanges(
        kappa=(0.0, 2),
        lambda_1=(1e-7, 1),
        optimizer_name=("Adam", "RMSprop", "SGD"),
        learning_rate=(1e-4, 5e-1),
        log_batch_size=(0, 8),
    )

    checkpoint_retriever = bcr.BestCheckpointRetriever.from_checkpoints_dir(
        checkpoints_dir=lcp.TRAINING_OUTPUT_DIR
        / "2023-06-14_14_40_10.365521"
        / "checkpoints"
    )
    best_checkpoint = checkpoint_retriever.get_extreme_checkpoint(
        metric=bcr.EvalMetric.VALIDATION_LOSS,
        direction=bcr.OptimizeDirection.MIN,
    )

    tuner = aht.AttackHyperParameterTuner(
        device=cur_device,
        model_path=lcp.TRAINING_OUTPUT_DIR
        / "2023-06-14_14_40_10.365521"
        / "model.pickle",
        checkpoint=best_checkpoint,
        epochs_per_batch=1000,
        max_num_samples=16,
        tuning_ranges=tuning_ranges,
    )

    study_result = tuner.tune(num_trials=20)
