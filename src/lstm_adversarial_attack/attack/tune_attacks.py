import sys
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.attack.attack_data_structs as ads
import lstm_adversarial_attack.attack.attack_hyperparameter_tuner as aht
import lstm_adversarial_attack.config_paths as lcp
import lstm_adversarial_attack.tune_train.cross_validation_summarizer as cvs


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

    fold_summarizer = cvs.FoldSummarizer.from_fold_checkpoint_dir(
        fold_checkpoint_dir=lcp.TRAINING_OUTPUT_DIR
        / "2023-06-17_18_43_05.989001"
        / "checkpoints"
    )

    best_checkpoint = fold_summarizer.get_extreme_checkpoint(
        metric=cvs.EvalMetric.VALIDATION_LOSS,
        optimize_direction=cvs.OptimizeDirection.MIN
    )

    tuner = aht.AttackHyperParameterTuner(
        device=cur_device,
        model_path=lcp.TRAINING_OUTPUT_DIR / "2023-06-17_18_43_05.989001" /
        "model.pickle",
        checkpoint=best_checkpoint,
        epochs_per_batch=1000,
        max_num_samples=16,
        tuning_ranges=tuning_ranges
    )

    study_result = tuner.tune(num_trials=20)
