import torch


from attack_hyperparameter_tuner import (
    AttackTuningRanges,
    AttackHyperParameterTuner,
)
from lstm_adversarial_attack.config_paths import DEFAULT_ATTACK_TARGET_DIR


if __name__ == "__main__":
    if torch.cuda.is_available():
        cur_device = torch.device("cuda:0")
    else:
        cur_device = torch.device("cpu")

    tuning_ranges = AttackTuningRanges(
        kappa=(0.0, 2),
        lambda_1=(1e-7, 1),
        optimizer_name=("Adam", "RMSprop", "SGD"),
        learning_rate=(1e-4, 5e-1),
        log_batch_size=(0, 8)
    )

    checkpoint_files = list(DEFAULT_ATTACK_TARGET_DIR.glob("*.tar"))
    assert len(checkpoint_files) == 1
    checkpoint_path = checkpoint_files[0]

    tuner = AttackHyperParameterTuner(
        device=cur_device,
        model_path=DEFAULT_ATTACK_TARGET_DIR / "model.pickle",
        checkpoint_path=checkpoint_path,
        epochs_per_batch=500,
        max_num_samples=250,
        tuning_ranges=tuning_ranges
    )

    study_result = tuner.tune(num_trials=20)
