import optuna
import torch
from dataclasses import dataclass
from typing import Callable
import lstm_adversarial_attack.config_settings as cfg_set


@dataclass
class AttackTuningRanges:
    kappa: tuple[float, float] = cfg_set.ATTACK_TUNING_KAPPA
    lambda_1: tuple[float, float] = cfg_set.ATTACK_TUNING_LAMBDA_1
    optimizer_name: tuple[str, ...] = cfg_set.ATTACK_TUNING_OPTIMIZER_OPTIONS
    learning_rate: tuple[float, float] = cfg_set.ATTACK_TUNING_LEARNING_RATE
    log_batch_size: tuple[int, int] = cfg_set.ATTACK_TUNING_LOG_BATCH_SIZE


@dataclass
class AttackHyperParameterSettings:
    """
    Container for hyperparameters used by AdversarialAttackTrainer
    :param kappa: arameter from Equation 1 in Sun et al
        (https://arxiv.org/abs/1802.04822). Defines a margin by which alternate
        class logit value needs to exceed original class logit value in order
        to reduce loss function.
    :param lambda_1: L1 regularization constant applied to perturbations
    :param optimizer_name: name of optimizer (must match an attribute of
    torch.optim)
    :param learning_rate: learning rate to use during search for adversarial
    example_data
    :param log_batch_size: log (base 2) of batch size. Use log for easy
    Optuna param selection.

    """

    kappa: float
    lambda_1: float
    optimizer_name: str
    learning_rate: float
    log_batch_size: int

    @property
    def optimizer_constructor(self) -> Callable:
        return getattr(
            torch.optim, self.optimizer_name
        )


class BuildAttackHyperParameterSettings:
    @staticmethod
    def from_optuna_trial(
        trial: optuna.Trial, tuning_ranges: AttackTuningRanges
    ) -> AttackHyperParameterSettings:
        return AttackHyperParameterSettings(
            kappa=trial.suggest_float(
                "kappa", *tuning_ranges.kappa, log=False
            ),
            lambda_1=trial.suggest_float(
                "lambda_1", *tuning_ranges.lambda_1, log=True
            ),
            optimizer_name=trial.suggest_categorical(
                "optimizer_name", list(tuning_ranges.optimizer_name)
            ),
            learning_rate=trial.suggest_float(
                "learning_rate", *tuning_ranges.learning_rate, log=True
            ),
            log_batch_size=trial.suggest_int(
                "log_batch_size", *tuning_ranges.log_batch_size
            ),
        )

