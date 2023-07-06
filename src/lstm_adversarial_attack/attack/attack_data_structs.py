import optuna
from dataclasses import dataclass


@dataclass
class AttackTuningRanges:
    kappa: tuple[float, float]
    lambda_1: tuple[float, float]
    optimizer_name: tuple[str, ...]
    learning_rate: tuple[float, float]
    log_batch_size: tuple[int, int]


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
    examples
    :param log_batch_size: log (base 2) of batch size. Use log for easy
    Optuna param selection.

    """
    kappa: float
    lambda_1: float
    optimizer_name: str
    learning_rate: float
    log_batch_size: int

    @classmethod
    def from_optuna_active_trial(
        cls, trial: optuna.Trial, tuning_ranges: AttackTuningRanges
    ):
        """
        Creates an AttackDriver using an active optuna Trial. Uses methods of
        trial to select specific hyperparams from tuning ranges.
        :param trial: an optuna Trial ()
        :param tuning_ranges: AttackTuningRanges dataclass object with range
        of hyperparameters to search
        """
        return cls(
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
