import argparse
import sys
import optuna
import torch
from os import PathLike
from pathlib import Path
from typing import Callable, Any

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.attack.attack_data_structs as ads
import lstm_adversarial_attack.attack.attack_result_data_structs as ards
import lstm_adversarial_attack.attack.attack_hyperparameter_tuner as aht
import lstm_adversarial_attack.config_paths as cfg_paths
import lstm_adversarial_attack.config_settings as cfg_settings
import lstm_adversarial_attack.resource_io as rio
import lstm_adversarial_attack.tune_train.cross_validation_summarizer as cvs
import lstm_adversarial_attack.attack.model_retriever as amr


class AttackTunerDriver:
    """
    Instantiates and runs (or re-starts) an AttackHyperParameterTuner
    """

    def __init__(
        self,
        device: torch.device,
        target_model_path: Path,
        objective: Callable[..., float],
        target_model_checkpoint: dict,
        objective_extra_kwargs: dict[str, Any] = None,
        tuning_ranges: ads.AttackTuningRanges = None,
        output_dir: Path = None,
        epochs_per_batch: int = cfg_settings.ATTACK_TUNING_EPOCHS,
        max_num_samples: int = cfg_settings.ATTACK_TUNING_MAX_NUM_SAMPLES,
        sample_selection_seed: int = cfg_settings.ATTACK_SAMPLE_SELECTION_SEED
    ):
        """
        :param device: the device to run on
        :param target_model_path: path to .pickle file w/ model to attack
        :param objective: method to user for computation of Optuna tuner
        objective function (typically use one of the methods in
        AttackTunerObjectivesBuilder)
        :param target_model_checkpoint: checkpoint file w/ params to load into
        model under attack
        :param tuning_ranges: hyperparamter tuning ranges (for use by Optuna)
        :param output_dir: directory where results will be saved. If not
        specified, default is timestamped dir under
        data/attack/attack_hyperparamter_tuning
        """
        self.device = device
        self.target_model_path = target_model_path
        self.objective = objective
        self.objective_extra_kwargs = objective_extra_kwargs
        self.target_model_checkpoint = target_model_checkpoint
        if tuning_ranges is None:
            tuning_ranges = ads.AttackTuningRanges(
                kappa=cfg_settings.ATTACK_TUNING_KAPPA,
                lambda_1=cfg_settings.ATTACK_TUNING_LAMBDA_1,
                optimizer_name=cfg_settings.ATTACK_TUNING_OPTIMIZER_OPTIONS,
                learning_rate=cfg_settings.ATTACK_TUNING_LEARNING_RATE,
                log_batch_size=cfg_settings.ATTACK_TUNING_LOG_BATCH_SIZE,
            )
        self.tuning_ranges = tuning_ranges
        if output_dir is None:
            output_dir = rio.create_timestamped_dir(
                parent_path=cfg_paths.ATTACK_HYPERPARAMETER_TUNING
            )
        self.epochs_per_batch = epochs_per_batch
        self.max_num_samples = max_num_samples
        self.output_dir = output_dir
        self.sample_selection_seed = sample_selection_seed
        self.export_dict()
        # rio.ResourceExporter().export(
        #     resource=self, path=self.output_dir / "attack_tuner_driver.pickle"
        # )

    def export_dict(self):
        if not (self.output_dir / "attack_tuner_driver.pickle").exists():
            rio.ResourceExporter().export(
                resource=self,
                path=self.output_dir / "attack_tuner_driver.pickle"
            )

    @classmethod
    def from_model_assessment(
        cls,
        device: torch.device,
        selection_metric: cvs.EvalMetric,
        optimize_direction: cvs.OptimizeDirection,
        objective: Callable[[ards.TrainerSuccessSummary], float],
        objective_extra_kwargs: dict[str, Any] = None,
        training_output_dir: Path = None,
    ):
        """
        Creates an AttackTunerDriver using info from either a cross-validation
        or single-fold assessment of model to be attacked.
        :param device: device to run on
        :param selection_metric: metric for choosing which target model checkpoint to use
        :param optimize_direction: min or max
        :param objective: function that calculates return val of AttackHyperparameterTuner objective_fn
        :param training_output_dir: directory where tuning data is saved
        :return: an AttackTunerDriver instance
        """
        model_retriever = amr.ModelRetriever(
            training_output_dir=training_output_dir,
        )

        model_path_checkpoint_pair = model_retriever.get_model(
            eval_metric=selection_metric,
            optimize_direction=optimize_direction,
        )

        return cls(
            device=device,
            target_model_path=model_path_checkpoint_pair.model_path,
            target_model_checkpoint=model_path_checkpoint_pair.checkpoint,
            objective=objective,
            objective_extra_kwargs=objective_extra_kwargs,
        )

    def run(self, num_trials: int) -> optuna.Study:
        """
        Instantiates and runs an AttackHyperParameterTuner
        :param num_trials:
        :return: an Optuna Study object (this also gets saved in .output_dir)
        """
        tuner = aht.AttackHyperParameterTuner(
            device=self.device,
            model_path=self.target_model_path,
            checkpoint=self.target_model_checkpoint,
            epochs_per_batch=cfg_settings.ATTACK_TUNING_EPOCHS,
            max_num_samples=cfg_settings.ATTACK_TUNING_MAX_NUM_SAMPLES,
            tuning_ranges=self.tuning_ranges,
            output_dir=self.output_dir,
            objective=self.objective,
        )

        return tuner.tune(num_trials=num_trials)

    def restart(self, output_dir: Path, num_trials: int) -> optuna.Study:
        """
        Restarts tuning using params of self. Uses existing AttackDriver.
        Creates new AttackHyperParamterTuner
        :param output_dir: directory containing previous output and where new
        output will be written.
        :param num_trials: max number of trials to run (OK to stop early with
        CTRL-C since results get saved after each trial)
        :return: Optuna Study object
        """
        tuner = aht.AttackHyperParameterTuner(
            device=self.device,
            model_path=self.target_model_path,
            checkpoint=self.target_model_checkpoint,
            epochs_per_batch=self.epochs_per_batch,
            max_num_samples=self.max_num_samples,
            tuning_ranges=self.tuning_ranges,
            continue_study_path=output_dir / "optuna_study.pickle",
            output_dir=output_dir,
            objective=self.objective,
        )

        return tuner.tune(num_trials=num_trials)


def start_new_tuning(
    num_trials: int,
    objective: Callable[..., float],
    objective_extra_kwargs: dict[str, Any] = None,
    target_model_dir: Path = None,
) -> optuna.Study:
    """
    Creates a new AttackTunerDriver. Causes new Optuna Study to be created via
    AttackHyperParamteterTuner that the driver creates.
    :param num_trials: max num Optuna trials to run
    :param objective: method for calculating return val of tuner objective_fn
    from an attack TrainerResult
    :param target_model_dir: directory containing model and params
    files for model to be attacked.
    :return: an Optuna study object (which also get saved as pickle)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    tuner_driver = AttackTunerDriver.from_model_assessment(
        device=device,
        selection_metric=cvs.EvalMetric.VALIDATION_LOSS,
        optimize_direction=cvs.OptimizeDirection.MIN,
        training_output_dir=target_model_dir,
        objective=objective,
        objective_extra_kwargs=objective_extra_kwargs,
    )

    print(
        "Starting new Attack Hyperparameter Tuning study using trained"
        f" predictive model in:\n {tuner_driver.target_model_path}\n\n"
        f"Tuning results will be saved in: {tuner_driver.output_dir}\n"
    )

    return tuner_driver.run(num_trials=num_trials)


def resume_tuning(
    num_trials: int, ongoing_tuning_dir: Path = None
) -> optuna.Study:
    """
    Resumes training using params of a previously used AttackTunerDriver and
    its associated Optuna Study. Default behavior saves new results to
    same directory as results of previous runs.
    :param num_trials: max # of trials to run
    :param ongoing_tuning_dir: directory where previous run data is saved
    and (under default settings) where new data will be saved.
    :return: an Optuna Study object (which also gets saved as .pickle)
    """
    if ongoing_tuning_dir is None:
        ongoing_tuning_dir = cvs.get_newest_sub_dir(
            path=cfg_paths.ATTACK_HYPERPARAMETER_TUNING
        )

    reloaded_tuner_driver_dict = (
        rio.ResourceImporter().import_pickle_to_object(
            path=ongoing_tuning_dir / "tuner_driver_dict.pickle"
        )
    )
    reloaded_tuner_driver = AttackTunerDriver(**reloaded_tuner_driver_dict)

    # reloaded_tuner_driver = rio.ResourceImporter().import_pickle_to_object(
    #     path=ongoing_tuning_dir / "attack_tuner_driver.pickle"
    # )

    print(
        "Resuming Attack Hyperparameter Tuning study data in:\n"
        f"{reloaded_tuner_driver.output_dir}\n"
    )

    return reloaded_tuner_driver.restart(
        output_dir=ongoing_tuning_dir, num_trials=num_trials
    )


def main(
    target_model_dir: str = None,
    existing_study_dir: str = None,
    num_trials: int = None,
    objective_name: str = None,
    max_perts: int = None,
) -> optuna.Study:
    """
    Tunes hyperparameters of an AdversarialAttackTrainer and its
    AdversarialAttacker. Can accept target_model_dir OR existing_study_dir or
    neither, but not both. If no args provided, starts new study using most
    recent cross-validation training results to build target model.

    :param target_model_dir: Directory containing model training results to
    use to create new attack target.
    :param existing_study_dir: directory containing an existing optuna.Study
    :param num_trials: number of trials to run. defaults to
    config_settings.ATTACK_TUNING_DEFAULT_NUM_TRIALS
    :param objective_name: name of method to use from AttackTunerObjectives
    :param max_perts: value to use as the max_num_perts arg when using
    AttackTunerObjectives.max_num_nonzero_perts (not needed for other
    objectives)
    :return: an optuna.Study object with results of completed trials
    """
    assert target_model_dir is None or existing_study_dir is None
    if objective_name == "max_num_nonzero_perts":
        assert max_perts > 0

    # Use if ______ is None pattern instead of default args for easier
    # integration with CLI and argparse

    if num_trials is None:
        num_trials = cfg_settings.ATTACK_TUNING_DEFAULT_NUM_TRIALS
    target_model_dir_path = (
        None if target_model_dir is None else Path(target_model_dir)
    )

    if objective_name is None:
        objective_name = cfg_settings.ATTACK_TUNING_DEFAULT_OBJECTIVE

    objective = getattr(aht.AttackTunerObjectives, objective_name)

    objective_extra_kwargs = (
        {"max_perts": max_perts} if max_perts is not None else {}
    )

    if existing_study_dir is None:
        initial_study = start_new_tuning(
            num_trials=num_trials,
            objective=objective,
            target_model_dir=target_model_dir_path,
            objective_extra_kwargs=objective_extra_kwargs,
        )
        return initial_study

    else:
        continued_study = resume_tuning(
            num_trials=num_trials, ongoing_tuning_dir=Path(existing_study_dir)
        )
        return continued_study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Runs hyperparameter tuning on the attack algorithm. "
            "If no args passed, will start a new Optuna study "
            "using the model data from the most recent data saved in "
            "data/tune_train/cross_validation (uses fold with "
            "median or near median best performance)"
        )
    )
    parser.add_argument(
        "-m",
        "--target_model_dir",
        type=str,
        action="store",
        nargs="?",
        help=(
            "Path to CV assessment directory with CV data and trained params "
            "to use for target model. If value passed for this arg, cannot "
            "use the -s (existing_study_dir) option."
        ),
    )
    parser.add_argument(
        "-n",
        "--num_trials",
        type=int,
        action="store",
        nargs="?",
        help="Number of trials to run",
    )

    parser.add_argument(
        "-o",
        "--objective_name",
        type=str,
        action="store",
        nargs="?",
        help=(
            "Name of method from AttackTunerObjectives to use to calculate "
            "return value at end of optuna objective_fn. Strongly affects the"
            " type of perturbation that tuning will favor. Defaults to value "
            "of cfg_settings.ATTACK_TUNING_DEFAULT_OBJECTIVE"
        ),
    )

    parser.add_argument(
        "-p",
        "--max_perts",
        type=int,
        action="store",
        nargs="?",
        help=(
            "Value to use for the max_perts argument when using "
            "'max_num_nonzero_perts' as the objective."
        ),
    )

    parser.add_argument(
        "-s",
        "--existing_study_dir",
        type=str,
        action="store",
        nargs="?",
        help=(
            "Path to parent directory of existing Optuna study to use as "
            "starting point for continued (ongoing) training. If value is "
            "provided for this arg, cannot use the -m (target_model_dir)"
            " option."
        ),
    )

    args_namespace = parser.parse_args()
    args_namespace.existing_study_dir = str(
        cfg_paths.ATTACK_HYPERPARAMETER_TUNING / "2023-06-28_12_11_46.874267"
    )
    main(**args_namespace.__dict__)
