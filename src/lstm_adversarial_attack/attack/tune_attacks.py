import sys
import optuna
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.attack.attack_data_structs as ads
import lstm_adversarial_attack.attack.attack_hyperparameter_tuner as aht
import lstm_adversarial_attack.config_paths as cfg_paths
import lstm_adversarial_attack.config_settings as cfg_settings
import lstm_adversarial_attack.resource_io as rio
import lstm_adversarial_attack.tune_train.cross_validation_summarizer as cvs
import lstm_adversarial_attack.attack.model_retriever as amr


class AttackTunerDriver:
    def __init__(
        self,
        device: torch.device,
        target_model_path: Path,
        target_model_checkpoint: dict,
        tuning_ranges: ads.AttackTuningRanges = None,
        output_dir: Path = None,
    ):
        self.device = device
        self.target_model_path = target_model_path
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
        self.output_dir = output_dir
        rio.ResourceExporter().export(
            resource=self, path=self.output_dir / "attack_tuner_driver.pickle"
        )

    @classmethod
    def from_model_assessment(
        cls,
        device: torch.device,
        assessment_type: amr.ModelAssessmentType,
        selection_metric: cvs.EvalMetric,
        optimize_direction: cvs.OptimizeDirection,
        training_output_dir: Path = None,
    ):
        model_retriever = amr.ModelRetriever(
            assessment_type=assessment_type,
            training_output_dir=training_output_dir,
        )

        model_path_checkpoint_pair = model_retriever.get_model(
            assessment_type=assessment_type,
            eval_metric=selection_metric,
            optimize_direction=optimize_direction,
        )

        return cls(
            device=device,
            target_model_path=model_path_checkpoint_pair.model_path,
            target_model_checkpoint=model_path_checkpoint_pair.checkpoint,
        )

    def run(self, num_trials: int) -> optuna.Study:
        tuner = aht.AttackHyperParameterTuner(
            device=self.device,
            model_path=self.target_model_path,
            checkpoint=self.target_model_checkpoint,
            epochs_per_batch=cfg_settings.ATTACK_TUNING_EPOCHS,
            max_num_samples=cfg_settings.ATTACK_TUNING_MAX_NUM_SAMPLES,
            tuning_ranges=self.tuning_ranges,
            output_dir=self.output_dir,
        )

        return tuner.tune(num_trials=num_trials)

    def restart(self, output_dir: Path, num_trials: int) -> optuna.Study:
        tuner = aht.AttackHyperParameterTuner(
            device=self.device,
            model_path=self.target_model_path,
            checkpoint=self.target_model_checkpoint,
            epochs_per_batch=cfg_settings.ATTACK_TUNING_EPOCHS,
            max_num_samples=cfg_settings.ATTACK_TUNING_MAX_NUM_SAMPLES,
            tuning_ranges=self.tuning_ranges,
            continue_study_path=output_dir / "optuna_study.pickle",
            output_dir=output_dir,
        )

        return tuner.tune(num_trials=num_trials)


def start_new_tuning(
    num_trials: int,
    target_model_assessment_type: amr.ModelAssessmentType,
    target_model_assessment_dir: Path = None,
) -> optuna.Study:
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    tuner_driver = AttackTunerDriver.from_model_assessment(
        device=device,
        assessment_type=target_model_assessment_type,
        selection_metric=cvs.EvalMetric.VALIDATION_LOSS,
        optimize_direction=cvs.OptimizeDirection.MIN,
        training_output_dir=target_model_assessment_dir,
    )

    print(
        "Starting new Attack Hyperparameter Tuning study using trained"
        f" predictive model in:\n {tuner_driver.target_model_path}\n\n"
        f"Tuning results will be saved in: {tuner_driver.output_dir}\n"
    )

    return tuner_driver.run(num_trials=num_trials)


def resume_tuning(ongoing_tuning_dir: Path = None) -> optuna.Study:
    if ongoing_tuning_dir is None:
        ongoing_tuning_dir = cvs.get_newest_sub_dir(
            path=cfg_paths.ATTACK_HYPERPARAMETER_TUNING
        )

    reloaded_tuner_driver = rio.ResourceImporter().import_pickle_to_object(
        path=ongoing_tuning_dir / "attack_tuner_driver.pickle"
    )

    print(
        f"Resuming Attack Hyperparameter Tuning study data in:\n"
        f"{reloaded_tuner_driver.output_dir}\n"
    )

    return reloaded_tuner_driver.restart(
        output_dir=ongoing_tuning_dir, num_trials=20
    )


if __name__ == "__main__":
    initial_study = start_new_tuning(
        num_trials=50,
        target_model_assessment_type=amr.ModelAssessmentType.KFOLD,
    )

    # continued_study = resume_tuning()
