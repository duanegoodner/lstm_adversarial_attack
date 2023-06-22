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
        self.output_dir = self.initialize_output_dir()

    def initialize_output_dir(self):
        initialized_output_dir = rio.create_timestamped_dir(
            parent_path=cfg_paths.ATTACK_HYPERPARAMETER_TUNING
        )
        rio.ResourceExporter().export(
            resource=self,
            path=initialized_output_dir / "attack_tuner_driver.pickle",
        )
        return initialized_output_dir

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
            output_dir=self.output_dir
        )

        return tuner.tune(num_trials=num_trials)


if __name__ == "__main__":
    if torch.cuda.is_available():
        cur_device = torch.device("cuda:0")
    else:
        cur_device = torch.device("cpu")

    tuner_driver = AttackTunerDriver.from_model_assessment(
        device=cur_device,
        assessment_type=amr.ModelAssessmentType.KFOLD,
        selection_metric=cvs.EvalMetric.VALIDATION_LOSS,
        optimize_direction=cvs.OptimizeDirection.MIN,
    )

    study_result = tuner_driver.run(num_trials=20)
