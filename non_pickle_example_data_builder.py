import collections
import json
import optuna
import sys
import torch
from collections import OrderedDict
from pathlib import Path
from typing import Any, NamedTuple

sys.path.append(str(Path(__file__).parent.parent))
import lstm_adversarial_attack.attack.model_retriever as mr
import lstm_adversarial_attack.config_paths as cfg_paths
import lstm_adversarial_attack.resource_io as rio
import lstm_adversarial_attack.tune_train.cross_validation_summarizer as cvs


class ModelTuningStudy(NamedTuple):
    label: str
    study_file_path: Path


class CrossValidationResult(NamedTuple):
    label: str
    output_dir: Path


class ExampleDataBuilder:
    def __init__(
        self,
        # predictive_model_names: tuple[str, ...],
        model_tuning_studies: tuple[ModelTuningStudy, ...] = None,
        example_root: Path = cfg_paths.EXAMPLE_DATA_DIR,
        cv_results: tuple[CrossValidationResult, ...] = None,
        # attack_tuning_studies: tuple[Path, ...],
        # attack_trainer_results: tuple[Path, ...],
    ):
        #     self.predictive_model_names = predictive_model_names
        self.model_tuning_studies = model_tuning_studies
        self.cv_results = cv_results
        #     self.attack_tuning_studies = attack_tuning_studies
        #     self.attack_trainer_results = attack_trainer_results
        self.example_root = example_root

    def study_best_params_to_json(self, model_tuning_study: ModelTuningStudy):
        study = rio.ResourceImporter().import_pickle_to_object(
            path=model_tuning_study.study_file_path
        )

        json_out_path = (
            self.example_root
            / "for_model_training"
            / model_tuning_study.label
            / "hyperparameters.json"
        )
        json_out_path.parent.mkdir(parents=True, exist_ok=True)
        with json_out_path.open(mode="w") as out_file:
            json.dump(obj=study.best_params, fp=out_file)

    def save_all_model_tuning_studies_best_params(self):
        for study in self.model_tuning_studies:
            self.study_best_params_to_json(model_tuning_study=study)

    def write_model_training_pre_reqs(self):
        self.save_all_model_tuning_studies_best_params()

    def median_fold_reduced_checkpoint_to_json(
        self, cv_result: CrossValidationResult
    ):
        model_retriever = mr.ModelRetriever(
            training_output_dir=cv_result.output_dir
        )
        modelpath_fold_checkpoint = model_retriever.get_cv_trained_model(
            metric=cvs.EvalMetric.VALIDATION_LOSS,
            optimize_direction=cvs.OptimizeDirection.MIN,
        )

        converted_state_dict = {
            key: state_tensor.tolist()
            for key, state_tensor in modelpath_fold_checkpoint.checkpoint[
                "state_dict"
            ].items()
        }

        reduced_checkpoint = {
            "epoch_num": modelpath_fold_checkpoint.checkpoint["epoch_num"],
            "state_dict": converted_state_dict,
        }

        with (
            self.example_root
            / "for_attack_hyperparameter_tuning"
            / f"{cv_result.label}.json"
        ).open(mode="w") as out_file:
            json.dump(obj=reduced_checkpoint, fp=out_file)

    def save_reduced_checkpoints_for_all_cv_results(self):
        for cv_result in self.cv_results:
            self.median_fold_reduced_checkpoint_to_json(cv_result=cv_result)


if __name__ == "__main__":
    example_data_builder = ExampleDataBuilder(
        model_tuning_studies=(
            ModelTuningStudy(
                label="default",
                study_file_path=cfg_paths.HYPERPARAMETER_OUTPUT_DIR
                / "continued_trials"
                / "checkpoints_tuner"
                / "optuna_study.pickle",
            ),
        ),
        # cv_results=(
        #     CrossValidationResult(
        #         label="default_cv_result",
        #         output_dir=cfg_paths.CV_ASSESSMENT_OUTPUT_DIR
        #         / "2023-06-17_23_57_23.366142",
        #     ),
        # ),
    )

    # example_data_builder.save_all_model_tuning_studies_best_params()
    # example_data_builder.save_reduced_checkpoints_for_all_cv_results()
    example_data_builder.write_model_training_pre_reqs()
