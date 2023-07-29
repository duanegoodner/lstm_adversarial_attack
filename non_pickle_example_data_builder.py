import collections
import json
from dataclasses import dataclass
from functools import cached_property

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
import lstm_adversarial_attack.tune_train.tuner_helpers as tuh


def verify_parent_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def export_to_json(obj: object, output_path: Path):
    verify_parent_dir(path=output_path)
    with output_path.open(mode="w") as out_file:
        json.dump(obj=obj, fp=out_file)


@dataclass
class ModelTuningResult:
    label: str
    output_dir: Path
    example_save_dir: Path = cfg_paths.EXAMPLE_DATA_DIR / "for_model_training"

    @property
    def optuna_study_path(self) -> Path:
        return self.output_dir / "checkpoints_tuner" / "optuna_study.pickle"

    @cached_property
    def optuna_study_object(self) -> optuna.Study:
        return rio.ResourceImporter().import_pickle_to_object(
            path=self.optuna_study_path
        )

    @property
    def best_params(self) -> dict[str, Any]:
        return self.optuna_study_object.best_params

    def save_data_for_model_training_example(self):
        export_to_json(
            obj=self.best_params,
            output_path=self.example_save_dir
            / self.label
            / "hyperparameters.json",
        )


@dataclass
class CrossValidationResult:
    label: str
    output_dir: Path
    example_save_dir: Path = cfg_paths.EXAMPLE_DATA_DIR / "for_attack_tuning"

    @property
    def hyperparameters_path(self) -> Path:
        return self.output_dir / "hyperparameters.pickle"

    @cached_property
    def hyperparameters_object(self) -> tuh.X19LSTMHyperParameterSettings:
        return rio.ResourceImporter().import_pickle_to_object(
            path=self.hyperparameters_path
        )

    @property
    def hyperparameters_dict(self) -> dict[str, Any]:
        return self.hyperparameters_object.__dict__

    @cached_property
    def best_checkpoint_of_median_performing_fold(self):
        model_retriever = mr.ModelRetriever(
            training_output_dir=self.output_dir
        )
        return model_retriever.get_cv_trained_model(
            metric=cvs.EvalMetric.VALIDATION_LOSS,
            optimize_direction=cvs.OptimizeDirection.MIN,
        ).checkpoint

    @cached_property
    def converted_state_dict_of_median_performing_fold(self) -> OrderedDict:
        original_state_dict = self.best_checkpoint_of_median_performing_fold[
            "state_dict"
        ]
        converted_state_dict = OrderedDict()
        for key, state_tensor in original_state_dict.items():
            converted_state_dict[key] = state_tensor.tolist()

        return converted_state_dict

    @cached_property
    def simplified_checkpoint(self) -> dict[str, Any]:
        return {
            "epoch_num": self.best_checkpoint_of_median_performing_fold[
                "epoch_num"
            ],
            "state_dict": self.converted_state_dict_of_median_performing_fold,
        }

    def save_data_for_attack_tuning_example(self):
        export_to_json(
            obj=self.hyperparameters_dict,
            output_path=self.example_save_dir
            / self.label
            / "hyperparameters.json",
        )
        export_to_json(
            obj=self.simplified_checkpoint,
            output_path=self.example_save_dir
            / self.label
            / "simplified_checkpoint.json",
        )


@dataclass
class AttackTuningResult:
    label: str
    output_dir: Path
    example_save_dir: Path = cfg_paths.EXAMPLE_DATA_DIR / "for_attack"


class ExampleDataBuilder:
    def __init__(
        self,
        # predictive_model_names: tuple[str, ...],
        model_tuning_studies: tuple[ModelTuningResult, ...] = None,
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

    @staticmethod
    def verify_parent_dir(path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def resource_dir_for_model_training(self) -> Path:
        return self.example_root / "for_model_training"

    @property
    def resource_dir_for_attack_tuning(self) -> Path:
        return self.example_root / "for_attack_tuning"

    @property
    def resource_dir_for_model_attack(self) -> Path:
        return self.example_root / "for_model_attack"

    @property
    def resource_dir_for_attack_result_plots(self) -> Path:
        return self.example_root / "for_attack_result_plots"

    def study_best_params_to_json(self, model_tuning_study: ModelTuningResult):
        json_out_path = (
            self.resource_dir_for_model_training
            / model_tuning_study.label
            / "hyperparameters.json"
        )

        self.verify_parent_dir(path=json_out_path)
        with json_out_path.open(mode="w") as out_file:
            json.dump(obj=model_tuning_study.best_params, fp=out_file)

    def save_all_model_tuning_studies_best_params(self):
        for study in self.model_tuning_studies:
            self.study_best_params_to_json(model_tuning_study=study)

    def write_model_training_pre_reqs(self):
        self.save_all_model_tuning_studies_best_params()

    def cv_hyperparameters_to_json(self, cv_result: CrossValidationResult):
        json_out_path = (
            self.resource_dir_for_attack_tuning
            / cv_result.label
            / "hyperparameters.json"
        )
        self.verify_parent_dir(path=json_out_path)
        json_out_path.parent.mkdir(parents=True, exist_ok=True)
        with json_out_path.open(mode="w") as out_file:
            json.dump(obj=cv_result.hyperparameters_dict, fp=out_file)

    def save_all_cv_hyperparameters(self):
        for cv_result in self.cv_results:
            self.cv_hyperparameters_to_json(cv_result=cv_result)

    def simplified_checkpoint_to_json(self, cv_result: CrossValidationResult):
        json_out_path = (
            self.resource_dir_for_attack_tuning
            / cv_result.label
            / "simplified_checkpoint.json"
        )
        self.verify_parent_dir(path=json_out_path)
        with json_out_path.open(mode="w") as out_file:
            json.dump(obj=cv_result.simplified_checkpoint, fp=out_file)

    def write_all_simplified_checkpoints(self):
        for cv_result in self.cv_results:
            self.simplified_checkpoint_to_json(cv_result=cv_result)

    def write_attack_tuning_pre_reqs(self):
        self.save_all_cv_hyperparameters()
        self.write_all_simplified_checkpoints()

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
            ModelTuningResult(
                label="default",
                output_dir=cfg_paths.HYPERPARAMETER_OUTPUT_DIR
                / "continued_trials",
            ),
        ),
        cv_results=(
            CrossValidationResult(
                label="default_cv_result",
                output_dir=cfg_paths.CV_ASSESSMENT_OUTPUT_DIR
                / "2023-06-17_23_57_23.366142",
            ),
        ),
    )

    model_tuning_study = ModelTuningResult(
        label="another_default",
        output_dir=cfg_paths.HYPERPARAMETER_OUTPUT_DIR / "continued_trials",
    )
    model_tuning_study.save_data_for_model_training_example()

    cv_result = CrossValidationResult(
        label="another_default",
        output_dir=cfg_paths.CV_ASSESSMENT_OUTPUT_DIR
        / "2023-06-17_23_57_23.366142",
    )
    cv_result.save_data_for_attack_tuning_example()

    # example_data_builder.write_model_training_pre_reqs()
    # example_data_builder.write_attack_tuning_pre_reqs()
