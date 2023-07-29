import json
from dataclasses import dataclass
from functools import cached_property

import optuna
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

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
class ModelTrainingPreReqArchiver:
    label: str
    model_tuning_result_dir: Path
    example_save_dir: Path = cfg_paths.EXAMPLE_DATA_DIR / "for_model_training"

    @property
    def optuna_study_path(self) -> Path:
        return (
            self.model_tuning_result_dir
            / "checkpoints_tuner"
            / "optuna_study.pickle"
        )

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
            / "model_training_hyperparameters.json",
        )


@dataclass
class AttackTuningPreReqArchiver:
    label: str
    model_training_result_dir: Path
    example_save_dir: Path = cfg_paths.EXAMPLE_DATA_DIR / "for_attack_tuning"

    @property
    def hyperparameters_path(self) -> Path:
        return self.model_training_result_dir / "hyperparameters.pickle"

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
            training_output_dir=self.model_training_result_dir
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
            / "model_training_hyperparameters.json",
        )
        export_to_json(
            obj=self.simplified_checkpoint,
            output_path=self.example_save_dir
            / self.label
            / "simplified_model_training_checkpoint.json",
        )


@dataclass
class AttackPreReqArchiver:
    label: str
    model_training_result_dir: Path
    attack_tuning_result_dir: Path
    example_save_dir: Path = cfg_paths.EXAMPLE_DATA_DIR / "for_attack"
    attack_tuning_pre_req_archiver: AttackTuningPreReqArchiver = None

    def __post_init__(self):
        if self.attack_tuning_pre_req_archiver is None:
            self.attack_tuning_pre_req_archiver = AttackTuningPreReqArchiver(
                label=self.label,
                model_training_result_dir=self.model_training_result_dir,
                example_save_dir=self.example_save_dir,
            )

    @property
    def attack_tuning_provenance_path(self) -> Path:
        return self.attack_tuning_result_dir / "provenance.pickle"

    @property
    def attack_tuning_provenance_object(self) -> object:
        return rio.ResourceImporter().import_pickle_to_object(
            path=self.attack_tuning_provenance_path
        )

    @property
    def optuna_study_path(self) -> Path:
        return self.attack_tuning_result_dir / "optuna_study.pickle"

    @cached_property
    def optuna_study_object(self) -> optuna.Study:
        return rio.ResourceImporter().import_pickle_to_object(
            path=self.optuna_study_path
        )

    @property
    def best_attack_hyperparams(self) -> dict[str, Any]:
        return self.optuna_study_object.best_params

    def save_data_for_attack_example(self):
        self.attack_tuning_pre_req_archiver.save_data_for_attack_tuning_example()
        export_to_json(
            obj=self.best_attack_hyperparams,
            output_path=self.example_save_dir
            / self.label
            / "attack_hyperparameters.json",
        )


@dataclass
class PlotPreReqArchiver:
    label: str
    attack_result_dir: Path
    example_save_dir: Path = cfg_paths.EXAMPLE_DATA_DIR / "for_plotting"




if __name__ == "__main__":
    model_training_pre_req_archiver = ModelTrainingPreReqArchiver(
        label="default",
        model_tuning_result_dir=cfg_paths.HYPERPARAMETER_OUTPUT_DIR
        / "continued_trials",
    )
    model_training_pre_req_archiver.save_data_for_model_training_example()

    attack_tuning_pre_req_archiver = AttackTuningPreReqArchiver(
        label="default",
        model_training_result_dir=cfg_paths.CV_ASSESSMENT_OUTPUT_DIR
        / "2023-06-17_23_57_23.366142",
    )
    attack_tuning_pre_req_archiver.save_data_for_attack_tuning_example()

    attack_pre_req_archiver = AttackPreReqArchiver(
        label="max_single_element_perts",
        model_training_result_dir=cfg_paths.CV_ASSESSMENT_OUTPUT_DIR
        / "2023-06-17_23_57_23.366142",
        attack_tuning_result_dir=cfg_paths.ATTACK_HYPERPARAMETER_TUNING
        / "2023-06-28_12_11_46.874267",
    )
    attack_pre_req_archiver.save_data_for_attack_example()

    # example_data_builder.write_model_training_pre_reqs()
    # example_data_builder.write_attack_tuning_pre_reqs()
