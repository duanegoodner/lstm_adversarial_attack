import collections
import json
import optuna
import sys
import torch
from collections import OrderedDict
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import lstm_adversarial_attack.attack.model_retriever as mr
import lstm_adversarial_attack.config_paths as cfg_paths
import lstm_adversarial_attack.resource_io as rio
import lstm_adversarial_attack.tune_train.cross_validation_summarizer as cvs


def best_params_optuna_study_to_json(
    study: optuna.Study, output_path: Path
) -> Path:
    with output_path.open(mode="w") as outfile:
        json.dump(study.best_params, outfile)
    return output_path


def best_params_optuna_study_path_to_json(
    study_path: Path, output_path: Path
) -> Path:
    study = rio.ResourceImporter().import_pickle_to_object(path=study_path)
    return best_params_optuna_study_to_json(
        study=study, output_path=output_path
    )


def state_dict_to_json(state_dict: OrderedDict, json_out_path: Path):
    converted_dict = {
        key: state_tensor.tolist() for key, state_tensor in state_dict.items()
    }
    with json_out_path.open(mode="w") as json_file:
        json.dump(obj=converted_dict, fp=json_file)

    return json_out_path


def cv_median_fold_state_dict_to_json(
    training_output_dir: Path, json_out_path: Path
):
    model_retriever = mr.ModelRetriever(
        training_output_dir=training_output_dir
    )
    modelpath_fold_checkpoint = model_retriever.get_cv_trained_model(
        metric=cvs.EvalMetric.VALIDATION_LOSS,
        optimize_direction=cvs.OptimizeDirection.MIN,
    )

    state_dict = modelpath_fold_checkpoint.checkpoint["state_dict"]

    return state_dict_to_json(
        state_dict=state_dict, json_out_path=json_out_path
    )


def json_to_ordered_dict(json_path: Path) -> OrderedDict:
    with json_path.open(mode="r") as json_file:
        loaded_dict = json.load(json_file)

    final_dict = collections.OrderedDict()
    for key, nested_list in loaded_dict.items():
        final_dict[key] = torch.tensor(nested_list, dtype=torch.float32)

    return final_dict


def attack_trainer_result_path_to_json(
    trainer_result_path: Path, json_out_path: Path
) -> Path:
    trainer_result = rio.ResourceImporter().import_pickle_to_object(
        path=trainer_result_path
    )
    with json_out_path.open(mode="w") as json_file:
        json.dump(obj=trainer_result.__dict__, fp=json_file)

    return json_out_path


if __name__ == "__main__":
    # tuned hyperparameters for training predictive model
    # best_params_optuna_study_path_to_json(
    #     study_path=cfg_paths.HYPERPARAMETER_OUTPUT_DIR
    #     / "continuing_study"
    #     / "checkpoints_tuner"
    #     / "optuna_study.pickle",
    #     output_path=cfg_paths.EXAMPLE_DATA_DIR
    #     / "trained_predictive_model"
    #     / "tuned_hyperparameters.json",
    # )

    # state dict for trained predictive model
    cv_training_output_dir = (
        cfg_paths.CV_ASSESSMENT_OUTPUT_DIR / "2023-06-17_23_57_23.366142"
    )
    my_state_dict_json_path = (
        cfg_paths.EXAMPLE_DATA_DIR
        / "trained_predictive_model"
        / "state_dict.json"
    )
    cv_median_fold_state_dict_to_json(
        training_output_dir=cv_training_output_dir,
        json_out_path=my_state_dict_json_path,
    )

    best_params_optuna_study_path_to_json(
        study_path=cfg_paths.ATTACK_HYPERPARAMETER_TUNING
        / "2023-06-28_12_11_46.874267"
        / "optuna_study.pickle",
        output_path=cfg_paths.EXAMPLE_DATA_DIR
        / "tuned_attack_hyperparameters"
        / "2023-06-28_12_11_46.874267"
        / "tuned_hyperparameters.json",
    )

    best_params_optuna_study_path_to_json(
        study_path=cfg_paths.ATTACK_HYPERPARAMETER_TUNING
        / "2023-06-30_10_52_24.059982"
        / "optuna_study.pickle",
        output_path=cfg_paths.EXAMPLE_DATA_DIR
        / "tuned_attack_hyperparameters"
        / "22023-06-30_10_52_24.059982"
        / "tuned_hyperparameters.json",
    )

    best_params_optuna_study_path_to_json(
        study_path=cfg_paths.ATTACK_HYPERPARAMETER_TUNING
        / "2023-07-01_11_03_13.591090"
        / "optuna_study.pickle",
        output_path=cfg_paths.EXAMPLE_DATA_DIR
        / "tuned_attack_hyperparameters"
        / "2023-07-01_11_03_13.591090"
        / "tuned_hyperparameters.json",
    )
