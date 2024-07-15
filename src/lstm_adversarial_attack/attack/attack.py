import argparse
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.attack.attack_driver as ad
import lstm_adversarial_attack.attack.attack_result_data_structs as ards
import lstm_adversarial_attack.config_paths as cfg_paths
import lstm_adversarial_attack.config_settings as cfg_settings
import lstm_adversarial_attack.path_searches as ps
import lstm_adversarial_attack.tuning_db.tuning_studies_database as tsd
import lstm_adversarial_attack.preprocess.encode_decode as edc




def main(study_name: str) -> ards.TrainerSuccessSummary:
    """
    Runs attack on dataset
    :param study_name: name of tuning study to use for hyperparameter selection
    """
    if torch.cuda.is_available():
        cur_device = torch.device("cuda:0")
    else:
        cur_device = torch.device("cpu")

    if study_name is None:
        study_name = tsd.ATTACK_TUNING_DB.get_latest_study().study_name

    attack_hyperparameters = tsd.ATTACK_TUNING_DB.get_best_params(
        study_name=study_name
    )

    tuning_result_dir_path = (
        cfg_paths.ATTACK_HYPERPARAMETER_TUNING / study_name
    )

    model_hyperparameters = (
        edc.X19LSTMHyperParameterSettingsReader().import_struct(
            path=tuning_result_dir_path / "model_hyperparameters.json"
        )
    )

    attack_tuner_driver_summary_path = (
        ps.latest_modified_file_with_name_condition(
            component_string="attack_tuner_driver_summary_",
            root_dir=tuning_result_dir_path,
            comparison_type=ps.StringComparisonType.PREFIX,
        )
    )

    attack_tuner_driver_summary = (
        edc.AttackTunerDriverSummaryReader().import_struct(
            path=attack_tuner_driver_summary_path
        )
    )

    # TODO resume work here on delay targetmodel instantiation after modify
    #  ModelRetriever to also provide checkpt Path. (& move call to
    #  get_representative_checkpoint back to tune_attack_new.py)
    # attack_driver = AttackDriver(
    #     device=cur_device,
    #     model_hyperparameters=model_hyperparameters,
    #     checkpoint=attack_tuner_driver_summary.
    #
    # )

    # tuning_result_dir_path = (
    #     Path(tuning_result_dir) if tuning_result_dir is not None else None
    # )

    attack_driver = ad.AttackDriver.from_attack_hyperparameter_tuning(
        device=cur_device,
        sample_selection_seed=cfg_settings.ATTACK_SAMPLE_SELECTION_SEED,
        checkpoint_interval=cfg_settings.ATTACK_CHECKPOINT_INTERVAL,
        tuning_result_dir=tuning_result_dir_path,
    )
    trainer_result = attack_driver()
    success_summary = ards.TrainerSuccessSummary(trainer_result=trainer_result)

    return success_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--study_name",
        action="store",
        nargs="?",
        help=(
            "Name of attack tuning study to use for attack hyperparameter "
            "selection"
        ),
    )
    args_namespace = parser.parse_args()
    # args_namespace.tuning_result_dir = str(
    #     cfg_paths.ATTACK_HYPERPARAMETER_TUNING / "2023-07-01_11_03_13.591090"
    # )
    cur_success_summary = main(**args_namespace.__dict__)
