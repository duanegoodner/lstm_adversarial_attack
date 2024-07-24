import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.attack.attack_data_structs as ads
import lstm_adversarial_attack.attack.attack_driver as ad
import lstm_adversarial_attack.attack.attack_result_data_structs as ards
import lstm_adversarial_attack.gpu_helpers as gh
import lstm_adversarial_attack.model.model_retriever as tmr
import lstm_adversarial_attack.path_searches as ps
import lstm_adversarial_attack.preprocess.encode_decode as edc
import lstm_adversarial_attack.tuning_db.tuning_studies_database as tsd
from lstm_adversarial_attack.config import CONFIG_READER


def main(attack_tuning_id: str) -> ards.TrainerSuccessSummary:
    """
    Runs attack on dataset
    :param attack_tuning_id: ID of attack tuning session to use for attack hyperparameter selection
    """
    cur_device = gh.get_device()

    attack_tuning_output_root = Path(
        CONFIG_READER.read_path("attack.tune.output_dir")
    )
    if attack_tuning_id is None:
        attack_tuning_id = ps.get_latest_sequential_child_dirname(
            root_dir=attack_tuning_output_root
        )

    attack_tuner_driver_summary = (
        edc.AttackTunerDriverSummaryReader().import_struct(
            path=attack_tuning_output_root
            / attack_tuning_id
            / f"attack_tuner_driver_summary_{attack_tuning_id}.json"
        )
    )

    attack_tuning_study_name = f"attack_tuning_{attack_tuning_id}"

    attack_hyperparameters_dict = tsd.ATTACK_TUNING_DB.get_best_params(
        study_name=attack_tuning_study_name
    )
    attack_hyperparameters = ads.AttackHyperParameterSettings(
        **attack_hyperparameters_dict
    )

    target_model_checkpoint_info = tmr.ModelRetriever(
        training_output_dir=Path(
            attack_tuner_driver_summary.model_training_result_dir
        )
    ).get_representative_checkpoint()

    attack_driver = ad.AttackDriver(
        target_model_checkpoint_info=target_model_checkpoint_info,
        device=cur_device,
        preprocess_id=attack_tuner_driver_summary.preprocess_id,
        attack_tuning_study_name=attack_tuning_study_name,
        model_hyperparameters=attack_tuner_driver_summary.model_hyperparameters,
        attack_hyperparameters=attack_hyperparameters,
    )

    trainer_result = attack_driver()
    success_summary = ards.TrainerSuccessSummary(trainer_result=trainer_result)

    return success_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--attack_tuning_id",
        action="store",
        nargs="?",
        help=(
            "ID of attack tuning session to use for attack hyperparameter selection"
        ),
    )
    args_namespace = parser.parse_args()
    cur_success_summary = main(**args_namespace.__dict__)
