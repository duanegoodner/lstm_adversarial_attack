import argparse
import sys
from pathlib import Path

import optuna

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.attack.attack_tuner_driver as atd
import lstm_adversarial_attack.gpu_helpers as gh
import lstm_adversarial_attack.path_searches as ps
import lstm_adversarial_attack.preprocess.encode_decode as edc
import lstm_adversarial_attack.tuning_db.tuning_studies_database as tsd
from lstm_adversarial_attack.config import CONFIG_READER


def main(
    # study_name: str = None,
    attack_tuning_id: str = None,
) -> optuna.Study:
    """
    Tunes hyperparameters of an AdversarialAttackTrainer and its
    AdversarialAttacker. Can accept target_model_dir OR existing_study_dir or
    neither, but not both. If no args provided, starts new study using most
    recent cross-validation training results to build target model.

    :param study_name: name of tuning study (in RDB)
    :return: an optuna.Study object with results of completed trials
    """

    # if study_name is None:
    #     study_name = tsd.ATTACK_TUNING_DB.get_latest_study().study_name

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

    # study_dir = (
    #     Path(CONFIG_READER.read_path("attack.tune.output_dir")) / study_name
    # )

    cur_device = gh.get_device()

    # attack_tuner_driver_summary_path = (
    #     ps.latest_modified_file_with_name_condition(
    #         component_string="attack_tuner_driver_summary_",
    #         root_dir=study_dir,
    #         comparison_type=ps.StringComparisonType.PREFIX,
    #     )
    # )
    # attack_tuner_driver_summary = (
    #     edc.AttackTunerDriverSummaryReader().import_struct(
    #         path=attack_tuner_driver_summary_path
    #     )
    # )

    study_name = f"attack_tuning_{attack_tuning_id}"

    attack_tuner_driver = atd.AttackTunerDriver(
        device=cur_device,
        preprocess_id=attack_tuner_driver_summary.preprocess_id,
        attack_tuning_id=attack_tuning_id,
        model_hyperparameters=attack_tuner_driver_summary.model_hyperparameters,
        settings=attack_tuner_driver_summary.settings,
        paths=attack_tuner_driver_summary.paths,
        study_name=study_name,
        tuning_ranges=attack_tuner_driver_summary.tuning_ranges,
        model_training_result_dir=Path(attack_tuner_driver_summary.model_training_result_dir),
    )

    # partial_constructor_kwargs = {
    #     key: val
    #     for key, val in attack_tuner_driver_summary.to_dict().items()
    #     if key not in ["is_continuation"]
    # }
    #
    # constructor_kwargs = {
    #     **{"device": cur_device, **partial_constructor_kwargs}
    # }
    #
    # attack_tuner_driver = atd.AttackTunerDriver(**constructor_kwargs)
    optuna.logging.set_verbosity(optuna.logging.INFO)
    continued_study = attack_tuner_driver.run()
    return continued_study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Runs hyperparameter tuning on the attack algorithm. "
            "If no args passed, will start a new Optuna study "
            "using the model data from the most recent data saved in "
            "data/model/cross_validation (uses fold with "
            "median or near median best performance)"
        )
    )
    parser.add_argument(
        "-t",
        "--attack_tuning_id",
        type=str,
        action="store",
        nargs="?",
        help="ID of attack tuning session to resume",
    )

    args_namespace = parser.parse_args()
    main(**args_namespace.__dict__)
