import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.attack.attack as atk
import lstm_adversarial_attack.attack.attack_result_data_structs as ards
import lstm_adversarial_attack.attack.attack_hyperparameter_tuner as aht
import lstm_adversarial_attack.attack.model_retriever as amr
import lstm_adversarial_attack.attack.tune_attacks as tua


def main() -> ards.TrainerSuccessSummary:
    """
    Tunes hyperparameters of an AdversarialAttackTrainer. Results saved in
    directory specified by config_paths.ATTACK_HYPERPARAMETER_TUNING.
    :return: a TrainerSuccessSummary with attack results and 1st phase analysis
    """
    tua.start_new_tuning(
        num_trials=100,
        target_model_assessment_type=amr.ModelAssessmentType.KFOLD,
        objective=aht.AttackTunerObjectivesBuilder.max_num_nonzero_perts(
            max_perts=1
        ),
    )
    success_summary = atk.attack_with_tuned_params()
    return success_summary


if __name__ == "__main__":
    cur_success_summary = main()





