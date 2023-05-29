import time
import torch
from pathlib import Path
from torch.utils.data import Subset
from adv_attack import AdversarialAttacker, AdversarialAttackTrainer
from lstm_model_stc_old import LSTMSun2018
from lstm_sun_2018_logit_out import LSTMSun2018Logit
from single_sample_feature_perturber import SingleSampleFeaturePerturber
from standard_model_inferrer import StandardModelInferrer
from dataset_full48_m19 import Full48M19DatasetWithIndex
from adv_attack import AdvAttackExperimentRunner

if __name__ == "__main__":
    if torch.cuda.is_available():
        cur_device = torch.device("cuda:0")
    else:
        cur_device = torch.device("cpu")

    # Instantiate a full model and load pretrained parameters
    pretrained_full_model = LSTMSun2018(model_device=cur_device)
    checkpoint_path = Path(
        "/home/duane/dproj/UIUC-DLH/project/ehr_adversarial_attack/data"
        "/cross_validate_sun2018_full48m19_01/2023-05-07_23:32:09.938445.tar"
    )
    checkpoint = torch.load(checkpoint_path)
    pretrained_full_model.load_state_dict(
        checkpoint["state_dict"], strict=False
    )

    # Run data through pretrained model & get all correctly predicted samples
    full_dataset = (
        Full48M19DatasetWithIndex.from_feature_finalizer_output()
    )
    inferrer = StandardModelInferrer(
        model=pretrained_full_model, dataset=full_dataset
    )
    correctly_predicted_data = inferrer.get_correctly_predicted_samples()
    small_correctly_predicted_data = Subset(
        dataset=correctly_predicted_data, indices=list(range(10))
    )

    # Instantiate version of model that stops at logit output
    my_logitout_model = LSTMSun2018Logit(model_device=cur_device)
    my_logitout_model.load_state_dict(
        checkpoint["state_dict"], strict=False
    )

    # Instantiate feature perturber
    my_feature_perturber = SingleSampleFeaturePerturber(
        device=cur_device,
        feature_dims=(1, *tuple(full_dataset[0][1].shape)),
    )

    f48_m19_lstm_attacker = AdversarialAttacker(
        device=cur_device,
        feature_perturber=my_feature_perturber,
        logitout_model=my_logitout_model,
    )

    # F48_01 (no weighted sampling)
    # experiment_runner = AdvAttackExperimentRunner(
    #     device=cur_device,
    #     attacker=f48_m19_lstm_attacker,
    #     dataset=correctly_predicted_data,
    #     l1_beta_vals=[0.05, 0.1, 0.2],
    #     learning_rates=[0.05, 0.1, 0.2],
    #     kappa_vals=[0.0],
    #     samples_per_run=100,
    #     max_attempts_per_sample=100,
    #     max_successes_per_sample=1,
    #     output_dir=Path(__file__).parent.parent
    #     / "data"
    #     / "attack_results_f48_01",
    # )


    # F48_02 (with weighted_sampling)
    # experiment_runner = AdvAttackExperimentRunner(
    #     device=cur_device,
    #     attacker=f48_m19_lstm_attacker,
    #     dataset=correctly_predicted_data,
    #     l1_beta_vals=[0.05, 0.1, 0.2],
    #     learning_rates=[0.05, 0.1, 0.2],
    #     kappa_vals=[0.0],
    #     samples_per_run=100,
    #     max_attempts_per_sample=100,
    #     max_successes_per_sample=1,
    #     output_dir=Path(__file__).parent.parent
    #                / "data"
    #                / "attack_results_f48_02",
    # )
    #
    # experiment_runner.run_experiments()


    # F48_03 (increase max successes to 3)
    # experiment_runner = AdvAttackExperimentRunner(
    #     device=cur_device,
    #     attacker=f48_m19_lstm_attacker,
    #     dataset=correctly_predicted_data,
    #     l1_beta_vals=[0.05, 0.1, 0.2],
    #     learning_rates=[0.05, 0.1, 0.2],
    #     kappa_vals=[0.0],
    #     samples_per_run=100,
    #     max_attempts_per_sample=100,
    #     max_successes_per_sample=3,
    #     output_dir=Path(__file__).parent.parent
    #                / "data"
    #                / "attack_results_f48_03",
    # )
    #
    # experiment_runner.run_experiments()

    # F48_04 (increase max successes to 5, samples per run to 500)
    # experiment_runner = AdvAttackExperimentRunner(
    #     device=cur_device,
    #     attacker=f48_m19_lstm_attacker,
    #     dataset=correctly_predicted_data,
    #     l1_beta_vals=[0.05, 0.1, 0.2],
    #     learning_rates=[0.05, 0.1, 0.2],
    #     kappa_vals=[0.0],
    #     samples_per_run=500,
    #     max_attempts_per_sample=100,
    #     max_successes_per_sample=5,
    #     output_dir=Path(__file__).parent.parent
    #                / "data"
    #                / "attack_results_f48_04",
    # )
    #
    # experiment_runner.run_experiments()

    # F48_05 (increase max successes to 5, samples per run to 1000; more l1_beta and lr vals)
    # experiment_runner = AdvAttackExperimentRunner(
    #     device=cur_device,
    #     attacker=f48_m19_lstm_attacker,
    #     dataset=correctly_predicted_data,
    #     l1_beta_vals=[0, 0.05, 0.1, 0.2],
    #     learning_rates=[0.01, 0.05, 0.1, 0.2],
    #     kappa_vals=[0.0],
    #     samples_per_run=1000,
    #     max_attempts_per_sample=100,
    #     max_successes_per_sample=1,
    #     output_dir=Path(__file__).parent.parent
    #                / "data"
    #                / "attack_results_f48_05",
    # )
    #
    # experiment_runner.run_experiments()

    # F48_06 Now have orig label in output
    experiment_runner = AdvAttackExperimentRunner(
        device=cur_device,
        attacker=f48_m19_lstm_attacker,
        dataset=correctly_predicted_data,
        l1_beta_vals=[0, 0.05, 0.1, 0.2],
        learning_rates=[0.01, 0.05, 0.1, 0.2],
        kappa_vals=[0.0],
        samples_per_run=1000,
        max_attempts_per_sample=100,
        max_successes_per_sample=1,
        output_dir=Path(__file__).parent.parent
                   / "data"
                   / "attack_results_f48_05",
    )

    experiment_runner.run_experiments()
