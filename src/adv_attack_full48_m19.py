import argparse
import time
import torch
from pathlib import Path
from torch.utils.data import Subset
from adv_attack import AdversarialAttacker, AdversarialAttackTrainer
from lstm_model_stc import LSTMSun2018
from lstm_sun_2018_logit_out import LSTMSun2018Logit
from single_sample_feature_perturber import SingleSampleFeaturePerturber
from standard_model_inferrer import StandardModelInferrer
from dataset_full48_m19 import Full48M19DatasetWithIndex


if __name__ == "__main__":
    cur_parser = argparse.ArgumentParser()
    cur_parser.add_argument(
        "-f",
        "--params_filename",
        type=str,
        nargs="?",
        action="store",
        help=(
            "Filename (not full path) of .tar file containing model "
            "parameters for LSTMSun2018 model. File needs to be in directory"
            "ehr_adversarial_attack/data/cross_validate_sun2018_full48m19_01."
        ),
    )
    args_namespace = cur_parser.parse_args()
    params_dir = Path(__file__).parent.parent / "data" / "cross_validate_sun2018_full48m19_01"
    params_file_path = params_dir / args_namespace.params_filename

    if torch.cuda.is_available():
        cur_device = torch.device("cuda:0")
    else:
        cur_device = torch.device("cpu")

    # Instantiate a full model and load pretrained parameters
    pretrained_full_model = LSTMSun2018(model_device=cur_device)
    checkpoint = torch.load(params_file_path)
    pretrained_full_model.load_state_dict(
        checkpoint["state_dict"], strict=False
    )

    # Run data through pretrained model & get all correctly predicted samples
    full_dataset = Full48M19DatasetWithIndex.from_feature_finalizer_output()
    inferrer = StandardModelInferrer(
        model=pretrained_full_model, dataset=full_dataset
    )
    correctly_predicted_data = inferrer.get_correctly_predicted_samples()
    small_correctly_predicted_data = Subset(
        dataset=correctly_predicted_data, indices=list(range(10))
    )

    # Instantiate version of model that stops at logit output
    my_logitout_model = LSTMSun2018Logit(model_device=cur_device)
    my_logitout_model.load_state_dict(checkpoint["state_dict"], strict=False)

    # Instantiate feature perturber
    my_feature_perturber = SingleSampleFeaturePerturber(
        device=cur_device, feature_dims=(1, *tuple(full_dataset[0][1].shape))
    )

    f48_m19_lstm_attacker = AdversarialAttacker(
        device=cur_device,
        feature_perturber=my_feature_perturber,
        logitout_model=my_logitout_model,
    )

    trainer = AdversarialAttackTrainer(
        device=cur_device,
        attacker=f48_m19_lstm_attacker,
        dataset=correctly_predicted_data,
        learning_rate=0.1,
        kappa=0,
        l1_beta=0.15,
        num_samples=1000,
        max_attempts_per_sample=100,
        max_successes_per_sample=1,
        output_dir=Path(__file__).parent.parent
        / "data"
        / "attack_results_f48_00",
    )
    trainer.train_attacker()


