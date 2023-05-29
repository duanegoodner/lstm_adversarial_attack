import torch
from pathlib import Path
from torch.utils.data import Subset
from adv_attack import AdversarialAttacker, AdversarialAttackTrainer
from lstm_model_stc_old import LSTMSun2018
from lstm_sun_2018_logit_out import LSTMSun2018Logit
from single_sample_feature_perturber import SingleSampleFeaturePerturber
from standard_model_inferrer import StandardModelInferrer
from x19_mort_dataset import X19MortalityDatasetWithIndex


if __name__ == "__main__":
    if torch.cuda.is_available():
        cur_device = torch.device("cuda:0")
    else:
        cur_device = torch.device("cpu")

    # Instantiate a full model and load pretrained parameters
    pretrained_full_model = LSTMSun2018(model_device=cur_device)
    checkpoint_path = Path(
        "/home/duane/dproj/UIUC-DLH/project/ehr_adversarial_attack/data"
        "/training_results/2023-04-30_18:49:09.556432.tar"
    )
    checkpoint = torch.load(checkpoint_path)
    pretrained_full_model.load_state_dict(
        checkpoint["model_state_dict"], strict=False
    )

    # Run data through pretrained model & get all correctly predicted samples
    full_dataset = X19MortalityDatasetWithIndex()
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
        checkpoint["model_state_dict"], strict=False
    )

    # Instantiate feature perturber
    my_feature_perturber = SingleSampleFeaturePerturber(
        device=cur_device, feature_dims=(1, *tuple(full_dataset[0][1].shape))
    )

    x19_lstm_attacker = AdversarialAttacker(
        device=cur_device,
        feature_perturber=my_feature_perturber,
        logitout_model=my_logitout_model,
    )

    trainer = AdversarialAttackTrainer(
        device=cur_device,
        attacker=x19_lstm_attacker,
        dataset=small_correctly_predicted_data,
        learning_rate=0.1,
        kappa=0,
        l1_beta=0.15,
        epochs_per_batch=100,
    )

    trainer.train_attacker()
