import torch.cuda
import torch.nn as nn
from datetime import datetime
from pathlib import Path
from cross_validator import CrossValidator
from lstm_model_stc import LSTMSun2018
from standard_model_trainer import StandardModelTrainer
from x19_mort_dataset import X19MortalityDataset
from dataset_full48_m19 import Full48M19Dataset


def main():
    if torch.cuda.is_available():
        cur_device = torch.device("cuda:0")
    else:
        cur_device = torch.device("cpu")

    # dataset = X19MortalityDataset()
    dataset = Full48M19Dataset.from_feature_finalizer_output()

    model = LSTMSun2018(model_device=cur_device)

    cross_validator = CrossValidator(
        model=model,
        dataset=dataset,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(
            params=model.parameters(), lr=1e-4, betas=(0.5, 0.999)
        ),
        num_folds=5,
        batch_size=128,
        epochs_per_fold=25,
        max_global_epochs=8,
        save_checkpoints=True,
        checkpoints_dir=Path(__file__).parent.parent
        / "data"
        / "cross_validate_sun2018_full48m19_01",
    )

    cross_validator.run()


if __name__ == "__main__":
    start = datetime.now()
    main()
    end = datetime.now()
