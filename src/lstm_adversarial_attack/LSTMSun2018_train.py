import torch.cuda
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torch.utils.tensorboard import SummaryWriter
import project_config_old as pc
from lstm_model_stc_old import (
    BidirectionalX19LSTM,
)
from standard_model_trainer import StandardModelTrainer
from weighted_dataloader_builder import WeightedDataLoaderBuilder
from x19_mort_general_dataset import x19m_collate_fn, X19MGeneralDataset


def main():
    if torch.cuda.is_available():
        cur_device = torch.device("cuda:0")
    else:
        cur_device = torch.device("cpu")

    input_size = 19
    lstm_hidden_size = 128
    fc_hidden_size = 32
    out_features = 2
    batch_size = 128
    loss_fn = nn.CrossEntropyLoss()
    learning_rate = 1e-4
    beta_1 = 0.5
    beta_2 = 0.999

    model_concise = nn.Sequential(
        BidirectionalX19LSTM(
            input_size=input_size, lstm_hidden_size=lstm_hidden_size
        ),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(
            in_features=2 * lstm_hidden_size, out_features=fc_hidden_size
        ),
        nn.ReLU(),
        nn.Linear(in_features=fc_hidden_size, out_features=out_features),
        nn.Softmax(dim=1),
    )

    optimizer = torch.optim.Adam(
        params=model_concise.parameters(),
        lr=learning_rate,
        betas=(beta_1, beta_2),
    )

    dataset = X19MGeneralDataset.from_feaure_finalizer_output()
    train_dataset_size = int(len(dataset) * 0.8)
    test_dataset_size = len(dataset) - train_dataset_size
    train_dataset, test_dataset = random_split(
        dataset=dataset, lengths=(train_dataset_size, test_dataset_size)
    )
    train_dataloader = WeightedDataLoaderBuilder().build(
        dataset=train_dataset,
        batch_size=batch_size,
        collate_fn=x19m_collate_fn,
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=x19m_collate_fn,
    )

    summary_writer = SummaryWriter(log_dir=pc.DATA_DIR / "tensorboard_data")

    trainer = StandardModelTrainer(
        train_device=cur_device,
        eval_device=cur_device,
        model=model_concise,
        train_loader=train_dataloader,
        test_loader=test_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        checkpoint_dir=Path(__file__).parent.parent.parent
        / "data"
        / "training_results"
        / "LSTM_Sun2018_x19m_6_48_b",
        summary_writer=summary_writer,
        summary_writer_group="Trial_A",
        summary_writer_subgroup="Group_1",
    )

    trainer.run_train_eval_cycles(
        num_cycles=5, epochs_per_cycle=1, save_checkpoints=False
    )


if __name__ == "__main__":
    main()
