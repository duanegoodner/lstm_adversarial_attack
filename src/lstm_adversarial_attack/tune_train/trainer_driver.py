import sys
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from typing import Callable

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.resource_io as rio
from lstm_adversarial_attack.config_paths import (
    BEST_TRIAL_RESULT_PATH,
    TRAINING_OUTPUT_DIR,
)
from lstm_adversarial_attack.data_structures import TrainEvalLogPair
from lstm_adversarial_attack.weighted_dataloader_builder import (
    WeightedDataLoaderBuilder,
)
from lstm_adversarial_attack.x19_mort_general_dataset import (
    x19m_collate_fn,
    X19MGeneralDataset,
)
from standard_model_trainer import StandardModelTrainer
from tuner_helpers import (
    TrainEvalDataLoaderPair,
    TrainEvalDatasetPair,
    X19LSTMBuilder,
    X19LSTMHyperParameterSettings,
)


class TrainerDriver:
    def __init__(
        self,
        train_device: torch.device,
        eval_device: torch.device,
        model: nn.Module,
        batch_size: int,
        optimizer: torch.optim.Optimizer,
        dataset: Dataset = X19MGeneralDataset.from_feaure_finalizer_output(),
        collate_fn: Callable = x19m_collate_fn,
        loss_fn: nn.Module = nn.CrossEntropyLoss(),
        output_dir: Path = None,
        # train_loader_builder=WeightedDataLoaderBuilder(),
        train_dataset_fraction: float = 0.8,
        random_seed: int = None,
    ):
        self.train_device = train_device
        self.eval_device = eval_device
        self.model = model
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.batch_size = batch_size
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        # self.train_loader_builder = train_loader_builder
        self.train_dataset_fraction = train_dataset_fraction
        self.random_seed = random_seed
        if random_seed is not None:
            torch.manual_seed(random_seed)
        self.dataset_pair = self.split_dataset()
        self.output_dir = self.initialize_output_dir(output_dir=output_dir)
        self.tensorboard_output_dir = self.output_dir / "tensorboard"
        self.checkpoint_dir = self.output_dir / "checkpoints"

    @classmethod
    def from_hyperparameter_settings(
        cls,
        train_device: torch.device,
        eval_device: torch.device,
        settings: X19LSTMHyperParameterSettings,
        *args,
        **kwargs,
    ):
        model = X19LSTMBuilder(settings=settings).build()
        return cls(
            train_device=train_device,
            eval_device=eval_device,
            model=model,
            batch_size=2**settings.log_batch_size,
            optimizer=getattr(torch.optim, settings.optimizer_name)(
                model.parameters(), lr=settings.learning_rate
            ),
        )

    @classmethod
    def from_optuna_completed_trial(
        cls,
        train_device: torch.device,
        eval_device: torch.device,
        trial_path: Path,
    ):
        completed_trial = rio.ResourceImporter().import_pickle_to_object(
            path=trial_path
        )
        settings = X19LSTMHyperParameterSettings(**completed_trial.params)
        return cls.from_hyperparameter_settings(
            train_device=train_device,
            eval_device=eval_device,
            settings=settings,
        )

    def initialize_output_dir(self, output_dir: Path = None) -> Path:
        """
        Creates output dir and places saves pickle of model there
        """
        if output_dir is None:
            dirname = f"{datetime.now()}".replace(" ", "_")
            output_dir = TRAINING_OUTPUT_DIR / dirname
        assert not output_dir.exists()
        output_dir.mkdir()
        rio.ResourceExporter().export(
            resource=self.model, path=output_dir / "model.pickle"
        )
        return output_dir

    def split_dataset(self) -> TrainEvalDatasetPair:
        train_dataset_size = int(
            len(self.dataset) * self.train_dataset_fraction
        )
        test_dataset_size = len(self.dataset) - train_dataset_size
        train_dataset, test_dataset = random_split(
            dataset=self.dataset,
            lengths=(train_dataset_size, test_dataset_size),
        )
        return TrainEvalDatasetPair(
            train=train_dataset, validation=test_dataset
        )

    def build_data_loaders(self) -> TrainEvalDataLoaderPair:
        # train_loader = self.train_loader_builder.build(
        #     dataset=self.dataset_pair.train,
        #     batch_size=self.batch_size,
        #     collate_fn=self.collate_fn,
        # )
        train_loader = WeightedDataLoaderBuilder(
            dataset=self.dataset_pair.train,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn
        ).build()
        test_loader = DataLoader(
            dataset=self.dataset_pair.validation,
            batch_size=128,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

        return TrainEvalDataLoaderPair(train=train_loader, eval=test_loader)

    def __call__(
        self, num_cycles: int, epochs_per_cycle: int, save_checkpoints: bool
    ) -> TrainEvalLogPair:
        data_loaders = self.build_data_loaders()
        trainer = StandardModelTrainer(
            train_device=self.train_device,
            eval_device=self.eval_device,
            model=self.model,
            train_loader=data_loaders.train,
            test_loader=data_loaders.eval,
            loss_fn=self.loss_fn,
            optimizer=self.optimizer,
            checkpoint_dir=self.checkpoint_dir,
            summary_writer=SummaryWriter(str(self.tensorboard_output_dir)),
        )

        train_eval_log_pair = trainer.run_train_eval_cycles(
            num_cycles=num_cycles,
            epochs_per_cycle=epochs_per_cycle,
            save_checkpoints=save_checkpoints,
        )

        return train_eval_log_pair


if __name__ == "__main__":
    if torch.cuda.is_available():
        cur_device = torch.device("cuda:0")
    else:
        cur_device = torch.device("cpu")

    driver = TrainerDriver.from_optuna_completed_trial(
        train_device=cur_device,
        eval_device=cur_device,
        trial_path=BEST_TRIAL_RESULT_PATH,
    )

    cur_train_eval_pair = driver(
        num_cycles=200, epochs_per_cycle=1, save_checkpoints=True
    )
