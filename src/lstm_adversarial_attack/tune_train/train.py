import sys
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from typing import Callable

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.resource_io as rio
import lstm_adversarial_attack.config_paths as lcp
import lstm_adversarial_attack.data_structures as ds
import lstm_adversarial_attack.weighted_dataloader_builder as wdl
import lstm_adversarial_attack.x19_mort_general_dataset as xmd
import lstm_adversarial_attack.tune_train.standard_model_trainer as smt
import lstm_adversarial_attack.tune_train.tuner_helpers as tuh


class TrainerDriver:
    def __init__(
        self,
        train_device: torch.device,
        eval_device: torch.device,
        model: nn.Module,
        batch_size: int,
        optimizer: torch.optim.Optimizer,
        dataset: Dataset = xmd.X19MGeneralDataset.from_feaure_finalizer_output(),
        collate_fn: Callable = xmd.x19m_collate_fn,
        loss_fn: nn.Module = nn.CrossEntropyLoss(),
        train_dataset_fraction: float = 0.8,
    ):
        self.train_device = train_device
        self.eval_device = eval_device
        self.model = model
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.batch_size = batch_size
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_dataset_fraction = train_dataset_fraction
        self.dataset_pair = self.split_dataset()
        self.output_dir = self.initialize_output_dir()
        self.tensorboard_output_dir = self.output_dir / "tensorboard"
        self.checkpoint_dir = self.output_dir / "checkpoints"

    @classmethod
    def from_hyperparameter_settings(
        cls,
        train_device: torch.device,
        eval_device: torch.device,
        settings: tuh.X19LSTMHyperParameterSettings,
        *args,
        **kwargs,
    ):
        model = tuh.X19LSTMBuilder(settings=settings).build()
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
        settings = tuh.X19LSTMHyperParameterSettings(**completed_trial.params)
        return cls.from_hyperparameter_settings(
            train_device=train_device,
            eval_device=eval_device,
            settings=settings,
        )

    def initialize_output_dir(self) -> Path:
        """
        Creates output dir and places saves pickle of model there
        """
        output_dir = rio.create_timestamped_dir(
            parent_path=lcp.TRAINING_OUTPUT_DIR
        )
        rio.ResourceExporter().export(
            resource=self.model, path=output_dir / "model.pickle"
        )
        return output_dir

    def split_dataset(self) -> tuh.TrainEvalDatasetPair:
        train_dataset_size = int(
            len(self.dataset) * self.train_dataset_fraction
        )
        test_dataset_size = len(self.dataset) - train_dataset_size
        train_dataset, test_dataset = random_split(
            dataset=self.dataset,
            lengths=(train_dataset_size, test_dataset_size),
        )
        return tuh.TrainEvalDatasetPair(
            train=train_dataset, validation=test_dataset
        )

    def build_data_loaders(self) -> tuh.TrainEvalDataLoaderPair:
        train_loader = wdl.WeightedDataLoaderBuilder(
            dataset=self.dataset_pair.train,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
        ).build()
        test_loader = DataLoader(
            dataset=self.dataset_pair.validation,
            batch_size=128,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

        return tuh.TrainEvalDataLoaderPair(train=train_loader, eval=test_loader)

    def __call__(
        self, num_cycles: int, epochs_per_cycle: int, save_checkpoints: bool
    ) -> ds.TrainEvalLogPair:
        data_loaders = self.build_data_loaders()
        trainer = smt.StandardModelTrainer(
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
        trial_path=lcp.BEST_TRIAL_RESULT_PATH,
    )

    cur_train_eval_pair = driver(
        num_cycles=200, epochs_per_cycle=1, save_checkpoints=True
    )
