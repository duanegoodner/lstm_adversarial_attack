import sys

import optuna
import torch
import torch.nn as nn
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Callable

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.resource_io as rio
import lstm_adversarial_attack.config_paths as cfg_paths
import lstm_adversarial_attack.config_settings as lcs
import lstm_adversarial_attack.weighted_dataloader_builder as wdl
import lstm_adversarial_attack.x19_mort_general_dataset as xmd
import lstm_adversarial_attack.tune_train.standard_model_trainer as smt
import lstm_adversarial_attack.tune_train.tuner_helpers as tuh


class TrainerDriver:
    def __init__(
        self,
        train_device: torch.device,
        eval_device: torch.device,
        hyperparameter_settings: tuh.X19LSTMHyperParameterSettings,
        train_eval_dataset_pair: tuh.TrainEvalDatasetPair,
        model: nn.Module,
        model_state_dict: dict = None,
        optimizer_state_dict: dict = None,
        collate_fn: Callable = xmd.x19m_collate_fn,
        loss_fn: nn.Module = nn.CrossEntropyLoss(),
        epoch_start_count: int = 0,
        output_root_dir: Path = None,
        tensorboard_output_dir: Path = None,
        checkpoint_output_dir: Path = None,
        summary_writer_group: str = "",
        summary_writer_subgroup: str = "",
        summary_writer_add_graph: bool = False,
    ):
        self.train_device = train_device
        self.eval_device = eval_device
        self.model = model
        if model_state_dict is not None:
            self.model.load_state_dict(state_dict=model_state_dict)
        self.train_eval_dataset_pair = train_eval_dataset_pair
        self.collate_fn = collate_fn
        self.hyperparameter_settings = hyperparameter_settings
        self.batch_size = 2**hyperparameter_settings.log_batch_size
        self.loss_fn = loss_fn
        self.optimizer = getattr(
            torch.optim, hyperparameter_settings.optimizer_name
        )(self.model.parameters(), lr=hyperparameter_settings.learning_rate)
        if optimizer_state_dict is not None:
            self.optimizer.load_state_dict(state_dict=optimizer_state_dict)
        self.epoch_start_count = epoch_start_count
        (
            self.output_root_dir,
            self.tensorboard_output_dir,
            self.checkpoint_output_dir,
        ) = self.initialize_output_dir(
            output_root_dir=output_root_dir,
            tensorboard_output_dir=tensorboard_output_dir,
            checkpoint_output_dir=checkpoint_output_dir,
        )
        self.summary_writer_group = summary_writer_group
        self.summary_writer_subgroup = summary_writer_subgroup
        self.summary_writer_add_graph = summary_writer_add_graph

    def initialize_output_dir(
        self,
        output_root_dir: Path = None,
        tensorboard_output_dir: Path = None,
        checkpoint_output_dir: Path = None,
    ) -> tuple[Path, Path, Path]:
        """
        Creates output root, tensorboard dir, and checkpoint dir. Puts copies
        model and hyperparameters in output root.
        """

        if output_root_dir is None:
            output_root_dir = rio.create_timestamped_dir(
                parent_path=cfg_paths.TRAINING_OUTPUT_DIR
            )
        else:
            output_root_dir.mkdir(parents=True, exist_ok=True)
        rio.ResourceExporter().export(
            resource=self.model, path=output_root_dir / "model.pickle"
        )
        rio.ResourceExporter().export(
            resource=self.hyperparameter_settings,
            path=output_root_dir / "hyperparameters.pickle",
        )

        if tensorboard_output_dir is None:
            tensorboard_output_dir = output_root_dir / "tensorboard"
        tensorboard_output_dir.mkdir(parents=True, exist_ok=True)

        if checkpoint_output_dir is None:
            checkpoint_output_dir = output_root_dir / "checkpoints"
        checkpoint_output_dir.mkdir(parents=True, exist_ok=True)

        return output_root_dir, tensorboard_output_dir, checkpoint_output_dir

    @classmethod
    def from_optuna_completed_trial_obj(
        cls,
        train_device: torch.device,
        eval_device: torch.device,
        completed_trial: optuna.Trial,
        train_eval_dataset_pair: tuh.TrainEvalDatasetPair,
    ):
        settings = tuh.X19LSTMHyperParameterSettings(**completed_trial.params)
        model = tuh.X19LSTMBuilder(settings=settings).build()
        return cls(
            train_device=train_device,
            eval_device=eval_device,
            hyperparameter_settings=settings,
            model=model,
            train_eval_dataset_pair=train_eval_dataset_pair,
        )

    @classmethod
    def from_optuna_completed_trial_path(
        cls,
        train_device: torch.device,
        eval_device: torch.device,
        trial_path: Path,
        train_eval_dataset_pair: tuh.TrainEvalDatasetPair,
    ):
        completed_trial = rio.ResourceImporter().import_pickle_to_object(
            path=trial_path
        )
        return cls.from_optuna_completed_trial_obj(
            train_device=train_device,
            eval_device=eval_device,
            completed_trial=completed_trial,
            train_eval_dataset_pair=train_eval_dataset_pair,
        )

    @classmethod
    def from_optuna_study_path(
        cls,
        train_device: torch.device,
        eval_device: torch.device,
        study_path: Path,
        train_eval_dataset_pair: tuh.TrainEvalDatasetPair,
    ):
        study = rio.ResourceImporter().import_pickle_to_object(path=study_path)
        return cls.from_optuna_completed_trial_obj(
            train_device=train_device,
            eval_device=eval_device,
            completed_trial=study.best_trial,
            train_eval_dataset_pair=train_eval_dataset_pair,
        )

    @classmethod
    def from_previous_training(
        cls,
        train_device: torch.device,
        eval_device: torch.device,
        train_eval_dataset_pair: tuh.TrainEvalDatasetPair,
        checkpoint_file: Path,
        hyperparameters_file: Path,
        additional_output_dir: Path,
    ):
        hyperparameter_settings = (
            rio.ResourceImporter().import_pickle_to_object(
                path=hyperparameters_file
            )
        )
        model = tuh.X19LSTMBuilder(settings=hyperparameter_settings).build()
        model.to(train_device)
        checkpoint = torch.load(checkpoint_file, map_location=train_device)
        return cls(
            train_device=train_device,
            eval_device=eval_device,
            train_eval_dataset_pair=train_eval_dataset_pair,
            model=model,
            hyperparameter_settings=hyperparameter_settings,
            model_state_dict=checkpoint["state_dict"],
            optimizer_state_dict=checkpoint["optimizer_state_dict"],
            epoch_start_count=checkpoint["epoch_num"],
            output_root_dir=additional_output_dir,
        )

    @classmethod
    def from_standard_previous_training(
        cls,
        train_device: torch.device,
        eval_device: torch.device,
        train_eval_dataset_pair: tuh.TrainEvalDatasetPair,
        training_output_dir: Path,
    ):
        checkpoint_file = sorted(
            (training_output_dir / "checkpoints").glob("*.tar")
        )[-1]

        return cls.from_previous_training(
            train_device=train_device,
            eval_device=eval_device,
            train_eval_dataset_pair=train_eval_dataset_pair,
            checkpoint_file=checkpoint_file,
            hyperparameters_file=training_output_dir
            / "hyperparameters.pickle",
            additional_output_dir=training_output_dir,
        )

    def build_data_loaders(self) -> tuh.TrainEvalDataLoaderPair:
        train_loader = wdl.WeightedDataLoaderBuilder(
            dataset=self.train_eval_dataset_pair.train,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
        ).build()
        test_loader = DataLoader(
            dataset=self.train_eval_dataset_pair.validation,
            batch_size=128,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

        return tuh.TrainEvalDataLoaderPair(
            train=train_loader, eval=test_loader
        )

    def add_model_graph_to_tensorboard(self, summary_writer: SummaryWriter):
        tensorboard_model = tuh.X19LSTMBuilder(
            settings=self.hyperparameter_settings
        ).build_for_tensorboard()
        dummy_input = torch.randn(
            self.batch_size, lcs.MAX_OBSERVATION_HOURS, 19
        )
        # dummy_dataloader = wdl.WeightedDataLoaderBuilder(
        #     self.train_eval_dataset_pair.train,
        #     batch_size=self.batch_size,
        #     collate_fn=self.collate_fn,
        # ).build()
        # dummy_data_iterator = iter(dummy_dataloader)
        # dummy_batch = next(dummy_data_iterator)
        # padded_inputs = pad_sequence(dummy_batch[0])

        summary_writer.add_graph(tensorboard_model, dummy_input)

    def run(
        self,
        num_epochs: int,
        eval_interval: int,
        evals_per_checkpoint,
        save_checkpoints: bool,
    ):
        torch.manual_seed(lcs.TRAINER_RANDOM_SEED)
        data_loaders = self.build_data_loaders()
        summary_writer = SummaryWriter(str(self.tensorboard_output_dir))

        if self.summary_writer_add_graph:
            self.add_model_graph_to_tensorboard(summary_writer=summary_writer)
            # dummy_dataloader = wdl.WeightedDataLoaderBuilder(
            #     dataset=self.train_eval_dataset_pair.train,
            #     batch_size=self.batch_size,
            #     collate_fn=self.collate_fn
            # ).build()
            # dummy_iterator = iter(dummy_dataloader)
            # dummy_features, dummy_labels = next(dummy_iterator)
            # summary_writer.add_graph(self.model, dummy_features)

        trainer = smt.StandardModelTrainer(
            train_device=self.train_device,
            eval_device=self.eval_device,
            model=self.model,
            train_loader=data_loaders.train,
            test_loader=data_loaders.eval,
            loss_fn=self.loss_fn,
            optimizer=self.optimizer,
            checkpoint_dir=self.checkpoint_output_dir,
            summary_writer=summary_writer,
            epoch_start_count=self.epoch_start_count,
            summary_writer_group=self.summary_writer_group,
            summary_writer_subgroup=self.summary_writer_subgroup,
        )

        print(
            "Training model.\nCheckpoints will be saved"
            f" in:\n{self.checkpoint_output_dir}\n\nTensorboard logs will be"
            f" saved in:\n {self.tensorboard_output_dir}\n\n"
        )

        # This function returns a TrainEval pair, but currently no need to
        # capture it. All data gets saved to disk.
        trainer.run_train_eval_cycles(
            num_epochs=num_epochs,
            eval_interval=eval_interval,
            evals_per_checkpoint=evals_per_checkpoint,
            save_checkpoints=save_checkpoints,
        )
