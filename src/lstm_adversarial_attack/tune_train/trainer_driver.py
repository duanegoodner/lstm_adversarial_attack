import sys

import optuna
import torch
import torch.nn as nn
from pathlib import Path
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
    """
    Isolation layer to avoid need to modify StandardModelTrainer.

    Instantiates and runs a StandardModelTrainer.

    Provides class methods to instantiate from Optuna output or checkpoints.

    """
    def __init__(
        self,
        device: torch.device,
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
        self.device = device
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
        of model and hyperparameters in output root.
        :param output_root_dir: root dir for files saved from training/eval
        :param tensorboard_output_dir: folder for tensorboard output
        :param checkpoint_output_dir: folder to save checkpoints in
        :return: output_root_dir, tensorboard_output_dir, checkpoint_output_dir
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
        device: torch.device,
        completed_trial: optuna.Trial,
        train_eval_dataset_pair: tuh.TrainEvalDatasetPair,
    ):
        """
        Creates a TrainerDriver using info from optuna.Trial object
        :param device: device to run on
        :param completed_trial: a completed optuna.Trial object
        :param train_eval_dataset_pair: train & eval datasets
        """
        settings = tuh.X19LSTMHyperParameterSettings(**completed_trial.params)
        model = tuh.X19LSTMBuilder(settings=settings).build()
        return cls(
            device=device,
            hyperparameter_settings=settings,
            model=model,
            train_eval_dataset_pair=train_eval_dataset_pair,
        )

    @classmethod
    def from_optuna_completed_trial_path(
        cls,
        device: torch.device,
        trial_path: Path,
        train_eval_dataset_pair: tuh.TrainEvalDatasetPair,
    ):
        """
        Creates a TrainerDriver using (pickle) file path of optuna.Trial
        :param device: device to run on
        :param trial_path: path to pickle with completed optuna.Trial
        :param train_eval_dataset_pair: train & eval datasets
        """
        completed_trial = rio.ResourceImporter().import_pickle_to_object(
            path=trial_path
        )
        return cls.from_optuna_completed_trial_obj(
            device=device,
            completed_trial=completed_trial,
            train_eval_dataset_pair=train_eval_dataset_pair,
        )

    @classmethod
    def from_optuna_study_path(
        cls,
        device: torch.device,
        study_path: Path,
        train_eval_dataset_pair: tuh.TrainEvalDatasetPair,
    ):
        """
        Creates TrainerDriver using filepath of optuna.Study pickle
        :param device: device to run on
        :param study_path: file path to optuna.Study pickle file
        :param train_eval_dataset_pair: train and test/eval datasets
        """
        study = rio.ResourceImporter().import_pickle_to_object(path=study_path)
        return cls.from_optuna_completed_trial_obj(
            device=device,
            completed_trial=study.best_trial,
            train_eval_dataset_pair=train_eval_dataset_pair,
        )

    @classmethod
    def from_previous_training(
        cls,
        device: torch.device,
        train_eval_dataset_pair: tuh.TrainEvalDatasetPair,
        checkpoint_file: Path,
        hyperparameters_file: Path,
        additional_output_dir: Path,
    ):
        """
        Creates TrainerDriver w/ info from files output by previous train run
        :param device: device to run on
        :param train_eval_dataset_pair: train and eval/test datasets
        :param checkpoint_file: path to previous training checkpoint
        :param hyperparameters_file: path X19LSTMHyperParameterSettings pickle
        :param additional_output_dir: path where new output will be saved
        """
        hyperparameter_settings = (
            rio.ResourceImporter().import_pickle_to_object(
                path=hyperparameters_file
            )
        )
        model = tuh.X19LSTMBuilder(settings=hyperparameter_settings).build()
        model.to(device)
        checkpoint = torch.load(checkpoint_file, map_location=device)
        return cls(
            device=device,
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
        device: torch.device,
        train_eval_dataset_pair: tuh.TrainEvalDatasetPair,
        training_output_dir: Path,
    ):
        """
        Creates TrainerDriver output root of previous training.

        Assumes standard training output file structure.
        :param device: device to run on
        :param train_eval_dataset_pair: train and test/eval datasets
        :param training_output_dir: root output dir of previous training
        """
        checkpoint_file = sorted(
            (training_output_dir / "checkpoints").glob("*.tar")
        )[-1]

        return cls.from_previous_training(
            device=device,
            train_eval_dataset_pair=train_eval_dataset_pair,
            checkpoint_file=checkpoint_file,
            hyperparameters_file=training_output_dir
            / "hyperparameters.pickle",
            additional_output_dir=training_output_dir,
        )

    def build_data_loaders(self) -> tuh.TrainEvalDataLoaderPair:
        """
        Builds train and test dataloaders.

        Trainloader uses WeightedRandomSampler (b/c project data is imbalanced)
        :return: dataclass w/ refs to both dataloaders.
        """
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
        """
        Writes a graph object representing self.model to Tensorboard.

        Actual graph written is for dummy_model with only tensor input.

        (writer cannot handle model with VariableLengthInput as input)

        :param summary_writer: SummaryWriter object
        """
        tensorboard_model = tuh.X19LSTMBuilder(
            settings=self.hyperparameter_settings
        ).build_for_model_graph()
        dummy_input = torch.randn(
            self.batch_size, lcs.MAX_OBSERVATION_HOURS, 19
        )

        summary_writer.add_graph(tensorboard_model, dummy_input)

    def run(
        self,
        num_epochs: int,
        eval_interval: int,
        evals_per_checkpoint,
        save_checkpoints: bool,
    ):
        """
        Writes graph to tensorboard, instantiates Trainer & runs train/eval cycles.
        :param num_epochs: Number of epochs to train
        :param eval_interval: Number of epochs between evals
        :param evals_per_checkpoint: Number of evals per checkpoint
        :param save_checkpoints: Whether to save checkpoints
        """
        torch.manual_seed(lcs.TRAINER_RANDOM_SEED)
        data_loaders = self.build_data_loaders()
        summary_writer = SummaryWriter(str(self.tensorboard_output_dir))

        if self.summary_writer_add_graph:
            self.add_model_graph_to_tensorboard(summary_writer=summary_writer)

        trainer = smt.StandardModelTrainer(
            device=self.device,
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
