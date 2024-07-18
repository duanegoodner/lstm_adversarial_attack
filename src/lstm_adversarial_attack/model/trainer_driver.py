import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from lstm_adversarial_attack import config

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.preprocess.encode_decode as edc
import lstm_adversarial_attack.resource_io as rio
import lstm_adversarial_attack.simple_logger as slg
import lstm_adversarial_attack.model.standard_model_trainer as smt
import lstm_adversarial_attack.model.tuner_helpers as tuh
import lstm_adversarial_attack.weighted_dataloader_builder as wdl
import lstm_adversarial_attack.x19_mort_general_dataset as xmd


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
        fold_idx: int = None,
        epoch_start_count: int = 0,
        output_dir: Path = None,
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
        self.fold_idx = fold_idx
        self.optimizer = getattr(
            torch.optim, hyperparameter_settings.optimizer_name
        )(self.model.parameters(), lr=hyperparameter_settings.learning_rate)
        if optimizer_state_dict is not None:
            self.optimizer.load_state_dict(state_dict=optimizer_state_dict)
        self.epoch_start_count = epoch_start_count
        # self.output_dir = self.initialize_output_dir(output_dir=output_dir)
        self.output_dirs = smt.TrainingOutputDirs(
            root_dir=output_dir, fold_index=fold_idx
        )
        self.initialize_output_content()
        self.summary_writer_group = summary_writer_group
        self.summary_writer_subgroup = summary_writer_subgroup
        self.summary_writer_add_graph = summary_writer_add_graph

    def initialize_output_content(self):
        hyperparameters_json_path = (
            self.output_dirs.root_dir / "hyperparameters.json"
        )
        edc.X19LSTMHyperParameterSettingsWriter().export(
            obj=self.hyperparameter_settings, path=hyperparameters_json_path
        )
        # with hyperparameters_json_path.open(mode="w") as out_file:
        #     json.dump(self.hyperparameter_settings.__dict__, out_file)

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
        config_reader = config.ConfigReader()
        max_observation_hours = config_reader.get_config_value("preprocess.max_observation_hours")

        tensorboard_model = tuh.X19LSTMBuilder(
            settings=self.hyperparameter_settings
        ).build_for_model_graph()
        dummy_input = torch.randn(
            self.batch_size, max_observation_hours, 19
        )

        summary_writer.add_graph(tensorboard_model, dummy_input)

    def run(
        self,
        num_epochs: int,
        eval_interval: int,
        save_checkpoints: bool,
    ):
        """
        Writes graph to tensorboard, instantiates Trainer & runs train/eval cycles.
        :param num_epochs: Number of epochs to train
        :param eval_interval: Number of epochs between evals
        :param save_checkpoints: Whether to save checkpoints
        """
        config_reader = config.ConfigReader()
        seed = config_reader.get_config_value("model.trainer.random_seed")

        torch.manual_seed(seed=seed)
        data_loaders = self.build_data_loaders()
        summary_writer = SummaryWriter(str(self.output_dirs.tensorboard_dir))

        if self.summary_writer_add_graph:
            self.add_model_graph_to_tensorboard(summary_writer=summary_writer)

        train_log_writer = slg.SimpleLogWriter(
            name=f"train_log_{uuid.uuid1().int>>64}",
            log_file=self.output_dirs.logs_dir / "train.log",
        )

        eval_log_writer = slg.SimpleLogWriter(
            name=f"eval_log_{uuid.uuid1().int>>64}",
            log_file=self.output_dirs.logs_dir / "eval.log",
        )

        trainer = smt.StandardModelTrainer(
            device=self.device,
            model=self.model,
            train_loader=data_loaders.train,
            test_loader=data_loaders.eval,
            loss_fn=self.loss_fn,
            optimizer=self.optimizer,
            checkpoint_dir=self.output_dirs.checkpoints_dir,
            summary_writer=summary_writer,
            epoch_start_count=self.epoch_start_count,
            summary_writer_group=self.summary_writer_group,
            summary_writer_subgroup=self.summary_writer_subgroup,
            train_log_writer=train_log_writer,
            eval_log_writer=eval_log_writer,
            # eval_log_metrics=cfs.TRAINER_EVAL_GENERAL_LOGGING_METRICS,
        )

        print(
            "Training model.\nCheckpoints will be saved"
            f" in:\n{self.output_dirs.checkpoints_dir}\n\nTensorboard logs"
            f" will be saved in:\n {self.output_dirs.tensorboard_dir}\n\n"
        )

        # This function returns a TrainEval pair, but currently no need to
        # capture it. All data gets saved to disk.
        trainer.run_train_eval_cycles(
            num_epochs=num_epochs,
            eval_interval=eval_interval,
            save_checkpoints=save_checkpoints,
        )


@dataclass
class HyperparametersModelPair:
    hyperparameters: tuh.X19LSTMHyperParameterSettings
    model: nn.Module


class BuildHyperparametersModelPair:
    @staticmethod
    def from_optuna_completed_trial_obj(
        completed_trial: optuna.Trial,
    ) -> HyperparametersModelPair:
        hyperparameters = tuh.X19LSTMHyperParameterSettings(
            **completed_trial.params
        )
        model = tuh.X19LSTMBuilder(settings=hyperparameters).build()
        return HyperparametersModelPair(
            hyperparameters=hyperparameters, model=model
        )

    @classmethod
    def from_optuna_completed_trial_pickle(
        cls, trial_path: Path
    ) -> HyperparametersModelPair:
        completed_trial = rio.ResourceImporter().import_pickle_to_object(
            path=trial_path
        )
        return cls.from_optuna_completed_trial_obj(
            completed_trial=completed_trial
        )

    @classmethod
    def from_optuna_study_obj(
        cls, study: optuna.Study
    ) -> HyperparametersModelPair:
        return cls.from_optuna_completed_trial_obj(
            completed_trial=study.best_trial
        )

    @classmethod
    def from_optuna_study_pickle(
        cls, pickle_path: Path
    ) -> HyperparametersModelPair:
        study = rio.ResourceImporter().import_pickle_to_object(
            path=pickle_path
        )
        return cls.from_optuna_study_obj(study=study)

    # @classmethod
    # def from_optuna_study(cls, study: optuna.Study):
