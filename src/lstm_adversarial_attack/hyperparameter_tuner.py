import optuna
import torch
import torch.nn as nn
from dataclasses import dataclass
from datetime import datetime
from early_stopping import PerformanceSelector
from optuna.pruners import BasePruner, MedianPruner
from optuna.samplers import BaseSampler, TPESampler
from optuna.trial import TrialState
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, Subset
from typing import Callable
import project_config_old as pc
import resource_io as rio
from data_structures import (
    CVTrialLogs,
    EvalEpochResult,
    EvalLogEntry,
    EvalLog,
    OptimizeDirection,
    TrainEvalLogPair,
)
from lstm_model_stc import BidirectionalX19LSTM
from standard_model_trainer import StandardModelTrainer
from torch.utils.tensorboard import SummaryWriter
from weighted_dataloader_builder import (
    WeightedDataLoaderBuilder,
)


@dataclass
class TrainEvalDatasetPair:
    train: Dataset
    validation: Dataset


@dataclass
class X19MLSTMTuningRanges:
    log_lstm_hidden_size: tuple[int, int]
    lstm_act_options: tuple[str, ...]
    dropout: tuple[float, float]
    log_fc_hidden_size: tuple[int, int]
    fc_act_options: tuple[str, ...]
    optimizer_options: tuple[str, ...]
    learning_rate: tuple[float, float]
    log_batch_size: tuple[int, int]


@dataclass
class X19LSTMHyperParameterSettings:
    log_lstm_hidden_size: int
    lstm_act_name: str
    dropout: float
    log_fc_hidden_size: int
    fc_act_name: str
    optimizer_name: str
    learning_rate: float
    log_batch_size: int

    @classmethod
    def from_optuna(cls, trial, tuning_ranges: X19MLSTMTuningRanges):
        return cls(
            log_lstm_hidden_size=trial.suggest_int(
                "log_lstm_hidden_size",
                *tuning_ranges.log_lstm_hidden_size,
            ),
            lstm_act_name=trial.suggest_categorical(
                "lstm_act", list(tuning_ranges.lstm_act_options)
            ),
            dropout=trial.suggest_float("dropout", *tuning_ranges.dropout),
            log_fc_hidden_size=trial.suggest_int(
                "log_fc_hidden_size", *tuning_ranges.log_fc_hidden_size
            ),
            fc_act_name=trial.suggest_categorical(
                "fc_act", list(tuning_ranges.fc_act_options)
            ),
            optimizer_name=trial.suggest_categorical(
                "optimizer", list(tuning_ranges.optimizer_options)
            ),
            learning_rate=trial.suggest_float(
                "lr", *tuning_ranges.learning_rate, log=True
            ),
            log_batch_size=trial.suggest_int(
                "log_batch_size", *tuning_ranges.log_batch_size
            ),
        )


@dataclass
class ObjectiveFunctionTools:
    settings: X19LSTMHyperParameterSettings
    summary_writer: SummaryWriter
    cv_means_log: EvalLog
    trainers: list[StandardModelTrainer]


class HyperParameterTuner:
    def __init__(
        self,
        device: torch.device,
        dataset: Dataset,
        collate_fn: Callable,
        num_folds: int,
        num_cv_epochs: int,
        epochs_per_fold: int,
        tuning_ranges: dataclass,
        fold_class: Callable = StratifiedKFold,
        train_loader_builder=WeightedDataLoaderBuilder(),
        kfold_random_seed: int = 1234,
        loss_fn: nn.Module = nn.CrossEntropyLoss(),
        cv_mean_metrics_of_interest: tuple[str] = ("AUC", "validation_loss"),
        performance_metric: str = "validation_loss",
        optimization_direction: OptimizeDirection = OptimizeDirection.MIN,
        performance_metric_selector: Callable = min,
        pruner: BasePruner = MedianPruner(
            n_startup_trials=2, n_warmup_steps=5
        ),
        hyperparameter_sampler: BaseSampler = TPESampler(),
        output_dir: Path = None,
        save_trial_info: bool = True,
        trial_prefix: str = "trial_",
    ):
        self.device = device
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.num_folds = num_folds
        self.num_cv_epochs = num_cv_epochs
        self.epochs_per_fold = epochs_per_fold
        self.fold_class = fold_class
        # use seed to keep same fold indices for all trials
        self.kfold_random_seed = kfold_random_seed
        self.cv_datasets = self.create_datasets()
        self.train_loader_builder = train_loader_builder
        self.loss_fn = loss_fn
        self.performance_metric = performance_metric
        self.performance_metric_selector = performance_metric_selector
        self.optimization_direction = optimization_direction
        self.optimization_direction_label = (
            "minimize"
            if optimization_direction == OptimizeDirection.MIN
            else "maximize"
        )
        self.pruner = pruner
        self.cv_mean_metrics_of_interest = cv_mean_metrics_of_interest
        self.tuning_ranges = tuning_ranges
        self.hyperparameter_sampler = hyperparameter_sampler
        self.output_dir = self.initialize_output_dir(output_dir=output_dir)
        self.tensorboard_output_dir = self.output_dir / "tensorboard"
        self.trainer_checkpoint_dir = self.output_dir / "checkpoints_trainer"
        self.tuner_checkpoint_dir = self.output_dir / "checkpoints_tuner"
        self.exporter = rio.ResourceExporter()
        self.save_trial_info = save_trial_info
        self.trial_prefix = trial_prefix

    @staticmethod
    def initialize_output_dir(output_dir: Path = None) -> Path:
        if output_dir is None:
            dirname = f"{datetime.now()}".replace(" ", "_")
            output_dir = pc.HYPERPARAMETER_TUNING_OUT_DIR / dirname
        assert not output_dir.exists()
        output_dir.mkdir()
        return output_dir

    def create_datasets(self) -> list[TrainEvalDatasetPair]:
        fold_generator_builder = self.fold_class(
            n_splits=self.num_folds,
            shuffle=True,
            random_state=self.kfold_random_seed,
        )
        fold_generator = fold_generator_builder.split(
            self.dataset[:][0], self.dataset[:][1]
        )

        all_train_eval_pairs = []

        for fold_idx, (train_indices, validation_indices) in enumerate(
            fold_generator
        ):
            train_dataset = Subset(dataset=self.dataset, indices=train_indices)
            validation_dataset = Subset(
                dataset=self.dataset, indices=validation_indices
            )
            all_train_eval_pairs.append(
                TrainEvalDatasetPair(
                    train=train_dataset, validation=validation_dataset
                )
            )

        return all_train_eval_pairs

    @staticmethod
    def define_model(settings: X19LSTMHyperParameterSettings):
        return nn.Sequential(
            BidirectionalX19LSTM(
                input_size=19,
                lstm_hidden_size=2**settings.log_lstm_hidden_size,
            ),
            getattr(nn, settings.lstm_act_name)(),
            nn.Dropout(p=settings.dropout),
            nn.Linear(
                in_features=2 * (2**settings.log_lstm_hidden_size),
                out_features=2**settings.log_fc_hidden_size,
            ),
            getattr(nn, settings.fc_act_name)(),
            nn.Linear(
                in_features=2**settings.log_fc_hidden_size, out_features=2
            ),
            nn.Softmax(dim=1),
        )

    @staticmethod
    def initialize_model(model: nn.Module):
        for name, param in model.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)

    def create_trainers(
        self,
        settings: X19LSTMHyperParameterSettings,
        summary_writer: SummaryWriter,
        trial_number: int,
    ):
        trainers = []
        for fold_idx, dataset_pair in enumerate(self.cv_datasets):
            model = self.define_model(settings=settings)
            train_loader = self.train_loader_builder.build(
                dataset=dataset_pair.train,
                batch_size=2**settings.log_batch_size,
                collate_fn=self.collate_fn,
            )
            validation_loader = DataLoader(
                dataset=dataset_pair.validation,
                batch_size=128,
                shuffle=False,
                collate_fn=self.collate_fn,
            )

            trainer = StandardModelTrainer(
                train_device=self.device,
                eval_device=self.device,
                model=model,
                loss_fn=self.loss_fn,
                optimizer=getattr(torch.optim, settings.optimizer_name)(
                    model.parameters(), lr=settings.learning_rate
                ),
                train_loader=train_loader,
                test_loader=validation_loader,
                summary_writer=summary_writer,
                summary_writer_group=f"{self.trial_prefix}{trial_number}",
                summary_writer_subgroup=f"fold_{fold_idx}",
                checkpoint_dir=self.trainer_checkpoint_dir,
            )

            trainers.append(trainer)

        return trainers

    def export_trial_info(
        self,
        trial,
        trainers: list[StandardModelTrainer],
        cv_means_log: EvalLog,
    ):
        if not self.tuner_checkpoint_dir.exists():
            self.tuner_checkpoint_dir.mkdir()

        trial_summary = CVTrialLogs(
            # trial=trial,
            trainer_logs=[
                TrainEvalLogPair(
                    train=trainer.train_log, eval=trainer.eval_log
                )
                for trainer in trainers
            ],
            cv_means_log=cv_means_log,
        )

        self.exporter.export(
            resource=trial_summary,
            path=self.tuner_checkpoint_dir
            / f"{self.trial_prefix}{trial.number}_logs.pickle",
        )

    def report_cv_means(
        self,
        log_entry: EvalLogEntry,
        summary_writer: SummaryWriter,
        trial: optuna.Trial,
    ):
        for metric in self.cv_mean_metrics_of_interest:
            summary_writer.add_scalar(
                f"{self.trial_prefix}{trial.number}/{metric}_mean",
                getattr(log_entry.result, metric),
                log_entry.epoch,
            )

    def build_objective_function_tools(self, trial: optuna.Trial):
        settings = X19LSTMHyperParameterSettings.from_optuna(
            trial=trial, tuning_ranges=self.tuning_ranges
        )
        summary_writer_path = (
            self.tensorboard_output_dir / f"{self.trial_prefix}{trial.number}"
        )

        return ObjectiveFunctionTools(
            settings=settings,
            summary_writer=SummaryWriter(str(summary_writer_path)),
            cv_means_log=EvalLog(),
            trainers=self.create_trainers(
                settings=settings,
                summary_writer=SummaryWriter(str(summary_writer_path)),
                trial_number=trial.number,
            ),
        )

    def objective_fn(self, trial) -> float | None:
        objective_tools = self.build_objective_function_tools(trial=trial)

        # TODO consider setting seed here so WeightedRandomSampler makes same
        #  selections across all trials
        for cv_epoch in range(self.num_cv_epochs):
            eval_epoch_results = []
            for fold_idx, trainer in enumerate(objective_tools.trainers):
                trainer.train_model(num_epochs=self.epochs_per_fold)
                eval_log = trainer.evaluate_model()
                eval_epoch_results.append(eval_log.latest_entry)
                trainer.model.to("cpu")
            mean_validation_vals = EvalEpochResult.mean(
                [item.result for item in eval_epoch_results]
            )

            cv_means_log_entry = EvalLogEntry(
                epoch=(cv_epoch + 1) * self.epochs_per_fold,
                result=mean_validation_vals,
            )

            objective_tools.cv_means_log.update(entry=cv_means_log_entry)

            # trial.report is NOT part of this function
            self.report_cv_means(
                log_entry=cv_means_log_entry,
                summary_writer=objective_tools.summary_writer,
                trial=trial,
            )

            trial.report(
                getattr(mean_validation_vals, self.performance_metric),
                cv_epoch,
            )

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if self.save_trial_info:
            self.export_trial_info(
                trial=trial,
                trainers=objective_tools.trainers,
                cv_means_log=objective_tools.cv_means_log,
            )

        return PerformanceSelector(
            optimize_direction=self.optimization_direction
        ).choose_best_val(
            values=[
                getattr(item.result, self.performance_metric)
                for item in objective_tools.cv_means_log.data
            ]
        )

    def export_study(self, study: optuna.Study):
        if not self.tuner_checkpoint_dir.exists():
            self.tuner_checkpoint_dir.mkdir()
        study_filename = f"optuna_study.pickle"
        study_export_path = self.tuner_checkpoint_dir / study_filename
        self.exporter.export(resource=study, path=study_export_path)

    @staticmethod
    def report_study_results(study: optuna.Study):
        pruned_trials = study.get_trials(
            deepcopy=False, states=[TrialState.PRUNED]
        )
        complete_trials = study.get_trials(
            deepcopy=False, states=[TrialState.COMPLETE]
        )

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        print("  Value: ", study.best_trial.value)
        print("  Params: ")
        for key, value in study.best_trial.params.items():
            print("    {}: {}".format(key, value))

    def tune(
        self, num_trials: int, timeout: int | None = None
    ) -> optuna.Study:
        study = optuna.create_study(
            direction=self.optimization_direction_label,
            sampler=self.hyperparameter_sampler,
            pruner=self.pruner,
        )
        for trial_num in range(num_trials):
            study.optimize(func=self.objective_fn, n_trials=1, timeout=timeout)
            self.export_study(study=study)

        return study
