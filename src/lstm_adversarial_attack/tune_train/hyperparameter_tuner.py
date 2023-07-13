import optuna
import sys
import torch
import torch.nn as nn
from optuna.pruners import BasePruner, MedianPruner
from optuna.samplers import BaseSampler, TPESampler
from optuna.trial import TrialState
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
from typing import Callable

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.config_settings as cfg_set
import lstm_adversarial_attack.resource_io as rio
import lstm_adversarial_attack.data_structures as ds
import lstm_adversarial_attack.config_paths as cfg_paths
import lstm_adversarial_attack.tune_train.standard_model_trainer as smt
import lstm_adversarial_attack.weighted_dataloader_builder as wdb
import lstm_adversarial_attack.tune_train.tuner_helpers as tuh


# TODO Try to replace cross-validation work here with CrossValidator (if able
#   to match generic CrossValidator with requirements of Optuna study / trials)
class HyperParameterTuner:
    """
    Tunes hyperparameters. Runs K-fold CV for each set of hyperparams.
    Uses Optuna pruning and optimization objects.
    """
    def __init__(
        self,
        device: torch.device,
        dataset: Dataset,
        collate_fn: Callable,
        tuning_ranges: tuh.X19MLSTMTuningRanges,
        num_folds: int,
        num_cv_epochs: int,
        epochs_per_fold: int,
        fold_class: Callable,
        kfold_random_seed: int,
        cv_mean_metrics_of_interest: tuple[
            str
        ],
        performance_metric: str,
        optimization_direction: optuna.study.StudyDirection,
        pruner: BasePruner,
        hyperparameter_sampler: BaseSampler,
        output_dir: Path,
        # save_trial_info: bool = True,
        trial_prefix: str = "trial_",
        continue_study_path: Path = None,
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
        self.loss_fn = nn.CrossEntropyLoss()
        self.performance_metric = performance_metric
        self.optimization_direction = optimization_direction
        self.optimization_direction_label = (
            "minimize"
            if optimization_direction == optuna.study.StudyDirection.MINIMIZE
            else "maximize"
        )
        self.pruner = pruner
        self.cv_mean_metrics_of_interest = cv_mean_metrics_of_interest
        self.tuning_ranges = tuning_ranges
        self.hyperparameter_sampler = hyperparameter_sampler
        # if output_dir is None:
        #     output_dir = rio.create_timestamped_dir(
        #         parent_path=cfg_paths.HYPERPARAMETER_OUTPUT_DIR
        #     )
        self.output_dir = output_dir
        self.tensorboard_output_dir = self.output_dir / "tensorboard"
        self.trainer_checkpoint_dir = self.output_dir / "checkpoints_trainer"
        self.tuner_checkpoint_dir = self.output_dir / "checkpoints_tuner"
        self.exporter = rio.ResourceExporter()
        # self.save_trial_info = save_trial_info
        self.trial_prefix = trial_prefix
        self.continue_study_path = continue_study_path

    def create_datasets(self) -> list[tuh.TrainEvalDatasetPair]:
        """
        Creates the K-fold datasets.
        Same datasets are used for each set of hyperparameters.
        :return: train/eval dataset pair for each fold
        """
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
                tuh.TrainEvalDatasetPair(
                    train=train_dataset, validation=validation_dataset
                )
            )

        return all_train_eval_pairs

    @staticmethod
    def initialize_model(model: nn.Module):
        """
        Initializes params of model. Currently not used.

        May need if using same model object for different folds
        :param model: a Pytorch model
        """
        for name, param in model.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)

    def create_trainers(
        self,
        settings: tuh.X19LSTMHyperParameterSettings,
        summary_writer: SummaryWriter,
        trial_number: int,
    ) -> list[smt.StandardModelTrainer]:
        """
        Creates one StandardModelTrainer per fold.
        :param settings: hyperparameter settings for an optuna.Trial
        :param summary_writer: object that writes to Tensorboard
        :param trial_number: .number value of optuna.Trial using trainers
        :return: list containing the StandardModelTrainer objects
        """
        trainers = []
        for fold_idx, dataset_pair in enumerate(self.cv_datasets):
            model = tuh.X19LSTMBuilder(settings=settings).build()
            train_loader = wdb.WeightedDataLoaderBuilder(
                dataset=dataset_pair.train,
                batch_size=2**settings.log_batch_size,
                collate_fn=self.collate_fn,
            ).build()
            validation_loader = DataLoader(
                dataset=dataset_pair.validation,
                batch_size=128,
                shuffle=False,
                collate_fn=self.collate_fn,
            )

            trainer = smt.StandardModelTrainer(
                device=self.device,
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
        trial: optuna.Trial,
        trainers: list[smt.StandardModelTrainer],
        cv_means_log: ds.EvalLog,
    ):
        """
        Saves a completed optuna.Trial object to pickle file.

        Trial objective metric is a mean across all folds.
        :param trial: completed optuna.Trial object
        :param trainers: the StandardModelTrainers form trial (1 per fold)
        :param cv_means_log: log of cross-fold means of performance metrics
        """
        if not self.tuner_checkpoint_dir.exists():
            self.tuner_checkpoint_dir.mkdir()

        trial_summary = ds.CVTrialLogs(
            trainer_logs=[
                ds.TrainEvalLogPair(
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
        log_entry: ds.EvalLogEntry,
        summary_writer: SummaryWriter,
        trial: optuna.Trial,
    ):
        """
        Writes cross-fold mean metric data to Tensorboard
        :param log_entry: metric mean data from block of epochs
        :param summary_writer: object that writes to Tensorboard output dir
        :param trial: ongoing optuna trial that generated data
        """
        for metric in self.cv_mean_metrics_of_interest:
            summary_writer.add_scalar(
                f"{self.trial_prefix}{trial.number}/{metric}_mean",
                getattr(log_entry.result, metric),
                log_entry.epoch,
            )

    def build_objective_function_tools(self, trial: optuna.Trial):
        """
        Builds ObjectiveFunctionTools object to be used by trial
        :param trial:
        :return: dataclass containing objects for use during trial
        """
        settings = tuh.X19LSTMHyperParameterSettings.from_optuna_active_trial(
            trial=trial, tuning_ranges=self.tuning_ranges
        )
        summary_writer_path = (
            self.tensorboard_output_dir / f"{self.trial_prefix}{trial.number}"
        )

        return tuh.ObjectiveFunctionTools(
            settings=settings,
            summary_writer=SummaryWriter(str(summary_writer_path)),
            cv_means_log=ds.EvalLog(),
            trainers=self.create_trainers(
                settings=settings,
                summary_writer=SummaryWriter(str(summary_writer_path)),
                trial_number=trial.number,
            ),
        )

    def objective_fn(self, trial) -> float | None:
        """
        Runs cross-validation & returns mean of a performance metric across
        folds (metric specified by self.performance_metric). Epochs are run in
        blocks, and each block is completed for all folds before proceeding to
        next block. This allows Optuna to prune trials based on cross-fold mean
        values.
        :param trial: Current optuna Trial
        :return: mean value of a performance metric
        :rtype:
        """
        objective_tools = self.build_objective_function_tools(trial=trial)

        # TODO consider setting seed here to ensure WeightedRandomSampler
        #  makes same selections across all trials
        for cv_epoch in range(self.num_cv_epochs):
            eval_epoch_results = []
            for fold_idx, trainer in enumerate(objective_tools.trainers):
                trainer.train_model(num_epochs=self.epochs_per_fold)
                trainer.evaluate_model()
                eval_epoch_results.append(trainer.eval_log.latest_entry)
                trainer.model.to("cpu")

            mean_validation_vals = ds.EvalEpochResult.mean(
                [item.result for item in eval_epoch_results]
            )

            cv_means_log_entry = ds.EvalLogEntry(
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

            # report result so Optuna can determine if trial should be pruned
            trial.report(
                getattr(mean_validation_vals, self.performance_metric),
                cv_epoch,
            )

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # if self.save_trial_info:
        self.export_trial_info(
            trial=trial,
            trainers=objective_tools.trainers,
            cv_means_log=objective_tools.cv_means_log,
        )

        return tuh.PerformanceSelector(
            optimize_direction=self.optimization_direction
        ).choose_best_val(
            values=[
                getattr(item.result, self.performance_metric)
                for item in objective_tools.cv_means_log.data
            ]
        )

    def export_study(self, study: optuna.Study):
        """
        Saves optuna.Study object to pickle.
        :param study: the optuna.Study object ot be saved
        """
        if not self.tuner_checkpoint_dir.exists():
            self.tuner_checkpoint_dir.mkdir()
        study_filename = f"optuna_study.pickle"
        study_export_path = self.tuner_checkpoint_dir / study_filename
        self.exporter.export(resource=study, path=study_export_path)

    @staticmethod
    def report_study_results(study: optuna.Study):
        """
        Prints summary info form study to terminal
        :param study: an optuna.Study
        """
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
        """
        Initiate and run an optuna study.
        Result saved to self.tuner_checkpoint_dir
        :param num_trials: number of trials to run
        :param timeout: max time for study (default is no limit)
        """
        print(
            "Starting hyperparameter tuning.\n\n"
            "Data for Tensorboard will be written to:\n"
            f"{self.tensorboard_output_dir}\n\n"
            "Optuna trial and study objects will be saved in:\n"
            f"{self.tuner_checkpoint_dir}\n\n"
        )

        if self.continue_study_path is not None:
            study = rio.ResourceImporter().import_pickle_to_object(
                path=self.continue_study_path
            )
            assert study.direction == self.optimization_direction
        else:
            study = optuna.create_study(
                direction=self.optimization_direction_label,
                sampler=self.hyperparameter_sampler,
                pruner=self.pruner,
            )
        for trial_num in range(num_trials):
            study.optimize(func=self.objective_fn, n_trials=1, timeout=timeout)
            self.export_study(study=study)

        return study
