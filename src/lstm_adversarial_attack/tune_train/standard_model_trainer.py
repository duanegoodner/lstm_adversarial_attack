import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import sklearn.metrics as skm
import torch.nn as nn
import torch.optim
import torch.utils.data as ud
from torch.utils.tensorboard import SummaryWriter

sys.path.append(str(Path(__file__).parent.parent))
import lstm_adversarial_attack.config_settings as cfs
import lstm_adversarial_attack.data_structures as ds
import lstm_adversarial_attack.resource_io as rio


class StandardModelTrainer:
    """
    Trains and evaluates a model. Can save checkpoints & write to Tensorboard.
    """

    def __init__(
        self,
        # train_device: torch.device,
        # eval_device: torch.device,
        device: torch.device,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: ud.DataLoader,
        test_loader: ud.DataLoader,
        checkpoint_dir: Path,
        epoch_start_count: int = 0,
        summary_writer: SummaryWriter = None,
        summary_writer_group: str = "",
        summary_writer_subgroup: str = "",
    ):
        self.device = device
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.checkpoint_dir = checkpoint_dir
        if not checkpoint_dir.exists():
            checkpoint_dir.mkdir()
        self.completed_epochs = epoch_start_count
        self.summary_writer = summary_writer
        self.summary_writer_group = summary_writer_group
        self.summary_writer_subgroup = summary_writer_subgroup
        self.train_log = ds.TrainLog()
        self.eval_log = ds.EvalLog()

    @staticmethod
    def calculate_performance_metrics(
        y_score: torch.tensor, y_pred: torch.tensor, y_true: torch.tensor
    ) -> ds.ClassificationScores:
        """
        Calcs performance metrics using data returned by self.evaluate_model()
        :param y_score: float outputs of final layer
        :param y_pred: predicted classes
        :param y_true: actual classes
        :return: ClassificationScores container w/ calculated metrics
        """
        y_true_one_hot = torch.nn.functional.one_hot(y_true)
        y_score_np = y_score.detach().numpy()
        y_pred_np = y_pred.detach().numpy()
        y_true_np = y_true.detach().numpy()

        return ds.ClassificationScores(
            accuracy=skm.accuracy_score(y_true=y_true_np, y_pred=y_pred_np),
            auc=skm.roc_auc_score(y_true=y_true_one_hot, y_score=y_score_np),
            precision=skm.precision_score(y_true=y_true_np, y_pred=y_pred_np),
            recall=skm.recall_score(y_true=y_true_np, y_pred=y_pred_np),
            f1=skm.f1_score(y_true=y_true_np, y_pred=y_pred_np),
        )

    @property
    def _current_checkpoint_info(self) -> dict[str, Any]:
        return {
            "epoch_num": deepcopy(self.completed_epochs),
            "train_log_entry": deepcopy(self.train_log.data[-1]),
            "eval_log_entry": deepcopy(self.eval_log.data[-1]),
            "state_dict": deepcopy(self.model.state_dict()),
            "optimizer_state_dict": deepcopy(self.optimizer.state_dict()),
        }

    def _save_checkpoint(
        self,
    ) -> dict[str, Any]:
        """
        Saves checkpoint w/ model/optimizer params & latest train/eval results
        :return: path of file where checkpoint is saved
        """
        output_path = rio.create_timestamped_filepath(
            parent_path=self.checkpoint_dir, file_extension="tar"
        )
        return output_path

    def train_model(
        self,
        num_epochs: int,
    ):
        """
        Trains model for num_epochs. Stores results in self.train_log.

        Optionally writes to Tensorboard SummaryWriter.
        :param num_epochs: number of epochs to run training
        """
        self.model.to(self.device)
        self.model.train()

        for epoch in range(num_epochs):
            running_loss = 0.0
            for num_batches, (inputs, y) in enumerate(self.train_loader):
                inputs.features, y = (
                    inputs.features.to(self.device),
                    y.to(self.device),
                )
                self.optimizer.zero_grad()
                y_hat = self.model(inputs).squeeze()
                loss = self.loss_fn(y_hat, y)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            epoch_loss = running_loss / (num_batches + 1)

            self.completed_epochs += 1
            self.train_log.update(
                entry=ds.TrainLogEntry(
                    epoch=self.completed_epochs,
                    result=ds.TrainEpochResult(loss=epoch_loss),
                )
            )
            self.report_epoch_loss(epoch_loss=epoch_loss)

    def report_epoch_loss(self, epoch_loss: float):
        """
        Writes loss val to Tensorboard output using SummaryWriter
        :param epoch_loss: loss val for epoch
        """
        print(
            f"{self.summary_writer_subgroup}, epoch_{self.completed_epochs},"
            f" Loss: {epoch_loss:.4f}"
        )

        if self.summary_writer is not None:
            self.summary_writer.add_scalars(
                f"{self.summary_writer_group}/_training_loss",
                {
                    f"{self.summary_writer_subgroup}": epoch_loss,
                },
                self.completed_epochs,
            )

    @torch.no_grad()
    def evaluate_model(self) -> ds.EvalLogEntry:
        """
        Evaluates model. Calculates and stores perfromance metrics.
        """
        running_loss = 0.0
        self.model.to(self.device)
        self.model.eval()
        all_y_true = torch.LongTensor()
        all_y_pred = torch.LongTensor()
        all_y_score = torch.FloatTensor()
        for num_batches, (inputs, y) in enumerate(self.train_loader):
            inputs.features, y = inputs.features.to(self.device), y.to(
                self.device
            )
            y_hat = self.model(inputs)
            loss = self.loss_fn(y_hat, y)
            running_loss += loss.item()
            y_pred = torch.argmax(input=y_hat, dim=1)
            all_y_true = torch.cat((all_y_true, y.to("cpu")), dim=0)
            all_y_pred = torch.cat((all_y_pred, y_pred.to("cpu")), dim=0)
            all_y_score = torch.cat((all_y_score, y_hat.to("cpu")), dim=0)
        epoch_loss = running_loss / (num_batches + 1)
        classification_scores = self.calculate_performance_metrics(
            y_score=all_y_score, y_pred=all_y_pred, y_true=all_y_true
        )
        eval_results = ds.EvalEpochResult(
            validation_loss=epoch_loss, **classification_scores.__dict__
        )

        eval_log_entry = ds.EvalLogEntry(
            epoch=self.completed_epochs, result=eval_results
        )

        self.eval_log.update(
            eval_log_entry
        )
        self.report_eval_results(
            eval_results=eval_results,
        )

        return eval_log_entry

    def report_eval_results(
        self,
        eval_results: ds.EvalEpochResult,
    ):
        """
        Writes eval results to Tensorboard via SummaryWriter
        :param eval_results:
        """
        print(
            f"\n{self.summary_writer_subgroup} performance on test"
            f" data:\n{eval_results}\n"
        )

        metrics_of_interest = [
            "accuracy",
            "auc",
            "f1",
            "precision",
            "recall",
            "validation_loss",
        ]

        if self.summary_writer is not None:
            report_attributes = [
                "accuracy",
                "auc",
                "f1",
                "precision",
                "recall",
                "validation_loss",
            ]
            for attribute in report_attributes:
                self.summary_writer.add_scalars(
                    f"{self.summary_writer_group}/{cfs.ATTR_DISPLAY[attribute]}",
                    {
                        f"{self.summary_writer_subgroup}": getattr(
                            eval_results, attribute
                        )
                    },
                    self.completed_epochs,
                )



    def run_train_eval_cycles(
        self,
        num_epochs: int,
        eval_interval: int,
        save_checkpoints: bool = False,
    ):
        """
        Runs train/eval cycles and optionally saves checkpoints.
        :param num_epochs: total number of epochs to run
        :param eval_interval: number of train epochs per eval
        :param save_checkpoints: whether to save checkpoints
        :return: object containing logs of train and eval data
        """

        for epoch in range(num_epochs):
            self.train_model(num_epochs=1)
            if (epoch + 1) % eval_interval == 0:
                self.evaluate_model()
                if save_checkpoints:
                    self._save_checkpoint()
