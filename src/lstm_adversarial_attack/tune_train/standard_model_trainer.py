import sklearn.metrics as skm
import sys
import torch.nn as nn
import torch.optim
import torch.utils.data as ud
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

sys.path.append(str(Path(__file__).parent.parent))
import lstm_adversarial_attack.resource_io as rio
from lstm_adversarial_attack.data_structures import (
    ClassificationScores,
    TrainEpochResult,
    TrainLogEntry,
    TrainLog,
    EvalEpochResult,
    EvalLogEntry,
    EvalLog,
    FullEvalResult,
    TrainEvalLogPair,
)


class StandardModelTrainer:
    def __init__(
        self,
        train_device: torch.device,
        eval_device: torch.device,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: ud.DataLoader,
        test_loader: ud.DataLoader,
        checkpoint_dir: Path,
        epoch_start_count: int = 0,
        train_log: TrainLog = TrainLog(),
        eval_log: EvalLog = EvalLog(),
        summary_writer: SummaryWriter = None,
        summary_writer_group: str = "",
        summary_writer_subgroup: str = "",
    ):
        self.train_device = train_device
        self.eval_device = eval_device
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
        self.train_log = train_log
        self.eval_log = eval_log

    @staticmethod
    def calculate_performance_metrics(
        y_score: torch.tensor, y_pred: torch.tensor, y_true: torch.tensor
    ) -> ClassificationScores:
        y_true_one_hot = torch.nn.functional.one_hot(y_true)
        y_score_np = y_score.detach().numpy()
        y_pred_np = y_pred.detach().numpy()
        y_true_np = y_true.detach().numpy()

        return ClassificationScores(
            accuracy=skm.accuracy_score(y_true=y_true_np, y_pred=y_pred_np),
            AUC=skm.roc_auc_score(y_true=y_true_one_hot, y_score=y_score_np),
            precision=skm.precision_score(y_true=y_true_np, y_pred=y_pred_np),
            recall=skm.recall_score(y_true=y_true_np, y_pred=y_pred_np),
            f1=skm.f1_score(y_true=y_true_np, y_pred=y_pred_np),
        )

    def reset_epoch_counts(self):
        self.completed_epochs = 0

    def save_checkpoint(
        self,
    ) -> Path:
        output_path = rio.create_timestamped_filepath(
            parent_path=self.checkpoint_dir, file_extension="tar"
        )
        output_object = {
            "epoch_num": self.completed_epochs,
            "train_log": self.train_log,
            "eval_log": self.eval_log,
            "state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(obj=output_object, f=output_path)
        return output_path

    def train_model(
        self,
        num_epochs: int,
    ):
        self.model.to(self.train_device)
        self.model.train()

        for epoch in range(num_epochs):
            running_loss = 0.0
            for num_batches, (inputs, y) in enumerate(self.train_loader):
                inputs.features, y = (
                    inputs.features.to(self.train_device),
                    y.to(self.train_device),
                )
                self.optimizer.zero_grad()
                y_hat = self.model(inputs).squeeze()
                loss = self.loss_fn(y_hat, y)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            epoch_loss = running_loss / (num_batches + 1)

            self.train_log.update(
                entry=TrainLogEntry(
                    epoch=self.completed_epochs,
                    result=TrainEpochResult(loss=epoch_loss),
                )
            )
            self.completed_epochs += 1
            self.report_epoch_loss(epoch_loss=epoch_loss)

    def report_epoch_loss(self, epoch_loss: float):
        print(
            f"{self.summary_writer_subgroup}, epoch_{self.completed_epochs},"
            f" Loss: {epoch_loss:.4f}"
        )

        if self.summary_writer is not None:
            self.summary_writer.add_scalars(
                f"{self.summary_writer_group}/training_loss",
                {
                    f"{self.summary_writer_subgroup}": epoch_loss,
                },
                self.completed_epochs,
            )

    @torch.no_grad()
    def evaluate_model(self, return_results: bool = False):
        running_loss = 0.0
        self.model.to(self.eval_device)
        self.model.eval()
        all_y_true = torch.LongTensor()
        all_y_pred = torch.LongTensor()
        all_y_score = torch.FloatTensor()
        for num_batches, (inputs, y) in enumerate(self.train_loader):
            inputs.features, y = inputs.features.to(self.eval_device), y.to(
                self.eval_device
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
        eval_results = EvalEpochResult(
            validation_loss=epoch_loss, **classification_scores.__dict__
        )

        self.eval_log.update(
            EvalLogEntry(epoch=self.completed_epochs, result=eval_results)
        )
        self.report_eval_results(
            eval_results=eval_results,
        )
        if return_results:
            return FullEvalResult(
                metrics=eval_results,
                y_pred=all_y_true,
                y_score=all_y_score,
                y_true=all_y_true,
            )

    def report_eval_results(
        self,
        eval_results: EvalEpochResult,
    ):
        print(
            f"\n{self.summary_writer_subgroup} performance on test"
            f" data:\n{eval_results}\n"
        )

        if self.summary_writer is not None:
            self.summary_writer.add_scalars(
                f"{self.summary_writer_group}/AUC",
                {f"{self.summary_writer_subgroup}": eval_results.AUC},
                self.completed_epochs,
            )
            self.summary_writer.add_scalars(
                f"{self.summary_writer_group}/validation_loss",
                {
                    f"{self.summary_writer_subgroup}": (
                        eval_results.validation_loss
                    ),
                },
                self.completed_epochs,
            )

    def run_train_eval_cycles(
        self,
        num_cycles: int,
        epochs_per_cycle: int = 1,
        save_checkpoints: bool = False,
        num_cycles_per_checkpoint: int = 10,
    ):
        for cycle_num in range(num_cycles):
            self.train_model(num_epochs=epochs_per_cycle)
            self.evaluate_model()
            if (
                save_checkpoints
                and ((cycle_num + 1) % num_cycles_per_checkpoint) == 0
            ):
                self.save_checkpoint()

        return TrainEvalLogPair(train=self.train_log, eval=self.eval_log)
