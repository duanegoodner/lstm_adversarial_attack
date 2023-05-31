import sklearn.metrics as skm
import torch.nn as nn
import torch.optim
import torch.utils.data as ud
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


# TODO Separate Trainer and Evaluator in to two classes

class ModuleWithDevice(nn.Module):
    def __init__(self, device: torch.device):
        super(ModuleWithDevice, self).__init__()
        self.device = device
        self.to(device)


@dataclass
class StandardClassificationMetrics:
    accuracy: float
    roc_auc: float
    precision: float
    recall: float
    f1: float

    def __str__(self) -> str:
        return (
            f"Accuracy:\t{self.accuracy:.4f}\n"
            f"AUC:\t\t{self.roc_auc:.4f}\n"
            f"Precision:\t{self.precision:.4f}\n"
            f"Recall:\t\t{self.recall:.4f}\n"
            f"F1:\t\t\t{self.f1:.4f}"
        )


class StandardModelTrainer:
    def __init__(
        self,
        model: ModuleWithDevice,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        save_checkpoints: bool,
        checkpoint_dir: Path = None,
        checkpoint_interval: int = 100,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.save_checkpoints = save_checkpoints
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval

    @staticmethod
    def interpret_output(model_output: torch.tensor) -> torch.tensor:
        return torch.argmax(input=model_output, dim=1)

    @staticmethod
    def calculate_performance_metrics(
        y_score: torch.tensor, y_pred: torch.tensor, y_true: torch.tensor
    ) -> StandardClassificationMetrics:
        y_true_one_hot = torch.nn.functional.one_hot(y_true)
        y_score_np = y_score.detach().numpy()
        y_pred_np = y_pred.detach().numpy()
        y_true_np = y_true.detach().numpy()

        return StandardClassificationMetrics(
            accuracy=skm.accuracy_score(y_true=y_true_np, y_pred=y_pred_np),
            roc_auc=skm.roc_auc_score(
                y_true=y_true_one_hot, y_score=y_score_np
            ),
            precision=skm.precision_score(y_true=y_true_np, y_pred=y_pred_np),
            recall=skm.recall_score(y_true=y_true_np, y_pred=y_pred_np),
            f1=skm.f1_score(y_true=y_true_np, y_pred=y_pred_np),
        )

    def save_checkpoint(
        self,
        epoch_num: int,
        loss: float,
        metrics: StandardClassificationMetrics,
    ) -> Path:
        filename = f"{datetime.now()}.tar".replace(" ", "_")
        output_path = self.checkpoint_dir / filename
        output_object = {
            "epoch_num": epoch_num,
            "loss": loss,
            "metrics": metrics,
            "state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(obj=output_object, f=output_path)
        return output_path

    def eval_model_and_save_checkpoint(
        self, epoch_num: int, epoch_loss: float, test_dataloader: ud.DataLoader
    ):
        metrics = self.evaluate_model(test_dataloader=test_dataloader)
        self.save_checkpoint(
            epoch_num=epoch_num, loss=epoch_loss, metrics=metrics
        )
        self.model.train()

    def train_model(
        self,
        num_epochs: int,
        train_dataloader: ud.DataLoader,
        test_dataloader: ud.DataLoader | None = None,
        loss_log: list = None,
    ):
        self.model.train()

        for epoch in range(num_epochs):
            running_loss = 0.0
            for num_batches, (x, y) in enumerate(train_dataloader):
                x, y = x.to(self.model.device), y.to(
                    self.model.device
                )
                self.optimizer.zero_grad()
                y_hat = self.model(x).squeeze()
                loss = self.loss_fn(y_hat, y)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            epoch_loss = running_loss / (num_batches + 1)
            # TODO move reporting work to separate method(s)
            if loss_log is not None:
                loss_log.append(epoch_loss)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")
            if (
                ((epoch + 1) % self.checkpoint_interval == 0)
                and self.save_checkpoints
                and (test_dataloader is not None)
                and (self.checkpoint_dir is not None)
            ):
                self.eval_model_and_save_checkpoint(
                    epoch_num=epoch,
                    epoch_loss=epoch_loss,
                    test_dataloader=test_dataloader,
                )

    @torch.no_grad()
    def evaluate_model(self, test_dataloader: ud.DataLoader):
        self.model.eval()
        all_y_true = torch.LongTensor()
        all_y_pred = torch.LongTensor()
        all_y_score = torch.FloatTensor()
        for x, y in test_dataloader:
            x, y = x.to(self.model.device), y.to(self.model.device)
            y_hat = self.model(x)
            y_pred = torch.argmax(input=y_hat, dim=1)
            all_y_true = torch.cat((all_y_true, y.to("cpu")), dim=0)
            all_y_pred = torch.cat((all_y_pred, y_pred.to("cpu")), dim=0)
            all_y_score = torch.cat((all_y_score, y_hat.to("cpu")), dim=0)
        metrics = self.calculate_performance_metrics(
            y_score=all_y_score, y_pred=all_y_pred, y_true=all_y_true
        )
        print(f"Predictive performance on test data:\n{metrics}\n")
        return metrics


# if __name__ == "__main__":
#     if torch.cuda.is_available():
#         cur_device = torch.device("cuda:0")
#     else:
#         cur_device = torch.device("cpu")
#
#     dataset = X19MortalityDataset()
#     data_loader = ud.DataLoader(dataset=dataset, batch_size=128, shuffle=True)
#     cur_model = LSTMSun2018(model_device=cur_device)
#     trainer = StandardModelTrainer(
#         model=cur_model,
#         # train_dataloader=data_loader,
#         # test_dataloader=data_loader,
#         loss_fn=nn.CrossEntropyLoss(),
#         optimizer=torch.optim.Adam(
#             params=cur_model.parameters(), lr=1e-4, betas=(0.5, 0.999)
#         ),
#         save_checkpoints=False,
#         checkpoint_dir=Path(__file__).parent.parent
#         / "data"
#         / "training_results"
#         / "troubleshooting_runs",
#     )
#
#     trainer.train_model(train_dataloader=data_loader, num_epochs=3)
#     trainer.evaluate_model(test_dataloader=data_loader)
