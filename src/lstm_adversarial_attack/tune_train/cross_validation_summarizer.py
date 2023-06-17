import pandas as pd
import sys
import torch
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.config_paths as lcp


class FoldSummarizer:
    def __init__(
        self, fold_checkpoints: list[dict], checkpoints_dirname: str = None
    ):
        fold_checkpoints.sort(key=lambda x: x["epoch_num"])
        self.fold_checkpoints = fold_checkpoints
        self.checkpoints_dirname = checkpoints_dirname

    @classmethod
    def from_fold_checkpoint_dir(cls, fold_checkpoint_dir: Path):
        checkpoint_files = sorted(fold_checkpoint_dir.glob("*.tar"))
        fold_checkpoints = [torch.load(item) for item in checkpoint_files]
        return cls(
            fold_checkpoints=fold_checkpoints,
            checkpoints_dirname=fold_checkpoint_dir.name,
        )

    @property
    def result_df(self) -> pd.DataFrame:
        result_dict = {
            "epoch": [item["epoch_num"] for item in self.fold_checkpoints],
            "train_loss": [
                item["train_log_entry"].result.loss
                for item in self.fold_checkpoints
            ],
            "validation_loss": [
                item["eval_log_entry"].result.validation_loss
                for item in self.fold_checkpoints
            ],
            "auc": [
                item["eval_log_entry"].result.AUC
                for item in self.fold_checkpoints
            ],
            "accuracy": [
                item["eval_log_entry"].result.accuracy
                for item in self.fold_checkpoints
            ],
            "f1": [
                item["eval_log_entry"].result.f1
                for item in self.fold_checkpoints
            ],
            "precision": [
                item["eval_log_entry"].result.precision
                for item in self.fold_checkpoints
            ],
            "recall": [
                item["eval_log_entry"].result.recall
                for item in self.fold_checkpoints
            ],
        }
        return pd.DataFrame(result_dict)


class CrossValidationSummarizer:
    def __init__(self, fold_summarizers: list[FoldSummarizer]):
        self.fold_summarizers = fold_summarizers

    @classmethod
    def from_cv_checkpoints_dir(cls, cv_checkpoints_dir: Path):
        fold_checkpoint_dirs = sorted(cv_checkpoints_dir.glob("fold*"))
        fold_summarizers = [
            FoldSummarizer.from_fold_checkpoint_dir(item)
            for item in fold_checkpoint_dirs
        ]
        return cls(fold_summarizers=fold_summarizers)

    @property
    def result_means_df(self) -> pd.DataFrame:
        all_fold_results = [item.result_df for item in self.fold_summarizers]
        stacked_result_dfs = pd.concat(all_fold_results, axis=0)
        return (
            stacked_result_dfs.groupby(stacked_result_dfs.epoch)
            .mean()
            .reset_index()
        )


if __name__ == "__main__":
    my_cv_checkpoints_dir = (
        lcp.CV_ASSESSMENT_OUTPUT_DIR
        / "2023-06-17_12_14_32.412523"
        / "checkpoints"
    )

    fold_0_summarizer = FoldSummarizer.from_fold_checkpoint_dir(
        fold_checkpoint_dir=my_cv_checkpoints_dir / "fold_0"
    )
    my_cv_summarizer = CrossValidationSummarizer.from_cv_checkpoints_dir(
        cv_checkpoints_dir=my_cv_checkpoints_dir
    )

    my_cv_summarizer.result_means_df
