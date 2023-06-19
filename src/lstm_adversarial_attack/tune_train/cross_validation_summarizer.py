import pandas as pd
import sys
import torch
from enum import Enum, auto
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.config_paths as lcp


class OptimizeDirection(Enum):
    MAX = auto()
    MIN = auto()


class EvalMetric(Enum):
    AUC = auto()
    ACCURACY = auto()
    F1 = auto()
    PRECISION = auto()
    RECALL = auto()
    VALIDATION_LOSS = auto()


class FoldSummarizer:
    def __init__(
        self,
        fold_checkpoints: list[dict],
        checkpoints_dirname: str = None,
        fold_num: int = None,
    ):
        fold_checkpoints.sort(key=lambda x: x["epoch_num"])
        self.fold_checkpoints = fold_checkpoints
        self.checkpoints_dirname = checkpoints_dirname
        self.fold_num = fold_num

    _metric_dispatch = {
        EvalMetric.AUC: "auc",
        EvalMetric.ACCURACY: "accuracy",
        EvalMetric.F1: "f1",
        EvalMetric.PRECISION: "precision",
        EvalMetric.RECALL: "recall",
        EvalMetric.VALIDATION_LOSS: "validation_loss",
    }

    @classmethod
    def from_fold_checkpoint_dir(
        cls, fold_checkpoint_dir: Path, fold_num: int
    ):
        checkpoint_files = sorted(fold_checkpoint_dir.glob("*.tar"))
        fold_checkpoints = [torch.load(item) for item in checkpoint_files]
        return cls(
            fold_checkpoints=fold_checkpoints,
            checkpoints_dirname=fold_checkpoint_dir.name,
            fold_num=fold_num,
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

    def _get_optimal_idx(
        self, metric: EvalMetric, optimize_direction: OptimizeDirection
    ) -> int:
        if optimize_direction == OptimizeDirection.MIN:
            extreme_idx = (
                self.result_df[[self._metric_dispatch[metric]]].idxmin().item()
            )
        else:
            extreme_idx = (
                self.result_df[[self._metric_dispatch[metric]]].idxmax().item()
            )
        return extreme_idx

    def get_extreme_checkpoint(
        self, metric: EvalMetric, optimize_direction: OptimizeDirection
    ) -> dict:
        extreme_idx = self._get_optimal_idx(
            metric=metric, optimize_direction=optimize_direction
        )
        return self.fold_checkpoints[extreme_idx]

    def get_optimal_result_row(
        self, metric: EvalMetric, optimize_direction: OptimizeDirection
    ) -> pd.DataFrame:
        idx = self._get_optimal_idx(
            metric=metric, optimize_direction=optimize_direction
        )
        return self.result_df.loc[idx].rename(self.fold_num)


class CrossValidationSummarizer:
    def __init__(self, fold_summarizers: list[FoldSummarizer]):
        self.fold_summarizers = fold_summarizers

    @classmethod
    def from_cv_checkpoints_dir(cls, cv_checkpoints_dir: Path = None):
        if cv_checkpoints_dir is None:
            cv_output_root = sorted(
                [
                    item
                    for item in lcp.CV_ASSESSMENT_OUTPUT_DIR.iterdir()
                    if item.is_dir()
                ]
            )[-1]
            cv_checkpoints_dir = cv_output_root / "checkpoints"

        fold_checkpoint_dirs = sorted(cv_checkpoints_dir.glob("fold*"))
        fold_summarizers = [
            FoldSummarizer.from_fold_checkpoint_dir(
                fold_checkpoint_dir=fold_checkpoint_dirs[i], fold_num=i
            )
            for i in range(len(fold_checkpoint_dirs))
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

    def get_optimal_checkpoints(
        self, metric: EvalMetric, optimize_direction: OptimizeDirection
    ) -> list[dict]:
        return [
            fold.get_extreme_checkpoint(
                metric=metric, optimize_direction=optimize_direction
            )
            for fold in self.fold_summarizers
        ]

    def get_optimal_results_df(
        self, metric: EvalMetric, optimize_direction: OptimizeDirection
    ) -> pd.DataFrame:
        best_rows = [
            item.get_optimal_result_row(
                metric=metric, optimize_direction=optimize_direction
            )
            for item in self.fold_summarizers
        ]
        df = pd.concat(best_rows, axis=1).T
        df["epoch"] = df["epoch"].astype("int")
        return df


def main():

    cv_summarizer = CrossValidationSummarizer.from_cv_checkpoints_dir()
    optimal_results_df = cv_summarizer.get_optimal_results_df(
        metric=EvalMetric.VALIDATION_LOSS,
        optimize_direction=OptimizeDirection.MIN,
    )

    return optimal_results_df


if __name__ == "__main__":
    main()
