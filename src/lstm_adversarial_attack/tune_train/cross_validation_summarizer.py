from dataclasses import dataclass

import pandas as pd
import sys
import torch
from enum import Enum, auto
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.config_paths as lcp


def get_newest_sub_dir(path: Path) -> Path | None:
    assert path.is_dir()
    sub_dirs = [item for item in path.iterdir() if item.is_dir()]
    if len(sub_dirs) != 0:
        return sorted(sub_dirs, key=lambda x: x.name)[-1]
    else:
        return None


class OptimizeDirection(Enum):
    """
    For use by methods in FoldSummarizer and CrossValidation Summarizer
    """

    MAX = auto()
    MIN = auto()


class EvalMetric(Enum):
    """
    For use by methods in FoldSummarizer and CrossValidation Summarizer
    """

    AUC = auto()
    ACCURACY = auto()
    F1 = auto()
    PRECISION = auto()
    RECALL = auto()
    VALIDATION_LOSS = auto()


class FoldSummarizer:
    """
    Summarizes train and eval data from single fold
    """

    def __init__(
        self,
        fold_checkpoints: list[dict],
        checkpoints_dirname: str = None,
        fold_num: int = None,
    ):
        """
        :param fold_checkpoints: list of checkpoint dictionaries
        :param checkpoints_dirname: directory where checkpoints are stored
        :param fold_num: integer index of fold
        """
        fold_checkpoints.sort(key=lambda x: x["epoch_num"])
        self.fold_checkpoints = fold_checkpoints
        self.checkpoints_dirname = checkpoints_dirname
        self.fold_num = fold_num

    # translated enums into object attributes
    metric_dispatch = {
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
        """
        Creates FoldSummarizer from provided path to dir containing checkpoints
        :param fold_checkpoint_dir: directory containing checkpoints
        :param fold_num: integer index of fold
        """
        checkpoint_files = sorted(fold_checkpoint_dir.glob("*.tar"))
        fold_checkpoints = [torch.load(item) for item in checkpoint_files]
        return cls(
            fold_checkpoints=fold_checkpoints,
            checkpoints_dirname=fold_checkpoint_dir.name,
            fold_num=fold_num,
        )

    @property
    def result_df(self) -> pd.DataFrame:
        """
        :return: Dataframe of eval metrics and train loss. 1 row per epoch
        """
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
                item["eval_log_entry"].result.auc
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
        """
        Gets results dataframe index of extreme (min or max) val of a metric
        :param metric: performance metric to use as criteria
        :param optimize_direction: min or max
        :return: index of row containing extreme val of metric
        """
        if optimize_direction == OptimizeDirection.MIN:
            extreme_idx = (
                self.result_df[[self.metric_dispatch[metric]]].idxmin().item()
            )
        else:
            extreme_idx = (
                self.result_df[[self.metric_dispatch[metric]]].idxmax().item()
            )
        return extreme_idx

    def get_extreme_checkpoint(
        self, metric: EvalMetric, optimize_direction: OptimizeDirection
    ) -> dict:
        """
        Gets the checkpoint dict corresponding to an extreme val of a metric
        :param metric: metric for criteria
        :param optimize_direction: min or max
        :return: a checkpoint dictionary
        """
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


@dataclass
class FoldCheckpointPair:
    fold: int
    checkpoint: dict


class CrossValidationSummarizer:
    def __init__(self, fold_summarizers: dict[int, FoldSummarizer]):
        self.fold_summarizers = fold_summarizers

    @classmethod
    def from_cv_checkpoints_dir(cls, cv_checkpoints_dir: Path = None):
        if cv_checkpoints_dir is None:
            cv_output_root = get_newest_sub_dir(
                path=lcp.CV_ASSESSMENT_OUTPUT_DIR
            )
            cv_checkpoints_dir = cv_output_root / "checkpoints"

        fold_checkpoint_dirs = list(cv_checkpoints_dir.glob("fold*"))
        fold_checkpoint_dirs.sort(key=lambda x: x.stat().st_mtime)
        fold_summarizers = {
            fold_index: FoldSummarizer.from_fold_checkpoint_dir(
                fold_checkpoint_dir=fold_checkpoint_dirs[fold_index],
                fold_num=fold_index,
            )
            for fold_index in range(len(fold_checkpoint_dirs))
        }
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
            for index, fold in self.fold_summarizers.items()
        ]

    def get_optimal_results_df(
        self, metric: EvalMetric, optimize_direction: OptimizeDirection
    ) -> pd.DataFrame:
        """
        Gets best epoch result wrt metric & optimize direction for each fold
        :param metric: metric to find min or max of
        :param optimize_direction: min or max
        :return: dataframe (cols --> metrics, rows --> folds)
        """
        best_rows = [
            fold.get_optimal_result_row(
                metric=metric, optimize_direction=optimize_direction
            )
            for index, fold in self.fold_summarizers.items()
        ]
        df = pd.concat(best_rows, axis=1).T.reset_index(names="fold")
        df["epoch"] = df["epoch"].astype("int")
        return df

    def get_midrange_fold(
        self, metric: EvalMetric, optimize_direction: OptimizeDirection
    ):
        optimal_results_df = self.get_optimal_results_df(
            metric=metric, optimize_direction=optimize_direction
        )
        metric_name = FoldSummarizer.metric_dispatch[metric]
        sorted_optimal_results_df = optimal_results_df.sort_values(
            by=[metric_name]
        )

        return sorted_optimal_results_df.iloc[
            int(len(sorted_optimal_results_df) / 2)
        ]["fold"].astype("int")

    def get_midrange_checkpoint(
        self, metric: EvalMetric, optimize_direction: OptimizeDirection
    ) -> FoldCheckpointPair:
        midrange_fold = self.get_midrange_fold(
            metric=metric, optimize_direction=optimize_direction
        )
        checkpoint = self.fold_summarizers[
            midrange_fold
        ].get_extreme_checkpoint(
            metric=metric, optimize_direction=optimize_direction
        )

        return FoldCheckpointPair(fold=midrange_fold, checkpoint=checkpoint)


def main():
    """
    Creates CrossValidationSummarizer from latest data in path defined by:
    config_paths.

    Gets best metrics from each fold.
    :return: df with 1 row per fold (has each folds best epoch's data)
    """

    cv_summarizer = CrossValidationSummarizer.from_cv_checkpoints_dir()
    optimal_results_df = cv_summarizer.get_optimal_results_df(
        metric=EvalMetric.VALIDATION_LOSS,
        optimize_direction=OptimizeDirection.MIN,
    )

    return optimal_results_df


if __name__ == "__main__":
    main()
