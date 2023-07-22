from dataclasses import dataclass
from enum import auto, Enum
from pathlib import Path
import lstm_adversarial_attack.config_paths as lcp
import lstm_adversarial_attack.tune_train.cross_validation_summarizer as cvs


class ModelAssessmentType(Enum):
    """
    No longer used, but keep here for compatibility with pickle files created
    when ModelRetriever used ModelAssessmentType
    """
    KFOLD = auto()
    SINGLE_FOLD = auto()


@dataclass
class ModelPathFoldCheckpointTrio:
    model_path: Path
    fold: int
    checkpoint: dict


class ModelRetriever:
    """
    Retrieves model and best checkpoint info from CV assessment of model.
    """

    def __init__(
        self,
        training_output_dir: Path = None,
    ):
        """
        :param training_output_dir: directory containing the assessment
        (training) results
        """
        if training_output_dir is None:
            training_output_dir = cvs.get_newest_sub_dir(
                path=lcp.CV_ASSESSMENT_OUTPUT_DIR
            )
        self.training_output_dir = training_output_dir
        self.checkpoints_dir = self.training_output_dir / "checkpoints"
        self.model_path = self.training_output_dir / "model.pickle"

    def get_model(
        self,
        eval_metric: cvs.EvalMetric,
        optimize_direction: cvs.OptimizeDirection,
    ) -> ModelPathFoldCheckpointTrio:
        """
        Calls appropriate method for model and checkpoint retrieval (depends
        on type of assessment we are pulling data from)
        :param eval_metric: the metric to use as checkpoint selection criteria
        :param optimize_direction: min or max (val of selection criteria)
        :return: a ModelPathCheckpointPair
        """
        return self.get_cv_trained_model(
            eval_metric, optimize_direction
        )

    def get_single_fold_trained_model(
        self,
        eval_metric: cvs.EvalMetric,
        optimization_direction: cvs.OptimizeDirection,
    ) -> ModelPathFoldCheckpointTrio:
        """
        Gets a ModelPathCheckPointPair corresponding to selected model &
        checkpoint from single fold evaluation.
        :param eval_metric: metric to use as checkpoint selection criteria
        :param optimization_direction: min or max (val of selection criteria)
        :return: a ModelPathCheckpointPair
        """
        fold_summarizer = cvs.FoldSummarizer.from_fold_checkpoint_dir(
            fold_checkpoint_dir=self.checkpoints_dir, fold_num=0
        )

        checkpoint = fold_summarizer.get_extreme_checkpoint(
            metric=eval_metric, optimize_direction=optimization_direction
        )

        return ModelPathFoldCheckpointTrio(
            model_path=self.model_path,
            fold=0,
            checkpoint=checkpoint
        )

    def get_cv_trained_model(
        self,
        metric: cvs.EvalMetric,
        optimize_direction: cvs.OptimizeDirection,
    ) -> ModelPathFoldCheckpointTrio:
        """
         Gets a ModelPathCheckPointPair corresponding to selected model &
        checkpoint from cross validation model assessment. Gets best
        checkpoint from fold with median performance (or "near median" if
        have even number of folds)
        :param metric: metric to use as selection criteria
        :param optimize_direction: min or max (val of selection criteria)
        :return:
        """
        cv_summarizer = cvs.CrossValidationSummarizer.from_cv_checkpoints_dir(
            cv_checkpoints_dir=self.checkpoints_dir
        )

        midrange_fold_checkpoint_pair = cv_summarizer.get_midrange_checkpoint(
            metric=metric, optimize_direction=optimize_direction
        )

        return ModelPathFoldCheckpointTrio(
            model_path=self.model_path,
            fold=midrange_fold_checkpoint_pair.fold,
            checkpoint=midrange_fold_checkpoint_pair.checkpoint
        )


if __name__ == "__main__":
    kfold_model_retriever = ModelRetriever(
    )
    kfold_model_checkpoint_pair = kfold_model_retriever.get_cv_trained_model(
        metric=cvs.EvalMetric.VALIDATION_LOSS,
        optimize_direction=cvs.OptimizeDirection.MIN,
    )
