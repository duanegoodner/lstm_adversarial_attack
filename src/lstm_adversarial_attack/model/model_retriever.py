from enum import Enum, auto
from pathlib import Path

# import lstm_adversarial_attack.config_paths as cfg_paths
import lstm_adversarial_attack.path_searches as ps
import lstm_adversarial_attack.model.cross_validation_summarizer as cvs


class ModelAssessmentType(Enum):
    """
    No longer used, but keep here for compatibility with pickle files created
    when ModelRetriever used ModelAssessmentType
    """

    KFOLD = auto()
    SINGLE_FOLD = auto()


# @dataclass
# class FoldCheckpointPair:
#     fold: int
#     checkpoint: ds.TrainingCheckpoint


class ModelRetriever:
    """
    Retrieves model and best checkpoint info from CV assessment of model.
    """

    def __init__(
        self,
        # training_output_dir: Path = None,
        training_output_dir: Path
    ):
        """
        :param training_output_dir: directory containing the assessment
        (training) results
        """
        # if training_output_dir is None:
        #     training_output_dir = ps.latest_modified_file_with_name_condition(
        #         component_string=".tar",
        #         root_dir=cfg_paths.CV_ASSESSMENT_OUTPUT_DIR,
        #         comparison_type=ps.StringComparisonType.SUFFIX,
        #     ).parent.parent.parent
        self.training_output_dir = training_output_dir
        self.checkpoints_dir = self.training_output_dir / "checkpoints"
        # self.model_path = self.training_output_dir / "model.pickle"

    def get_representative_checkpoint(
        self,
        eval_metric: cvs.EvalMetric = cvs.EvalMetric.VALIDATION_LOSS,
        optimize_direction: cvs.OptimizeDirection = cvs.OptimizeDirection.MIN,
        rel_fold_result: cvs.RelativeFoldResult = cvs.RelativeFoldResult.MID_RANGE,
    ) -> cvs.CheckpointInfo:
        """
        Calls appropriate method for model and checkpoint retrieval (depends
        on type of assessment we are pulling data from)
        :param eval_metric: the metric to use as checkpoint selection criteria
        :param optimize_direction: min or max (val of selection criteria)
        :param rel_fold_result: specifies which fold to select based on value
        of metric (min, max, or mid-range).
        :return: a ModelPathCheckpointPair
        """
        return self.get_cv_trained_model(
            metric=eval_metric,
            optimize_direction=optimize_direction,
            rel_fold_result=rel_fold_result,
        )

    def get_single_fold_trained_model(
        self,
        eval_metric: cvs.EvalMetric,
        optimization_direction: cvs.OptimizeDirection,
    ) -> cvs.CheckpointInfo:
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

        return fold_summarizer.get_extreme_checkpoint_info(
            metric=eval_metric, optimize_direction=optimization_direction
        )

        # return cvs.FoldCheckpointInfoPair(
        #     fold=0, checkpoint_info=checkpoint_info
        # )

    def get_cv_trained_model(
        self,
        metric: cvs.EvalMetric,
        optimize_direction: cvs.OptimizeDirection,
        rel_fold_result: cvs.RelativeFoldResult,
    ) -> cvs.CheckpointInfo:
        """
         Gets a ModelPathCheckPointPair corresponding to selected model &
        checkpoint from cross validation model assessment. Gets best
        checkpoint from fold with median performance (or "near median" if
        have even number of folds)
        :param metric: metric to use as selection criteria
        :param optimize_direction: min or max (val of selection criteria)
        :param rel_fold_result: specifies which fold to select based on value
        of metric (min, max, or mid-range).
        :return:
        """
        cv_summarizer = cvs.CrossValidationSummarizer.from_cv_checkpoints_dir(
            cv_checkpoints_dir=self.checkpoints_dir
        )

        return cv_summarizer.get_fold_checkpoint_info(
            metric=metric,
            optimize_direction=optimize_direction,
            rel_fold_result=rel_fold_result,
        )


if __name__ == "__main__":
    kfold_model_retriever = ModelRetriever()
    kfold_model_checkpoint_info_pair = kfold_model_retriever.get_cv_trained_model(
        metric=cvs.EvalMetric.VALIDATION_LOSS,
        optimize_direction=cvs.OptimizeDirection.MIN,
        rel_fold_result=cvs.RelativeFoldResult.MID_RANGE
    )
