from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
import lstm_adversarial_attack.config_paths as lcp
import lstm_adversarial_attack.tune_train.cross_validation_summarizer as cvs


class ModelAssessmentType(Enum):
    KFOLD = auto()
    SINGLE_FOLD = auto()


@dataclass
class ModelPathCheckpointPair:
    model_path: Path
    checkpoint: dict


class ModelRetriever:
    _dispatch_dict = {
        ModelAssessmentType.KFOLD: lcp.CV_ASSESSMENT_OUTPUT_DIR,
        ModelAssessmentType.SINGLE_FOLD: lcp.SINGLE_FOLD_OUTPUT_DIR,
    }

    def __init__(
        self,
        assessment_type: ModelAssessmentType,
        training_output_dir: Path = None,
    ):
        self.assessment_type = assessment_type
        if training_output_dir is None:
            training_output_dir = cvs.get_newest_sub_dir(
                path=self._dispatch_dict[assessment_type]
            )
        self.training_output_dir = training_output_dir
        self.checkpoints_dir = self.training_output_dir / "checkpoints"
        self.model_path = self.training_output_dir / "model.pickle"

    def get_single_fold_trained_model(
        self,
        eval_metric: cvs.EvalMetric,
        optimization_direction: cvs.OptimizeDirection,
    ):
        fold_summarizer = cvs.FoldSummarizer.from_fold_checkpoint_dir(
            fold_checkpoint_dir=self.checkpoints_dir, fold_num=0
        )

        checkpoint = fold_summarizer.get_extreme_checkpoint(
            metric=eval_metric, optimize_direction=optimization_direction
        )

        return ModelPathCheckpointPair(
            model_path=self.model_path, checkpoint=checkpoint
        )

    def get_cv_trained_model(
        self,
        metric: cvs.EvalMetric,
        optimize_direction: cvs.OptimizeDirection,
    ):
        cv_summarizer = cvs.CrossValidationSummarizer.from_cv_checkpoints_dir(
            cv_checkpoints_dir=self.checkpoints_dir
        )

        midrange_checkpoint = cv_summarizer.get_midrange_checkpoint(
            metric=metric, optimize_direction=optimize_direction
        )

        return ModelPathCheckpointPair(
            model_path=self.model_path, checkpoint=midrange_checkpoint
        )


if __name__ == "__main__":
    kfold_model_retriever = ModelRetriever(
        assessment_type=ModelAssessmentType.KFOLD
    )
    kfold_model_checkpoint_pair = kfold_model_retriever.get_cv_trained_model(
        metric=cvs.EvalMetric.VALIDATION_LOSS,
        optimize_direction=cvs.OptimizeDirection.MIN
    )
