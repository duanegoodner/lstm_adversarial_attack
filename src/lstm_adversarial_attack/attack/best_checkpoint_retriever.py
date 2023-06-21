import sys
import torch
from enum import Enum, auto
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.config_paths as cfg_paths


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


class BestCheckpointRetriever:
    _metric_dispatch = {
        EvalMetric.AUC: "AUC",
        EvalMetric.ACCURACY: "accuracy",
        EvalMetric.F1: "f1",
        EvalMetric.PRECISION: "precision",
        EvalMetric.RECALL: "recall",
        EvalMetric.VALIDATION_LOSS: "validation_loss",
    }

    def __init__(self, checkpoint_paths: list[Path]):
        self.checkpoint_paths = checkpoint_paths

    @classmethod
    def from_checkpoints_dir(cls, checkpoints_dir: Path):
        return cls(checkpoint_paths=list(checkpoints_dir.glob("*.tar")))

    def get_extreme_checkpoint(
        self, metric: EvalMetric, direction: OptimizeDirection
    ) -> dict:
        checkpoints = [torch.load(item) for item in self.checkpoint_paths]
        checkpoints.sort(
            key=lambda x: getattr(
                x["eval_log"].latest_entry.result,
                self._metric_dispatch[metric],
            ),
            reverse=(direction == OptimizeDirection.MAX),
        )
        return checkpoints[0]


if __name__ == "__main__":
    my_checkpoints_dir = (
        cfg_paths.TRAINING_OUTPUT_DIR / "2023-06-14_14_40_10.365521" / "checkpoints"
    )
    checkpoint_retriever = BestCheckpointRetriever.from_checkpoints_dir(
        checkpoints_dir=my_checkpoints_dir
    )
    best_checkpoint = checkpoint_retriever.get_extreme_checkpoint(
        metric=EvalMetric.VALIDATION_LOSS, direction=OptimizeDirection.MIN
    )
