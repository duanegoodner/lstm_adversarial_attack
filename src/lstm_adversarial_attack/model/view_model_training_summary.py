import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.model.cross_validation_summarizer as cvs
import lstm_adversarial_attack.utils.path_searches as ps
from lstm_adversarial_attack.config.read_write import PATH_CONFIG_READER


def main(cv_training_id: str = None):
    cv_root_dir = Path(PATH_CONFIG_READER.read_path("model.cv_driver.output_dir"))

    if cv_training_id is None:
        cv_training_id = ps.get_latest_sequential_child_dirname(
            root_dir=cv_root_dir
        )

    cv_checkpoints_dir = cv_root_dir / cv_training_id / "checkpoints"

    cv_summarizer = cvs.CrossValidationSummarizer.from_cv_checkpoints_dir(
        cv_checkpoints_dir=cv_checkpoints_dir
    )
    optimal_results_df = cv_summarizer.get_optimal_results_df(
        metric=cvs.EvalMetric.VALIDATION_LOSS,
        optimize_direction=cvs.OptimizeDirection.MIN,
    )

    summary_df = optimal_results_df.describe().loc[
        ["mean", "std"],
        (optimal_results_df.columns != "epoch")
        & (optimal_results_df.columns != "fold"),
    ]

    print(f"Summary of Cross Validation Training Session {cv_training_id}\n")

    print("Best Performing Checkpoints by Fold")
    print(optimal_results_df.T)

    print("\nPerformance Metrics Means and Standard Deviations")
    print(summary_df.T)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Displays summary of model training session"
    )
    parser.add_argument(
        "-t",
        "--cv_training_id",
        type=str,
        action="store",
        nargs="?",
        help="ID of cross validation training session to summarize. Defaults "
        "to most recently created session",
    )
    args_namespace = parser.parse_args()
    main(cv_training_id=args_namespace.cv_training_id)
