import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.attack_analysis.all_results_plotter as arp
import lstm_adversarial_attack.config_settings as cfg_settings


def main(
    attack_result_path: str = None,
    label: str = None,
    seq_length: int = None,
    min_num_perts: int = None,
    max_num_perts: int = None,
    output_dir: str = None,
    single_histograms: list[list] = None,
):
    """
    Generates set of plots using attack data. Plots are saved as .png. Also
    generates provenance.pickle and provenance.txt files with details of
    files and settings used to during attack hyperparamter tuning the final
    attack (using fixed set of hyperparameters).
    :param attack_result_path: Directory containing attack results. If not
    specified, defaults to parent directory of latest result under directory
    specified by config_paths.FROZEN_HYPERPARAMETER_ATTACK"
    # TODO currently not using label. Either start using it, or remove it.
    :param label: String to be included as part of plot titles
    :param seq_length: "Input sequence length of results to analyze. Defaults
    to config_settings.ATTACK_ANALYSIS_DEFAULT_SEQ_LENGTH which is typically
    the same as max observation window used during data preprocessing."
    :param min_num_perts: min number of nonzero perturbation elements required
    of examples to summarize"
    :param max_num_perts: max number of nonzero perturbation elements required
    of examples to summarize
    :param output_dir: Directory where plot figures will be saved. If not
    specified, a new directory with timestamp in name will be created under
    data/attack_analysis"
    :param single_histograms: "Information for histograms to be plotted separately from grid of histograms.Must be of form:\n
    grid_index[0] grid_index[1] num_bins x_min x_max.
    grid_index are the coordinates of the plot within the grid that contains
    the data to be replotted. num_bins is the number of bins to use in the
    histogram. x_min & x_max specify the range of bin values"
    :return:
    """
    if attack_result_path is not None:
        attack_result_path = Path(attack_result_path)
    if seq_length is None:
        seq_length = cfg_settings.ATTACK_ANALYSIS_DEFAULT_SEQ_LENGTH
    if single_histograms is not None:
        single_histograms_info = [
            arp.SingleHistogramInfo(entry) for entry in single_histograms
        ]
    else:
        single_histograms_info = None

    plotter = arp.AllResultsPlotter(
        attack_result_path=attack_result_path,
        seq_length=seq_length,
        min_num_perts=min_num_perts,
        max_num_perts=max_num_perts,
        label=label,
        output_dir=output_dir,
        single_histograms_info=single_histograms_info,
    )

    plotter.plot_all()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--attack_result_path",
        type=str,
        action="store",
        nargs="?",
        help=(
            "Directory containing attack results. If not specified, defaults "
            "to parent directory of latest result under directory specified "
            "by config_paths.FROZEN_HYPERPARAMETER_ATTACK"
        ),
    )
    parser.add_argument(
        "-b",
        "--label",
        type=str,
        action="store",
        nargs="?",
        help="String to be included as part of plot titles",
    )
    parser.add_argument(
        "-m",
        "--max_num_perts",
        type=int,
        action="store",
        nargs="?",
        help=(
            "max number of nonzero perturbation elements required of examples"
            " to summarize"
        ),
    )
    parser.add_argument(
        "-n",
        "--min_num_perts",
        type=int,
        action="store",
        nargs="?",
        help=(
            "min number of nonzero perturbation elements required of examples"
            " to summarize"
        ),
    )
    parser.add_argument(
        "-q",
        "--seq_length",
        type=int,
        action="store",
        nargs="?",
        help=(
            "Input sequence length of results to analyze. Defaults to "
            "config_settings.ATTACK_ANALYSIS_DEFAULT_SEQ_LENGTH which is "
            "typically the same as max observation window used during data "
            "preprocessing."
        ),
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        action="store",
        nargs="?",
        help=(
            "Directory where plot figures will be saved. If not specified, a "
            "new directory with timestamp in name will be created under "
            "data/attack_analysis"
        ),
    )
    parser.add_argument(
        "-s",
        "--single_histograms",
        action="append",
        nargs="+",
        help=(
            "Information for histograms to be plotted separately from "
            "grid of histograms. Useful when want to re-plot a histogram "
            "using different axis or bin settings than those used for it in "
            "the grid. Must be of form:\n"
            "grid_index[0] grid_index[1] num_bins x_min x_max\n"
            "grid_index are the coordinates of the plot within the grid that "
            "contains the data to be replotted. num_bins is the number of "
            "bins to use in the histogram. x_min & x_max specify the range "
            "of bin values"
        ),
    )

    args_namespace = parser.parse_args()
    main(**args_namespace.__dict__)
