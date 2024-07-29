import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.attack_analysis.all_results_plotter as arp
import lstm_adversarial_attack.path_searches as ps
from lstm_adversarial_attack.config import CONFIG_READER


def main(
    # attack_result_path: str = None,
    attack_id: str,
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
    :param attack_id: ID of attack session to use as source of data for plots.
    Defaults to most recently created session.
    # TODO currently not using label. Either start using it, or remove it.
    :param label: String to be included as part of plot titles
    :param seq_length: "Input sequence length of results to analyze. Defaults
    to config_settings.ATTACK_ANALYSIS_DEFAULT_SEQ_LENGTH which is typically
    the same as max observation window used during data preprocessing."
    :param min_num_perts: min number of nonzero perturbation elements required
    of example_data to summarize"
    :param max_num_perts: max number of nonzero perturbation elements required
    of example_data to summarize
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

    attack_analysis_id = "".join(
        char for char in str(datetime.now()) if char.isdigit()
    )

    attack_results_root = Path(
        CONFIG_READER.read_path("attack.attack_driver.output_dir")
    )

    if attack_id is None:
        attack_id = ps.get_latest_sequential_child_dirname(
            root_dir=attack_results_root
        )

    # if attack_result_path is not None:
    #     attack_result_path = Path(attack_result_path)
    if seq_length is None:
        seq_length = CONFIG_READER.get_config_value(
            "attack.analysis.default_seq_length"
        )
    if single_histograms is not None:
        single_histograms_info = [
            arp.SingleHistogramInfo(entry) for entry in single_histograms
        ]
    else:
        single_histograms_info = None

    plotter = arp.AllResultsPlotter(
        attack_id=attack_id,
        attack_analysis_id=attack_analysis_id,
        # attack_result_path=attack_result_path,
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
        "--attack_id",
        type=str,
        action="store",
        nargs="?",
        help=(
            "ID of attack session to use for plot data. Defaults to most "
            "recently created attack session"
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
            "max number of nonzero perturbation elements required of example_data"
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
            "min number of nonzero perturbation elements required of example_data"
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
