import argparse
import sys
from datetime import datetime
from pathlib import Path
import ast
import re

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.attack_analysis.all_results_plotter as arp
import lstm_adversarial_attack.path_searches as ps
import lstm_adversarial_attack.session_id_generator as sig
from lstm_adversarial_attack.config import CONFIG_READER


def main(
    attack_id: str,
    label: str = None,
    min_num_perts: int = None,
    max_num_perts: int = None,
    single_histograms: list[dict] = None,
):
    """
    Generates set of plots using attack data. Plots are saved as .png. Also
    generates provenance.pickle and provenance.txt files with details of
    files and settings used to during attack hyperparamter tuning the final
    attack (using fixed set of hyperparameters).
    :param attack_id: ID of attack session to use as source of data for plots.
    Defaults to most recently created session.
    :param label: String to be included as part of plot titles
    :param min_num_perts: min number of nonzero perturbation elements required
    of example_data to summarize
    :param max_num_perts: max number of nonzero perturbation elements required
    of example_data to summarize
    :param single_histograms: List of dictionaries for histograms to be plotted separately from grid of histograms.
    :return:
    """

    attack_analysis_id = sig.generate_session_id()

    attack_results_root = Path(
        CONFIG_READER.read_path("attack.attack_driver.output_dir")
    )

    if attack_id is None:
        attack_id = ps.get_latest_sequential_child_dirname(
            root_dir=attack_results_root
        )

    if single_histograms is not None:
        single_histograms_info = [
            arp.SingleHistogramInfo(
                plot_indices=(
                    entry["grid_indices"][0],
                    entry["grid_indices"][1],
                ),
                num_bins=entry["num_bins"],
                x_min=entry["x_min"],
                x_max=entry["x_max"],
                filename=entry.get("filename"),
            )
            for entry in single_histograms
        ]
    else:
        single_histograms_info = None

    plotter = arp.AllResultsPlotter(
        attack_id=attack_id,
        attack_analysis_id=attack_analysis_id,
        min_num_perts=min_num_perts,
        max_num_perts=max_num_perts,
        label=label,
        single_histograms_info=single_histograms_info,
    )

    plotter.plot_all()


if __name__ == "__main__":

    def preprocess_single_histogram_string(s):
        # Add quotes around keys
        s = re.sub(r"(\w+):", r'"\1":', s)
        return s

    def parse_single_histogram(arg):
        try:
            preprocessed_arg = preprocess_single_histogram_string(arg)
            # Safely evaluate the string as a Python literal (a dict in this case)
            return ast.literal_eval(preprocessed_arg)
        except (SyntaxError, ValueError) as e:
            raise argparse.ArgumentTypeError(
                f"Invalid format for single_histogram: {arg}"
            ) from e

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
        "-s",
        "--single_histograms",
        type=parse_single_histogram,
        action="append",
        help=(
            "Information for histograms to be plotted separately from "
            "grid of histograms. Useful when want to re-plot a histogram "
            "using different axis or bin settings than those used for it in "
            "the grid. Must be of form:\n"
            "{grid_indices: (0, 1), num_bins: 25, x_min: 0.0, x_max: 0.25}\n"
            "grid_indices are the coordinates of the plot within the grid that "
            "contains the data to be replotted. num_bins is the number of "
            "bins to use in the histogram. x_min & x_max specify the range "
            "of bin values"
        ),
    )

    args_namespace = parser.parse_args()
    main(**args_namespace.__dict__)
