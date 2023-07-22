import argparse
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.attack_analysis.attack_analysis as ata
import lstm_adversarial_attack.attack_analysis.perts_histogram_plotter as php
import lstm_adversarial_attack.attack_analysis.susceptibility_plotter as ssp
import lstm_adversarial_attack.config_paths as cfg_paths
import lstm_adversarial_attack.config_settings as cfg_settings
import lstm_adversarial_attack.path_searches as ps


def main(
    attack_result_path: Path = None,
    seq_length: int = cfg_settings.ATTACK_ANALYSIS_DEFAULT_SEQ_LENGTH,
    title: str = "Perturbation Element Histograms",
):
    if attack_result_path is None:
        attack_result_path = ps.latest_modified_file_with_name_condition(
            component_string="attack_result.pickle",
            root_dir=cfg_paths.FROZEN_HYPERPARAMETER_ATTACK,
            comparison_type=ps.StringComparisonType.SUFFIX,
        )

    print(
        f"Generating histograms from attack result in:\n{attack_result_path}"
    )

    full_attack_results = ata.FullAttackResults.from_trainer_result_path(
        trainer_result_path=attack_result_path
    )

    attack_condition_summaries = (
        full_attack_results.get_standard_attack_condition_summaries(
            seq_length=seq_length
        )
    )

    hist_plotter = php.HistogramPlotter(
        title=title,
        perts_dfs=attack_condition_summaries.data_for_histogram_plotter,
    )

    hist_grid = hist_plotter.plot_all_histograms()

    susceptibility_plotter = ssp.SusceptibilityPlotter(
        
    )


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--single_hist",
    #     "-s",
    #     nargs="*",
    #     action="append"
    # )
    # args_namespace = parser.parse_args()
    # print(args_namespace.single_hist)
    main()
