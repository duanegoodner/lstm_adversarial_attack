import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.attack_analysis.attack_analysis as ata
import lstm_adversarial_attack.attack_analysis.perts_histogram_plotter as hpl
import lstm_adversarial_attack.attack_analysis.susceptibility_plotter as ssp


if __name__ == "__main__":
    newest_attack_results = ata.FullAttackResults.from_most_recent_attack()
    attack_condition_summaries = (
        newest_attack_results.get_standard_attack_condition_summaries(
            seq_length=48,
        )
    )
    histogram_plotter = hpl.HistogramPlotter(
        title="Perturbation Element Histograms",
        perts_dfs=attack_condition_summaries.data_for_histogram_plotter,
    )
    histogram_plotter.plot_all_histograms()
    histogram_plotter.plot_single_histogram(
        plot_indices=(0, 1),
        num_bins=100,
        x_min=0,
        x_max=0.05,
        title=(
            "Mean Non-zero Perturbation Element Magnitude\nfor 0 \u2192 1"
            " Attacks"
        ),
    )
    histogram_plotter.plot_single_histogram(
        plot_indices=(1, 1),
        num_bins=100,
        x_min=0,
        x_max=0.05,
        title=(
            "Mean Non-zero Perturbation Element Magnitude\nfor 1 \u2192 0"
            " Attacks"
        ),
    )

    ssp.plot_metric_maps(
        seq_length=48,
        metric="gpp_ij",
        plot_title="Perturbation Probability",
        colorbar_title="Perturbation Probability",
    )
    ssp.plot_metric_maps(
        seq_length=48,
        metric="ganzp_ij",
        plot_title="Mean Magnitude of Non-zero Perturbation Elements",
        colorbar_title="Perturbation Element Magnitude",
    )
    ssp.plot_metric_maps(
        seq_length=48,
        metric="sensitivity_ij",
        plot_title="Perturbation Sensitivity",
        colorbar_title="Perturbation Sensitivity",
    )

    # aad.plot_latest_result()
