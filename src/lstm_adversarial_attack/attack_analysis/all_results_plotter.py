from pathlib import Path

import matplotlib.pyplot as plt

import lstm_adversarial_attack.attack_analysis.attack_analysis as ata
import lstm_adversarial_attack.attack_analysis.discovery_epoch_plotter as dep
import lstm_adversarial_attack.attack_analysis.perts_histogram_plotter as php
import lstm_adversarial_attack.attack_analysis.susceptibility_plotter as ssp
import lstm_adversarial_attack.config_paths as cfg_paths
import lstm_adversarial_attack.data_provenance as dpr
import lstm_adversarial_attack.path_searches as ps
import lstm_adversarial_attack.resource_io as rio
from lstm_adversarial_attack.config import CONFIG_READER


class SingleHistogramInfo:
    def __init__(self, command_info: list):
        self.plot_indices = (int(command_info[0]), int(command_info[1]))
        self.num_bins = int(command_info[2])
        self.x_min = float(command_info[3])
        self.x_max = float(command_info[4])


class AllResultsPlotter(dpr.HasDataProvenance):
    def __init__(
        self,
        attack_result_path: Path = None,
        seq_length: int = CONFIG_READER.get_config_value("attack.analysis.default_seq_length"),
        min_num_perts: int = None,
        max_num_perts: int = None,
        label: str = None,
        output_dir: Path = None,
        single_histograms_info: list[SingleHistogramInfo] = None,
        save_output: bool = True,
    ):
        if attack_result_path is None:
            attack_result_path = ps.latest_modified_file_with_name_condition(
                component_string="attack_result.pickle",
                root_dir=cfg_paths.FROZEN_HYPERPARAMETER_ATTACK,
                comparison_type=ps.StringComparisonType.SUFFIX,
            )
        self.attack_result_path = attack_result_path
        self.seq_length = seq_length
        self.min_num_perts = min_num_perts
        self.max_num_perts = max_num_perts
        self.label = label
        self.single_histograms_info = single_histograms_info
        self.save_output = save_output
        if self.save_output and output_dir is None:
            output_dir = rio.create_timestamped_dir(
                parent_path=cfg_paths.ATTACK_ANALYSIS_DIR
            )
        self.output_dir = output_dir

        self.full_attack_results = (
            ata.FullAttackResults.from_trainer_result_path(
                trainer_result_path=self.attack_result_path
            )
        )
        self.attack_condition_summaries = (
            self.full_attack_results.get_standard_attack_condition_summaries(
                seq_length=self.seq_length,
                min_num_perts=min_num_perts,
                max_num_perts=max_num_perts,
            )
        )
        self.histogram_plotter = php.HistogramPlotter(
            perts_dfs=self.attack_condition_summaries.data_for_histogram_plotter,
        )
        self.discovery_epoch_plotter = dep.DiscoveryEpochCDFPlotter(
            # title="Adversarial Examples Discovery Epoch CDFs",
            xy_dfs=self.attack_condition_summaries.data_for_epochfound_cdf_plotter,
            cfg=dep.DiscoveryEpochCDFPlotterSettings(),
        )
        self.gpp_ij_plotter = ssp.SusceptibilityPlotter(
            susceptibility_dfs=self.attack_condition_summaries.data_for_susceptibility_plotter(
                metric="gpp_ij"
            ),
            main_plot_title="Perturbation Probability",
            colorbar_title="Perturbation Probability",
        )
        self.ganzp_ij_plotter = ssp.SusceptibilityPlotter(
            susceptibility_dfs=self.attack_condition_summaries.data_for_susceptibility_plotter(
                metric="ganzp_ij",
            ),
            main_plot_title="Mean Magnitude of Non-zero Perturbation Elements",
            colorbar_title="Perturbation Element Magnitude",
        )
        self.sensitivity_ij_plotter = ssp.SusceptibilityPlotter(
            susceptibility_dfs=self.attack_condition_summaries.data_for_susceptibility_plotter(
                metric="sensitivity_ij",
            ),
            main_plot_title="Perturbation Sensitivity",
            colorbar_title="Perturbation Sensitivity",
        )

        self.export(
            filename="all_results_plotter_dict.pickle",
            provenance_only=True,
            provenance_text_file=True,
        )

    @property
    def provenance_info(self) -> dpr.ProvenanceInfo:
        return dpr.ProvenanceInfo(
            previous_info=(
                self.attack_result_path.parent / "provenance.pickle"
                if self.attack_result_path is not None
                else None
            ),
            category_name="result_plotter",
            new_items={
                "attack_result_path": self.attack_result_path,
                "seq_length": self.seq_length,
                "min_num_perts": self.min_num_perts,
                "max_num_perts": self.max_num_perts,
            },
            output_dir=self.output_dir,
        )

    def save_figure(self, fig: plt.Figure, label: str):
        if self.save_output:
            output_path = rio.create_timestamped_filepath(
                parent_path=self.output_dir,
                file_extension="png",
                suffix=f"_{label}",
            )
            fig.savefig(fname=str(output_path))

    def plot_all_histograms(self, fig_title: str = None):
        hist_grid_fig = self.histogram_plotter.plot_all_histograms(
            fig_title=fig_title
        )
        self.save_figure(fig=hist_grid_fig, label="histogram_grid")

    def plot_single_histogram(
        self,
        plot_indices: tuple[int, int],
        num_bins: int,
        x_min: float | int,
        x_max: float | int,
        fig_title: str = None,
        label: str = "single_histogram",
    ):
        single_hist_fig = self.histogram_plotter.plot_single_histogram(
            plot_indices=plot_indices,
            num_bins=num_bins,
            x_min=x_min,
            x_max=x_max,
            fig_title=fig_title,
        )
        self.save_figure(fig=single_hist_fig, label=label)

    def plot_discovery_epoch_cdfs(self, fig_title: str = None):
        discovery_epoch_fig = self.discovery_epoch_plotter.plot_all_cdfs()
        self.save_figure(fig=discovery_epoch_fig, label="discovery_epochs")

    def plot_gpp_ij(self):
        gpp_ij_fig = self.gpp_ij_plotter.plot_susceptibilities()
        self.save_figure(fig=gpp_ij_fig, label="gpp_ij")

    def plot_ganzp_ij(self):
        ganzp_ij_fig = self.ganzp_ij_plotter.plot_susceptibilities()
        self.save_figure(fig=ganzp_ij_fig, label="ganzp_ij")

    def plot_sensitivity_ij(self):
        sensitivity_ij_fig = (
            self.sensitivity_ij_plotter.plot_susceptibilities()
        )
        self.save_figure(fig=sensitivity_ij_fig, label="sensitivity_ij")

    def plot_all(self):
        self.plot_all_histograms()
        if self.single_histograms_info:
            for entry in self.single_histograms_info:
                self.plot_single_histogram(**entry.__dict__)
        self.plot_discovery_epoch_cdfs()
        self.plot_gpp_ij()
        self.plot_ganzp_ij()
        self.plot_sensitivity_ij()


if __name__ == "__main__":
    plotter = AllResultsPlotter()
    plotter.plot_all_histograms()
    plotter.plot_single_histogram(
        plot_indices=(0, 1),
        num_bins=100,
        x_min=0,
        x_max=0.05,
    )
    plotter.plot_discovery_epoch_cdfs()
    plotter.plot_gpp_ij()
    plotter.plot_ganzp_ij()
    plotter.plot_sensitivity_ij()
