from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import msgspec

import lstm_adversarial_attack.attack.attack_data_structs as ads
import lstm_adversarial_attack.attack_analysis.attack_analysis as ata
import lstm_adversarial_attack.attack_analysis.discovery_epoch_plotter as dep
import lstm_adversarial_attack.attack_analysis.perts_histogram_plotter as php
import lstm_adversarial_attack.attack_analysis.susceptibility_plotter as ssp
import lstm_adversarial_attack.utils.msgspec_io as mio
from lstm_adversarial_attack.config import CONFIG_READER, PATH_CONFIG_READER


@dataclass
class SingleHistogramInfo:
    plot_indices: tuple[int, int]
    num_bins: int
    x_min: float
    x_max: float
    filename: str

class AllResultsPlotterSummary(msgspec.Struct):
    preprocess_id: str
    model_tuning_id: str
    cv_training_id: str
    attack_tuning_id: str
    attack_id: str
    attack_analysis_id: str
    min_num_perts: int
    max_num_perts: int
    single_histograms_info: list[SingleHistogramInfo]


class AllResultsPlotterSummaryIO(mio.MsgspecIO):
    def __init__(self):
        super().__init__(msgspec_struct_type=AllResultsPlotterSummary)


ALL_RESULTS_PLOTTER_SUMMARY_IO = AllResultsPlotterSummaryIO()


class AllResultsPlotter:
    def __init__(
        self,
        attack_id: str,
        attack_analysis_id: str,
        min_num_perts: int = None,
        max_num_perts: int = None,
        label: str = None,
        single_histograms_info: list[SingleHistogramInfo] = None,
    ):
        self.attack_id = attack_id
        self.attack_analysis_id = attack_analysis_id
        self.min_num_perts = min_num_perts
        self.max_num_perts = max_num_perts
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seq_length = CONFIG_READER.get_config_value(
            "attack.analysis.default_seq_length"
        )
        self.label = label
        self.single_histograms_info = single_histograms_info
        self.full_attack_results = (
            ata.FullAttackResults.from_attack_trainer_result_path(
                attack_trainer_result_path=self.attack_result_path
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

    @property
    def attack_result_path(self) -> Path:
        attack_results_root = Path(
            PATH_CONFIG_READER.read_path("attack.attack_driver.output_dir")
        )
        return (
            attack_results_root
            / self.attack_id
            / f"final_attack_result_{self.attack_id}.json"
        )

    @property
    def attack_driver_summary(self) -> ads.AttackDriverSummary:
        attack_results_root = Path(
            PATH_CONFIG_READER.read_path("attack.attack_driver.output_dir")
        )

        return ads.ATTACK_DRIVER_SUMMARY_IO.import_to_struct(
            path=attack_results_root
            / self.attack_id
            / f"attack_driver_summary_{self.attack_id}.json"
        )

    @property
    def output_dir(self) -> Path:
        analysis_results_root = Path(
            PATH_CONFIG_READER.read_path("attack_analysis.output_dir")
        )
        return analysis_results_root / f"{self.attack_analysis_id}"

    @property
    def summary(self) -> AllResultsPlotterSummary:
        return AllResultsPlotterSummary(
            preprocess_id=self.attack_driver_summary.preprocess_id,
            model_tuning_id=self.attack_driver_summary.model_tuning_id,
            cv_training_id=self.attack_driver_summary.cv_training_id,
            attack_tuning_id=self.attack_driver_summary.attack_tuning_id,
            attack_id=self.attack_id,
            attack_analysis_id=self.attack_analysis_id,
            min_num_perts=self.min_num_perts,
            max_num_perts=self.max_num_perts,
            single_histograms_info=self.single_histograms_info
        )

    def save_figure(self, fig: plt.Figure, label: str):
        # if self.save_output:
        output_path = (
            self.output_dir / f"{self.attack_analysis_id}_{label}.png"
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
        filename: str = "single_histogram",
    ):
        single_hist_fig = self.histogram_plotter.plot_single_histogram(
            plot_indices=plot_indices,
            num_bins=num_bins,
            x_min=x_min,
            x_max=x_max,
            fig_title=fig_title,
        )
        self.save_figure(fig=single_hist_fig, label=filename)

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
        ALL_RESULTS_PLOTTER_SUMMARY_IO.export(
            obj=self.summary,
            path=self.output_dir
            / f"{self.attack_analysis_id}_all_results_plotter_summary.json",
        )

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
