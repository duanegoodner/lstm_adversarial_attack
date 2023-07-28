from dataclasses import dataclass
from functools import cached_property
import matplotlib.pyplot as plt
import pandas as pd
import lstm_adversarial_attack.attack_analysis.attack_analysis as ata


@dataclass
class DiscoveryEpochCDFPlotterSettings:
    data_cols: tuple[str, str] = (
        "first_example_cumsum",
        "best_example_cumsum",
    )
    num_plot_rows: int = 2
    num_plot_cols: int = 1
    fig_size: tuple[float, float] = (5.5, 7.7)
    title_x_position: float = 0.12
    title_y_position: float = 0.92
    title_horizontal_alignment: str = "left"
    title_fontsize: int = 14
    xlabel_fontsize: int = 12
    ylabel_fontsize: int = 12
    xlabel: str = "Attack Epoch Number"
    ylabels: tuple[str, str] = (
        "First Examples",
        "Best Examples",
    )
    subplot_left_adjust: float = 0.2
    subplot_right_adjust: float = 0.85
    subplot_bottom_adjust: float = 0.2
    subplot_top_adjust: float = 0.85
    hspace: float = 0.1
    wspace: float = 0.1
    legend_bbox_to_anchor: tuple[float, float] = (1.04, -0.3)
    legend_ncol: int = 2
    legend_fontsize: int = 12


class DiscoveryEpochCDFPlotter:
    def __init__(
        self,
        # title: str,
        xy_dfs: ata.ZerosOnesDFPair,
        cfg: DiscoveryEpochCDFPlotterSettings,
    ):
        # self.title = title
        self.xy_dfs = xy_dfs
        self.cfg = cfg

    @property
    def _df_tuple(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self.xy_dfs.zero_to_one, self.xy_dfs.one_to_zero

    def _set_figure_layout(self, fig_title: str):
        fig, axes = plt.subplots(
            figsize=self.cfg.fig_size,
            nrows=self.cfg.num_plot_rows,
            ncols=self.cfg.num_plot_cols,
        )

        fig.text(
            x=self.cfg.title_x_position,
            y=self.cfg.title_y_position,
            s=fig_title,
            ha=self.cfg.title_horizontal_alignment,
            rotation="horizontal",
            fontsize=self.cfg.title_fontsize,
        )

        plt.subplots_adjust(
            left=self.cfg.subplot_left_adjust,
            right=self.cfg.subplot_right_adjust,
            top=self.cfg.subplot_top_adjust,
            bottom=self.cfg.subplot_bottom_adjust,
            hspace=self.cfg.hspace,
            wspace=self.cfg.wspace,
        )

        return fig, axes

    def plot_all_cdfs(self, fig_title: str = None) -> plt.Figure:
        if fig_title is None:
            fig_title = "Adversarial Examples Discovery Epoch CDFs"
        fig, axes = self._set_figure_layout(fig_title=fig_title)
        for plot_row in range(len(self._df_tuple)):
            df = self._df_tuple[plot_row]
            ax = axes[plot_row]
            ax.plot(
                self.xy_dfs.zero_to_one["epoch_found"],
                self.xy_dfs.zero_to_one[self.cfg.data_cols[plot_row]],
                linestyle="--",
                label="0 \u2192 1 Attacks",
            )
            ax.plot(
                self.xy_dfs.one_to_zero["epoch_found"],
                self.xy_dfs.one_to_zero[self.cfg.data_cols[plot_row]],
                label="1 \u2192 0 Attacks",
            )

            ax.set_ylabel(
                self.cfg.ylabels[plot_row], fontsize=self.cfg.xlabel_fontsize
            )
            if plot_row == 0:
                ax.set_xticklabels([])
            if plot_row == 1:
                ax.set_xlabel(
                    self.cfg.xlabel, fontsize=self.cfg.ylabel_fontsize
                )
                ax.legend(
                    bbox_to_anchor=self.cfg.legend_bbox_to_anchor,
                    ncol=self.cfg.legend_ncol,
                    fontsize=self.cfg.legend_fontsize
                )

        plt.show()
        return fig


if __name__ == "__main__":
    full_attack_results = ata.FullAttackResults.from_most_recent_attack()
    attack_condition_summaries = (
        full_attack_results.get_standard_attack_condition_summaries(
            seq_length=48,
        )
    )
    epochfound_plotter = DiscoveryEpochCDFPlotter(
        # title="Adversarial Examples Discovery Epoch CDFs",
        xy_dfs=attack_condition_summaries.data_for_epochfound_cdf_plotter,
        cfg=DiscoveryEpochCDFPlotterSettings(),
    )
    epochfound_plotter.plot_all_cdfs()
