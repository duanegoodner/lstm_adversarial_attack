from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm
import lstm_adversarial_attack.attack_analysis.attack_analysis as ata
import lstm_adversarial_attack.resource_io as rio


@dataclass
class SusceptibilityPlotterFixedSettings:
    df_titles: tuple[tuple[str, ...], ...] = (
        (
            "0 \u2192 1 Attack, First Examples",
            "0 \u2192 1 Attack, Best Examples",
        ),
        (
            "1 \u2192 0 Attack, First Examples",
            "1 \u2192 0 Attack, Best Examples",
        ),
    )
    fig_size: tuple[int, int] = (10, 7)
    fig_title_x: float = 0.5
    fig_title_y: float = 0.9
    fit_title_fontsize: int = 18
    fig_title_ha: str = "center"
    xtick_major: int = 12
    xtick_minor: int = 4
    x_axis_label: str = "Elapsed Time (hours)"
    xticklabel_fontsize: int = 9
    ytick_major: int = 2
    ytick_minor: int = 1
    y_axis_label: str = "Measurement ID"
    yticklabel_fontsize: int = 9
    ytick_label_fontsize: int = 9
    color_bar_coords: tuple[float, float, float, float] = (
        0.85,
        0.3,
        0.02,
        0.4,
    )
    color_scheme: str = "RdYlBu_r"
    subplot_num_rows: int = 2
    subplot_num_cols: int = 2
    subplot_top_adjust: float = 0.80
    subplot_bottom_adjust: float = 0.15
    subplot_left_adjust: float = 0.15
    subplot_right_adjust: float = 0.80
    subplot_hspace: float = 0.5
    subplot_wspace: float = 0.35


FIXED_SETTINGS = SusceptibilityPlotterFixedSettings()


class SusceptibilityPlotter:
    """
    Plots heatmaps of susceptibility metrics for perturbations causing
    adversarial examples
    """

    def __init__(
        self,
        susceptibility_dfs: ata.StandardDataFramesForPlotter,
        main_plot_title: str,
    ):
        self.susceptibility_dfs = susceptibility_dfs
        self.main_plot_title = main_plot_title
        self.measurement_names = (
            self.susceptibility_dfs.zero_to_one_first.columns
        )
        self.yticks_labels = self.susceptibility_dfs.zero_to_one_first.columns
        self.yticks_positions = np.arange(len(self.yticks_labels) + 0.5)

    @property
    def _dataframe_grid(
        self,
    ) -> tuple[
        tuple[pd.DataFrame, pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]
    ]:
        return (
            (
                self.susceptibility_dfs.zero_to_one_first,
                self.susceptibility_dfs.zero_to_one_best,
            ),
            (
                self.susceptibility_dfs.one_to_zero_first,
                self.susceptibility_dfs.one_to_zero_best,
            ),
        )

    def _set_figure_layout(self):
        """
        Sets overall layout of figue (subplots, and figure labels)
        """
        fig, axes = plt.subplots(
            nrows=FIXED_SETTINGS.subplot_num_rows,
            ncols=FIXED_SETTINGS.subplot_num_cols,
            figsize=FIXED_SETTINGS.fig_size,
        )

        plt.subplots_adjust(
            left=FIXED_SETTINGS.subplot_left_adjust,
            right=FIXED_SETTINGS.subplot_right_adjust,
            top=FIXED_SETTINGS.subplot_top_adjust,
            bottom=FIXED_SETTINGS.subplot_bottom_adjust,
            hspace=FIXED_SETTINGS.subplot_hspace,
            wspace=FIXED_SETTINGS.subplot_wspace,
        )

        fig.text(
            FIXED_SETTINGS.fig_title_x,
            FIXED_SETTINGS.fig_title_y,
            self.main_plot_title,
            ha=FIXED_SETTINGS.fig_title_ha,
            rotation="horizontal",
            fontsize=FIXED_SETTINGS.fit_title_fontsize,
        )

        colorbar_axes = fig.add_axes(FIXED_SETTINGS.color_bar_coords)

        return fig, axes, colorbar_axes

    @staticmethod
    def _decorate_subplots(
        axes: plt.Axes,
    ):
        """
        Adds titles and labels to individual subplots
        """
        # x-axis tick marks and title
        for plot_row in range(FIXED_SETTINGS.subplot_num_rows):
            for plot_col in range(FIXED_SETTINGS.subplot_num_cols):
                ax = axes[plot_row][plot_col]

                # x-axis settings
                ax.xaxis.set_major_locator(
                    ticker.IndexLocator(FIXED_SETTINGS.xtick_major, 0)
                )
                ax.xaxis.set_major_formatter("{x:.0f}")
                ax.xaxis.set_minor_locator(
                    ticker.IndexLocator(FIXED_SETTINGS.xtick_minor, 0)
                )
                ax.tick_params(
                    axis="x", labelsize=FIXED_SETTINGS.xticklabel_fontsize
                )
                ax.set_xlabel(FIXED_SETTINGS.x_axis_label)

                # y-axis settings
                ax.yaxis.set_major_locator(
                    ticker.IndexLocator(FIXED_SETTINGS.ytick_major, 0.5)
                )
                ax.yaxis.set_major_formatter("{x:.0f}")
                ax.yaxis.set_minor_locator(
                    ticker.IndexLocator(FIXED_SETTINGS.ytick_minor, 0.5)
                )
                ax.tick_params(
                    axis="y", labelsize=FIXED_SETTINGS.yticklabel_fontsize
                )
                ax.set_ylabel(FIXED_SETTINGS.y_axis_label)

                # subplot title
                ax.set_title(
                    f"{FIXED_SETTINGS.df_titles[plot_row][plot_col]}",
                    x=0.45,
                    y=1.0,
                )

    @staticmethod
    def decorate_heatmap(heatmap: sns.heatmap, source_df: pd.DataFrame):
        heatmap.axhline(y=0, color="k", linewidth=2)
        heatmap.axhline(y=len(source_df.columns), color="k", linewidth=2)
        heatmap.axvline(x=0, color="k", linewidth=2)
        heatmap.axvline(x=source_df.index.max() + 1, color="k", linewidth=2)

    def plot_susceptibilities(
        self,
        color_bar_title: str = "",
    ) -> plt.Figure:
        """
        Generates all heatmap susceptibility plots in figure
        """
        fig, axes, colorbar_axes = self._set_figure_layout()

        for plot_row in range(FIXED_SETTINGS.subplot_num_rows):
            for plot_col in range(FIXED_SETTINGS.subplot_num_cols):
                cur_df = self._dataframe_grid[plot_row][plot_col]
                cur_plot = sns.heatmap(
                    data=cur_df.T,
                    ax=axes[plot_row, plot_col],
                    cmap=FIXED_SETTINGS.color_scheme,
                    cbar=(plot_row == FIXED_SETTINGS.subplot_num_rows - 1)
                    and (plot_col == FIXED_SETTINGS.subplot_num_cols - 1),
                    cbar_ax=colorbar_axes,
                    norm=LogNorm(),
                    edgecolor="black",
                )

                self.decorate_heatmap(heatmap=cur_plot, source_df=cur_df)

                if (plot_row == FIXED_SETTINGS.subplot_num_rows - 1) and (
                    plot_col == FIXED_SETTINGS.subplot_num_cols - 1
                ):
                    cur_plot.collections[0].colorbar.set_label(color_bar_title)

        self._decorate_subplots(
            axes=axes,
        )

        plt.show()

        return fig


def plot_metric_maps(
    seq_length: int,
    metric: str,
    plot_title: str,
    colorbar_title: str,
    full_attack_results: ata.FullAttackResults = None,
):
    if full_attack_results is None:
        full_attack_results = ata.FullAttackResults.from_most_recent_attack()
    attack_condition_summaries = (
        full_attack_results.get_standard_attack_condition_summaries(
            seq_length=seq_length,
        )
    )
    plotter = SusceptibilityPlotter(
        susceptibility_dfs=attack_condition_summaries.data_for_susceptibility_plotter(
            metric=metric,
        ),
        main_plot_title=plot_title,
    )

    susceptibility_fig = plotter.plot_susceptibilities(
        color_bar_title=colorbar_title
    )

    return susceptibility_fig


if __name__ == "__main__":
    gpp_ij_figure = plot_metric_maps(
        seq_length=48,
        metric="gpp_ij",
        plot_title="Perturbation Probability",
        colorbar_title="Perturbation Probability",
    )
    ganzp_ij_figure = plot_metric_maps(
        seq_length=48,
        metric="ganzp_ij",
        plot_title="Mean Magnitude of Non-zero Perturbation Elements",
        colorbar_title="Perturbation Element Magnitude",
    )
    sensitivity_ij_figure = plot_metric_maps(
        seq_length=48,
        metric="sensitivity_ij",
        plot_title="Perturbation Sensitivity",
        colorbar_title="Perturbation Sensitivity",
    )
