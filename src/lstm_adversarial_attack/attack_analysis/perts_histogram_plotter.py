import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from enum import Enum, auto
from typing import NamedTuple

import lstm_adversarial_attack.attack_analysis.attack_analysis as ata


class PlotLimits(NamedTuple):
    """
    Used for inset definition
    """

    x_min: float
    x_max: float
    y_min: float
    y_max: float


class PerturbationHistogramPlotter:
    """
    Plots histograms of perturbation-related info: num non-zero elements
    per example, mean perturbation element magnitude, and max perturbation
    element magnitude.
    """

    def __init__(
        self,
        pert_summary_dfs: tuple[tuple[pd.DataFrame, ...], ...],
        title: str,
        data_col_names: tuple[str, ...] = (
            "num_perts",
            "pert_mean_nonzero_abs",
            "pert_max_abs",
        ),
        histogram_num_bins: tuple[int, ...] = (912, 50, 50),
        histogram_plot_ranges: tuple[tuple[int | float, ...], ...] = (
            (0, 912),
            (0, 1.0),
            (0, 1.0),
        ),
        subplot_col_titles: tuple[str] = (
            "Number of non-zero\nperturbation elements",
            "Mean of magnitude of\nnon-zero perturbations",
            "Max magnitude of\nnon-zero perturbations",
        ),
        subplot_xlabels: tuple[str] = (
            "# non-zero elements",
            "Mean perturbation magnitude",
            "Max perturbation magnitude",
        ),
        subplot_row_titles: tuple[str] = (
            "0 \u2192 1 attack counts",
            "1 \u2192 0 attack counts",
        ),
        fig_size: tuple[int, int] = (10, 8),
        subplot_left_adjust: float = 0.1,
        subplot_right_adjust: float = 0.9,
        subplot_top_adjust: float = 0.78,
        subplot_bottom_adjust: float = 0.17,
        subplot_hspace: float = 0.05,
        subplot_wspace: float = 0.3,
        subplot_title_x_position: float = 0.45,
        subplot_title_y_position: float = 1.0,
        title_x_position: float = 0.05,
        title_y_position: float = 0.93,
        title_fontsize: int = 18,
        title_horizontal_alignment: float = "left",
        histogram_bar_transparency: float = 0.5,
    ):
        self.pert_summary_dfs = pert_summary_dfs
        self.title = title
        self.data_col_names = data_col_names
        self.histogram_num_bins = histogram_num_bins
        self.histogram_plot_ranges = histogram_plot_ranges
        self.subplot_col_titles = subplot_col_titles
        self.subplot_xlabels = subplot_xlabels
        self.subplot_row_titles = subplot_row_titles
        self.subplot_num_rows = len(self.pert_summary_dfs)
        self.subplot_num_cols = len(self.data_col_names)
        self.fig_size = fig_size
        self.subplot_left_adjust = subplot_left_adjust
        self.subplot_right_adjust = subplot_right_adjust
        self.subplot_top_adjust = subplot_top_adjust
        self.subplot_bottom_adjust = subplot_bottom_adjust
        self.subplot_hspace = subplot_hspace
        self.subplot_wspace = subplot_wspace
        self.subplot_title_x_position = subplot_title_x_position
        self.subplot_title_y_position = subplot_title_y_position
        self.title_x_position = title_x_position
        self.title_y_position = title_y_position
        self.title_fontsize = title_fontsize
        self.title_horizontal_alignment = title_horizontal_alignment
        self.histogram_bar_transparency = histogram_bar_transparency

    def _set_figure_layout(self):
        """
        Sets overall layout of figue (subplots, and figure labels)
        """
        fig, axes = plt.subplots(
            nrows=self.subplot_num_rows,
            ncols=self.subplot_num_cols,
            figsize=self.fig_size,
        )

        plt.subplots_adjust(
            left=self.subplot_left_adjust,
            right=self.subplot_right_adjust,
            top=self.subplot_top_adjust,
            bottom=self.subplot_bottom_adjust,
            hspace=self.subplot_hspace,
            wspace=self.subplot_wspace,
        )

        fig.text(
            x=self.title_x_position,
            y=self.title_y_position,
            s=self.title,
            ha=self.title_horizontal_alignment,
            rotation="horizontal",
            fontsize=self.title_fontsize,
        )

        return fig, axes

    def _decorate_subplots(self, axes: plt.Axes):
        """
        Adds titles and labels to individual subplots
        """
        for plot_row in range(self.subplot_num_rows):
            axes[plot_row][0].set_ylabel(self.subplot_row_titles[plot_row])

            # if not bottom row, remove x-axis ticklabels
            if plot_row != self.subplot_num_rows - 1:
                for plot_col in range(self.subplot_num_cols):
                    axes[plot_row][plot_col].set_xticklabels([])

            # top row subplots gets titles
            if plot_row == 0:
                for plot_col in range(self.subplot_num_cols):
                    axes[plot_row][plot_col].set_title(
                        self.subplot_col_titles[plot_col]
                    )

            # bottom row gets xlabels
            if plot_row == self.subplot_num_rows - 1:
                for plot_col in range(self.subplot_num_cols):
                    axes[plot_row][plot_col].set_xlabel(
                        self.subplot_xlabels[plot_col]
                    )

    def _plot_histogram(
        self,
        ax: plt.Axes,
        data: pd.Series,
        bins: int,
        label: str,
        plot_range: tuple[int, int] = None,
    ):
        """
        Plots a single histogram
        """
        counts, bins = np.histogram(a=data, bins=bins, range=plot_range)
        ax.hist(
            bins[:-1],
            bins,
            alpha=self.histogram_bar_transparency,
            weights=counts,
            label=label,
        )

    def _plot_first_best_overlay(
        self,
        ax: plt.Axes,
        first_df: pd.DataFrame,
        best_df: pd.DataFrame,
        col_name: str,
        bins: int,
        plot_range: tuple[int, int] = None,
        add_legend: bool = False,
    ):
        """
        Plots two histograms on single axes.
        """
        self._plot_histogram(
            ax=ax,
            data=first_df[col_name],
            bins=bins,
            plot_range=plot_range,
            label="First examples found",
        )
        self._plot_histogram(
            ax=ax,
            data=best_df[col_name],
            bins=bins,
            plot_range=plot_range,
            label="Examples with lowest loss",
        )

        if add_legend:
            ax.legend(loc="upper right", bbox_to_anchor=(1.3, -0.25), ncol=2)

    def plot_all_histograms(self):
        """
        Plots all histograms in figure
        """
        fig, axes = self._set_figure_layout()
        self._decorate_subplots(axes=axes)

        for plot_row in range(self.subplot_num_rows):
            for plot_col in range(self.subplot_num_cols):
                self._plot_first_best_overlay(
                    ax=axes[plot_row][plot_col],
                    first_df=self.pert_summary_dfs[plot_row][0],
                    best_df=self.pert_summary_dfs[plot_row][1],
                    col_name=self.data_col_names[plot_col],
                    bins=self.histogram_num_bins[plot_col],
                    plot_range=self.histogram_plot_ranges[plot_col],
                    add_legend=(plot_row == self.subplot_num_rows - 1)
                    and (plot_col == self.subplot_num_cols // 2),
                )

        plt.show()


if __name__ == "__main__":
    full_attack_results = ata.FullAttackResults.from_most_recent_attack()
    attack_condition_summaries = (
        full_attack_results.get_standard_attack_condition_summaries(
            seq_length=48,
        )
    )
    hist_plotter = PerturbationHistogramPlotter(
        pert_summary_dfs=attack_condition_summaries.data_for_histogram_plotter,
        title="Perturbation Distributions from Most Recent Attack",
    )

    hist_plotter.plot_all_histograms()

