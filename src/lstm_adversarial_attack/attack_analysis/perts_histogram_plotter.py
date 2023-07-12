from dataclasses import dataclass
from functools import cached_property
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lstm_adversarial_attack.attack_analysis.attack_analysis as ata


@dataclass
class HistLegendInfo:
    loc: str
    bbox_to_anchor: tuple[float, float]
    ncol: int


@dataclass
class HistogramInfo:
    dfs: tuple[pd.DataFrame, ...]
    df_labels: tuple[str, ...]
    data_col_name: str
    title: str
    x_label: str
    y_label: str
    default_num_bins: int
    default_x_min: int
    default_x_max: int

    @cached_property
    def data_series(self) -> tuple[pd.Series, ...]:
        return tuple([df[self.data_col_name] for df in self.dfs])

    def plot(
        self,
        ax: plt.Axes,
        num_bins: int = None,
        x_min: int | float = None,
        x_max: int | float = None,
        transparency: float = 0.5,
    ):
        if num_bins is None:
            num_bins = self.default_num_bins
        if x_min is None:
            x_min = self.default_x_min
        if x_max is None:
            x_max = self.default_x_max

        for idx, series in enumerate(self.data_series):
            counts_array, bins_array = np.histogram(
                a=series, bins=num_bins, range=(x_min, x_max)
            )
            ax.hist(
                x=bins_array[:-1],
                bins=bins_array,
                alpha=transparency,
                weights=counts_array,
                label=self.df_labels[idx],
            )


@dataclass
class HistogramPlotterFixedSettings:
    df_labels: tuple[str, ...] = ("First examples", "Best examples")
    num_plot_rows: int = 2
    num_plot_cols: int = 3
    fig_size: tuple[int, int] = (10, 7)
    ylabels: tuple[str, ...] = (
        "Counts from 0 \u2192 1 Attacks",
        "Counts from 1 \u2192 0 Attacks",
    )
    xlabels: tuple[str, ...] = (
        "# Non-Zero Elements",
        "Perturbation Element Magnitude",
        "Perturbation Element Magnitude",
    )

    default_num_bins: tuple[int, ...] = (912, 50, 50)
    default_x_min_vals: tuple[int, ...] = (0, 0, 0)
    default_x_max_vals: tuple[int, ...] = (912, 1, 1)
    data_col_names: tuple[str, ...] = (
        "num_perts",
        "pert_mean_nonzero_abs",
        "pert_max_abs",
    )
    plot_titles: tuple[str, ...] = (
        "Non-zero Perturbation\nElements",
        "Mean Non-zero Perturbation\nElement Magnitude",
        "Max Perturbation\nElement Magnitudes",
    )
    subplot_left_adjust: float = 0.1
    subplot_right_adjust: float = 0.9
    subplot_top_adjust: float = 0.78
    subplot_bottom_adjust: float = 0.17
    title_x_position: float = 0.05
    title_y_position: float = 0.93
    title_fontsize: int = 18
    title_horizontal_alignment: float = "left"
    hspace: float = 0.05
    wspace: float = 0.3


class HistogramPlotter:
    def __init__(
        self,
        title: str,
        perts_dfs: ata.StandardDataFramesForPlotter,
        cfg: HistogramPlotterFixedSettings = None,
    ):
        self.title = title
        self.perts_dfs = perts_dfs
        if cfg is None:
            self.cfg = HistogramPlotterFixedSettings()

    @property
    def _df_pairs(
        self,
    ) -> tuple[
        tuple[pd.DataFrame, pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]
    ]:
        return (
            self.perts_dfs.zero_to_one_first,
            self.perts_dfs.zero_to_one_best,
        ), (self.perts_dfs.one_to_zero_first, self.perts_dfs.one_to_zero_best)

    @property
    def hist_info_grid(self):
        my_grid = [
            [
                HistogramInfo(
                    dfs=self._df_pairs[row_idx],
                    df_labels=self.cfg.df_labels,
                    data_col_name=self.cfg.data_col_names[col_idx],
                    title=self.cfg.plot_titles[col_idx],
                    x_label=self.cfg.xlabels[col_idx],
                    y_label=self.cfg.ylabels[row_idx],
                    default_num_bins=self.cfg.default_num_bins[col_idx],
                    default_x_min=self.cfg.default_x_min_vals[col_idx],
                    default_x_max=self.cfg.default_x_max_vals[col_idx],
                )
                for col_idx in range(self.cfg.num_plot_cols)
            ]
            for row_idx in range(self.cfg.num_plot_rows)
        ]

        return my_grid

    def _set_figure_layout(self):
        fig, axes = plt.subplots(
            figsize=self.cfg.fig_size,
            nrows=self.cfg.num_plot_rows,
            ncols=self.cfg.num_plot_cols,
        )

        plt.subplots_adjust(
            left=self.cfg.subplot_left_adjust,
            right=self.cfg.subplot_right_adjust,
            top=self.cfg.subplot_top_adjust,
            bottom=self.cfg.subplot_bottom_adjust,
            hspace=self.cfg.hspace,
            wspace=self.cfg.wspace,
        )

        fig.text(
            x=self.cfg.title_x_position,
            y=self.cfg.title_y_position,
            s=self.title,
            ha=self.cfg.title_horizontal_alignment,
            rotation="horizontal",
            fontsize=self.cfg.title_fontsize,
        )

        return fig, axes

    def _decorate_subplots(self, axes: plt.Axes):
        """
        Adds titles and labels to individual subplots
        """
        for plot_row in range(self.cfg.num_plot_rows):
            axes[plot_row][0].set_ylabel(self.cfg.ylabels[plot_row])

            # if not bottom row, remove x-axis ticklabels
            if plot_row != self.cfg.num_plot_rows - 1:
                for plot_col in range(self.cfg.num_plot_cols):
                    axes[plot_row][plot_col].set_xticklabels([])

            # top row subplots gets titles
            if plot_row == 0:
                for plot_col in range(self.cfg.num_plot_cols):
                    axes[plot_row][plot_col].set_title(
                        label=self.cfg.plot_titles[plot_col], loc="left"
                    )

            # bottom row gets xlabels
            if plot_row == self.cfg.num_plot_rows - 1:
                for plot_col in range(self.cfg.num_plot_cols):
                    axes[plot_row][plot_col].set_xlabel(
                        self.cfg.xlabels[plot_col]
                    )

    def plot_all_histograms(self):
        fig, axes = self._set_figure_layout()
        for plot_row in range(self.cfg.num_plot_rows):
            for plot_col in range(self.cfg.num_plot_cols):
                ax = axes[plot_row][plot_col]
                self.hist_info_grid[plot_row][plot_col].plot(ax=ax)

        self._decorate_subplots(axes=axes)

        plt.show()

    def plot_single_histogram(
        self,
        plot_indices: tuple[int, int],
        num_bins: int,
        x_min: int | float,
        x_max: int | float,
        title: str = None,
    ):
        if title is None:
            title = self.cfg.plot_titles[plot_indices[1]]

        fig, axes = plt.subplots(nrows=1, ncols=1)
        self.hist_info_grid[plot_indices[0]][plot_indices[1]].plot(
            ax=axes, num_bins=num_bins, x_min=x_min, x_max=x_max
        )

        axes.set_ylabel(self.cfg.ylabels[plot_indices[0]])
        axes.set_xlabel(self.cfg.xlabels[plot_indices[1]])
        axes.set_title(title, loc="left")
        plt.show()


if __name__ == "__main__":
    full_attack_results = ata.FullAttackResults.from_most_recent_attack()
    attack_condition_summaries = (
        full_attack_results.get_standard_attack_condition_summaries(
            seq_length=48,
        )
    )

    new_plotter = HistogramPlotter(
        title="Perturbation Element Histograms from Latest Attack",
        perts_dfs=attack_condition_summaries.data_for_histogram_plotter,
    )
    new_plotter.plot_all_histograms()

    new_plotter.plot_single_histogram(
        plot_indices=(0, 1),
        num_bins=100,
        x_min=0,
        x_max=0.05,
        title=(
            "Mean Non-zero Perturbation Element Magnitude\nfor 0 \u2192 1"
            " Attacks"
        ),
    )
