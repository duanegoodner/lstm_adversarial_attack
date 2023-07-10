import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from enum import Enum, auto
from typing import Callable, NamedTuple

import lstm_adversarial_attack.config_paths as cfg_paths
import lstm_adversarial_attack.resource_io as rio


class DataPairDisplayType(Enum):
    OVERLAY = auto()
    DELTA = auto()


class PlotLimits(NamedTuple):
    """
    Used for inset definition
    """
    x_min: float
    x_max: float
    y_min: float
    y_max: float


class InsetSpec(NamedTuple):
    """
    Specs that define an inset
    """
    bounds: list[float]
    plot_limits: PlotLimits


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
        subtitle: str,
        data_col_names: tuple[str, ...] = (
            "num_perts",
            "pert_mean_nonzero_abs",
            "pert_max_abs",
        ),
        data_pair_display_types: tuple[DataPairDisplayType] = (
            DataPairDisplayType.OVERLAY,
            DataPairDisplayType.OVERLAY,
            DataPairDisplayType.OVERLAY,
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
        create_insets: tuple[tuple[bool, ...], ...] = (
            (False, False, False),
            (False, False, False),
        ),
        # TODO These specs are for sparse-small-max. Move to
        #  PerturbationHistogramPlotter constructor args for those results.
        inset_specs: tuple[tuple[InsetSpec | None, ...], ...] = (
            (None, None, None),
            (None, None, None),
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
        subtitle_x_offset: float = 0,
        subtitle_y_offset: float = -0.04,
        subtitle_fontsize: int = 14,
        title_horizontal_alignment: float = "left",
        histogram_bar_transparency: float = 0.5,
    ):
        self.pert_summary_dfs = pert_summary_dfs
        self.title = title
        self.subtitle = subtitle
        self.data_col_names = data_col_names
        self.data_pair_display_types = data_pair_display_types
        self.histogram_num_bins = histogram_num_bins
        self.histogram_plot_ranges = histogram_plot_ranges
        self.subplot_col_titles = subplot_col_titles
        self.subplot_xlabels = subplot_xlabels
        self.subplot_row_titles = subplot_row_titles
        self.create_insets = create_insets
        self.inset_specs = inset_specs
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
        self.subtitle_x_offset = subtitle_x_offset
        self.subtitle_y_offset = subtitle_y_offset
        self.subtitle_fontsize = subtitle_fontsize
        self.histogram_bar_transparency = histogram_bar_transparency

    @property
    def _display_type_histogram_dispatch(
        self,
    ) -> dict[DataPairDisplayType, Callable]:
        """
        Determines which type of histogram to plot for a first example / best
        example pair (overlay or delta)
        :return: dispatch dictionary
        """
        return {
            DataPairDisplayType.OVERLAY: self._plot_first_best_overlay,
            DataPairDisplayType.DELTA: self._plot_first_best_delta,
        }

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

        fig.text(
            x=self.title_x_position + self.subtitle_x_offset,
            y=self.title_y_position + self.subtitle_y_offset,
            s=self.subtitle,
            ha=self.title_horizontal_alignment,
            rotation="horizontal",
            fontsize=self.subtitle_fontsize,
        )

        return fig, axes

    def _decorate_subplots(self, axes: plt.Axes):
        """
        Adds titles and labels to individual subplots
        """
        for plot_row in range(self.subplot_num_rows):
            # first plot in row gets ylabel
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

    def _plot_first_best_delta(
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
        Plots difference between two analogous data series
        """
        self._plot_histogram(
            ax=ax,
            data=best_df[col_name] - first_df[col_name],
            bins=bins,
            plot_range=plot_range,
            label=f"{col_name} delta (Lowest loss example) - (First example)",
        )

        if add_legend:
            ax.legend(loc="upper right", bbox_to_anchor=(1.3, -0.25), ncol=2)

    def _add_inset(
        self,
        ax: plt.Axes,
        plotting_method: Callable,
        plot_row: int,
        plot_col: int,
        bounds: list[float],
        plot_limits: PlotLimits,
    ):
        """
        Adds inset to histogram
        """
        ax_inset = ax.inset_axes(bounds=bounds)
        ax_inset.set_xlim(plot_limits.x_min, plot_limits.x_max)
        ax_inset.set_ylim(plot_limits.y_min, plot_limits.y_max)
        ax_inset.set_yticklabels([])
        ax.indicate_inset_zoom(inset_ax=ax_inset, edgecolor="black")

        plotting_method(
            ax=ax_inset,
            first_df=self.pert_summary_dfs[plot_row][0],
            best_df=self.pert_summary_dfs[plot_row][1],
            col_name=self.data_col_names[plot_col],
            bins=self.histogram_num_bins[plot_col],
            plot_range=self.histogram_plot_ranges[plot_col],
            add_legend=False,
        )

    def plot_histograms(self):
        """
        Plots all histograms in figure
        """
        fig, axes = self._set_figure_layout()
        self._decorate_subplots(axes=axes)

        for plot_row in range(self.subplot_num_rows):
            for plot_col in range(self.subplot_num_cols):
                plotting_method = self._display_type_histogram_dispatch[
                    self.data_pair_display_types[plot_col]
                ]

                plotting_method(
                    ax=axes[plot_row][plot_col],
                    first_df=self.pert_summary_dfs[plot_row][0],
                    best_df=self.pert_summary_dfs[plot_row][1],
                    col_name=self.data_col_names[plot_col],
                    bins=self.histogram_num_bins[plot_col],
                    plot_range=self.histogram_plot_ranges[plot_col],
                    add_legend=(plot_row == self.subplot_num_rows - 1)
                    and (plot_col == self.subplot_num_cols // 2),
                )

                if self.create_insets[plot_row][plot_col]:
                    self._add_inset(
                        ax=axes[plot_row][plot_col],
                        plotting_method=plotting_method,
                        plot_row=plot_row,
                        plot_col=plot_col,
                        bounds=self.inset_specs[plot_row][plot_col].bounds,
                        plot_limits=self.inset_specs[plot_row][
                            plot_col
                        ].plot_limits,
                    )

        plt.show()


if __name__ == "__main__":
    my_attack_analyses = rio.ResourceImporter().import_pickle_to_object(
        path=cfg_paths.ATTACK_OUTPUT_DIR
        / "plot_practice"
        / "attack_analyses.pickle"
    )

    plotter = PerturbationHistogramPlotter(
        my_attack_analyses.df_tuple_for_histogram_plotter,
        title="Perturbation density and magnitude distributions",
        subtitle=(
            "Tuning objective: Maximize # of perturbation elements with "
            "exactly one non-zero element"
        ),
    )

    plotter.plot_histograms()
