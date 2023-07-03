import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from enum import Enum, auto
from typing import Callable

import lstm_adversarial_attack.config_paths as cfg_paths
import lstm_adversarial_attack.resource_io as rio
import lstm_adversarial_attack.attack.attack_results_analyzer as ara


# @dataclass
# class PerturbationHistogramInfo:
#     title: str
#     subtitle: str
#     source_dfs: list[list[pd.DataFrame]]
#     data_col_names: tuple[str] = (
#         "num_perts",
#         "pert_mean_nonzero_abs",
#         "pert_max_abs",
#     )
#     num_bins: tuple[int] = (30, 50, 50)
#     plot_ranges: tuple[tuple[int | float]] = (
#         (0, 30),
#         (0, 1.0),
#         (0, 1.0),
#     )
#     subplot_col_titles: tuple[str] = (
#         "Number of non-zero\nperturbation elements",
#         "Mean of magnitude of\nnon-zero perturbations",
#         "Max magnitude of\nnon-zero perturbations",
#     )
#     subplot_xlabels: tuple[str] = (
#         "# non-zero elements",
#         "Mean perturbation magnitude",
#         "Max perturbation magnitude",
#     )
#     subplot_row_titles: tuple[str] = (
#         "0 \u2192 1 attack counts",
#         "1 \u2192 0 attack counts",
#     )
#
#     def __post_init__(self):
#         assert len(self.source_dfs) == len(self.subplot_row_titles)
#         assert (
#             len(self.data_col_names)
#             == len(self.num_bins)
#             == len(self.plot_ranges)
#             == len(self.subplot_col_titles)
#             == len(self.subplot_xlabels)
#         )


class DataPairDisplayType(Enum):
    OVERLAY = auto()
    DELTA = auto()


class PerturbationHistogramPlotter:
    def __init__(
        self,
        pert_summary_dfs: list[list[pd.DataFrame]],
        title: str,
        subtitle: str,
        data_col_names: tuple[str] = (
            "num_perts",
            "pert_mean_nonzero_abs",
            "pert_max_abs",
            # "epoch_found",
            # "loss",
        ),
        data_pair_display_types: tuple[DataPairDisplayType] = (
            DataPairDisplayType.OVERLAY,
            DataPairDisplayType.OVERLAY,
            DataPairDisplayType.OVERLAY,
            # DataPairDisplayType.DELTA,
            # DataPairDisplayType.OVERLAY,
        ),
        histogram_num_bins: tuple[int] = (30, 50, 50),
        histogram_plot_ranges: tuple[tuple[int | float]] = (
            (0, 30),
            (0, 1.),
            (0, 1.),
            # (0, 100),
            # (-1.2, 0.0),
        ),
        subplot_col_titles: tuple[str] = (
            "Number of non-zero\nperturbation elements",
            "Mean of magnitude of\nnon-zero perturbations",
            "Max magnitude of\nnon-zero perturbations",
            # "Epoch when example found",
            # "Regularized adversarial loss",
        ),
        subplot_xlabels: tuple[str] = (
            "# non-zero elements",
            "Mean perturbation magnitude",
            "Max perturbation magnitude",
            # "Epoch #",
            # "Loss",
        ),
        subplot_row_titles: tuple[str] = (
            "0 \u2192 1 attack counts",
            "1 \u2192 0 attack counts",
        ),
        fig_size: tuple[int, int] = (10, 7),
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

    @classmethod
    def from_standard_attack_analyses(
        cls,
        attack_analyses: ara.StandardAttackAnalyses,
        title: str,
        subtitle: str,
        histogram_num_bins: tuple[int] = (30, 50, 50),
        histogram_plot_ranges: tuple[tuple[int | float]] = (
                (0, 30),
                (0, 1.),
                (0, 1.),
        ),
    ):
        return cls(
            pert_summary_dfs=[
                [
                    attack_analyses.zero_to_one_first.filtered_examples_df,
                    attack_analyses.zero_to_one_best.filtered_examples_df,
                ],
                [
                    attack_analyses.one_to_zero_first.filtered_examples_df,
                    attack_analyses.one_to_zero_best.filtered_examples_df,
                ],
            ],
            title=title,
            subtitle=subtitle,
            histogram_num_bins=histogram_num_bins,
            histogram_plot_ranges=histogram_plot_ranges,
        )

    @property
    def _display_type_histogram_dispatch(
        self,
    ) -> dict[DataPairDisplayType, Callable]:
        return {
            DataPairDisplayType.OVERLAY: self._plot_first_best_overlay,
            DataPairDisplayType.DELTA: self._plot_first_best_delta,
        }

    def _set_figure_layout(self):
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
        self._plot_histogram(
            ax=ax,
            data=best_df[col_name] - first_df[col_name],
            bins=bins,
            plot_range=plot_range,
            label=f"{col_name} delta (Lowest loss example) - (First example)",
        )

        if add_legend:
            ax.legend(loc="upper right", bbox_to_anchor=(1.3, -0.25), ncol=2)

    def plot_histograms(self):
        fig, axes = self._set_figure_layout()
        self._decorate_subplots(axes=axes)

        for plot_row in range(self.subplot_num_rows):
            for plot_col in range(self.subplot_num_cols):
                self._display_type_histogram_dispatch[
                    self.data_pair_display_types[plot_col]
                ](
                    ax=axes[plot_row][plot_col],
                    first_df=self.pert_summary_dfs[plot_row][0],
                    best_df=self.pert_summary_dfs[plot_row][1],
                    col_name=self.data_col_names[plot_col],
                    bins=self.histogram_num_bins[plot_col],
                    plot_range=self.histogram_plot_ranges[plot_col],
                    add_legend=(plot_row == self.subplot_num_rows - 1)
                    and (plot_col == self.subplot_num_cols // 2),
                )

                # self._plot_first_best_overlay(
                #     ax=axes[plot_row][plot_col],
                #     first_df=self.pert_summary_dfs[plot_row][0],
                #     best_df=self.pert_summary_dfs[plot_row][1],
                #     col_name=self.data_col_names[plot_col],
                #     bins=self.histogram_num_bins[plot_col],
                #     plot_range=self.histogram_plot_ranges[plot_col],
                #     add_legend=(plot_row == self.subplot_num_rows - 1)
                #     and (plot_col == self.subplot_num_cols // 2),
                # )

        plt.show()


if __name__ == "__main__":
    my_attack_analyses = rio.ResourceImporter().import_pickle_to_object(
        path=cfg_paths.ATTACK_OUTPUT_DIR
        / "plot_practice"
        / "attack_analyses.pickle"
    )
    plotter = PerturbationHistogramPlotter.from_standard_attack_analyses(
        attack_analyses=my_attack_analyses,
        title="Perturbation density and magnitude distributions",
        subtitle=(
            "Tuning objective: Maximize # of perturbation elements with "
            "exactly one non-zero element"
        ),
    )

    plotter.plot_histograms()
