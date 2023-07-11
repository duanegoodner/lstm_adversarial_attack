import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum, auto


class ExampleType(Enum):
    FIRST = auto
    BEST = auto


@dataclass
class PerturbationHistogramPlotter:
    title: str
    dfs: tuple[tuple[pd.DataFrame, ...], ...]
    data_cols_to_plot: tuple[str, ...] = (
            "num_perts",
            "pert_mean_nonzero_abs",
            "pert_max_abs",
        )
    bin_counts: tuple[int, ...] = (912, 50, 50),
    value_ranges: tuple[tuple[int | float, ...], ...] = (
            (0, 912),
            (0, 1.0),
            (0, 1.0),
        ),
    # grid_row_titles: tuple[str] = (
    #         "0 \u2192 1 attack counts",
    #         "1 \u2192 0 attack counts",
    #     ),
    grid_col_titles: tuple[str] = (
            "Number of non-zero\nperturbation elements",
            "Mean of magnitude of\nnon-zero perturbations",
            "Max magnitude of\nnon-zero perturbations",
        ),
    x_labels: tuple[str] = (
            "# non-zero elements",
            "Mean perturbation magnitude",
            "Max perturbation magnitude",
        )
    y_labels: tuple[str] = (
            "0 \u2192 1 attack counts",
            "1 \u2192 0 attack counts",
        ),
    fig_size: tuple[int, int] = (10, 8)
    subplot_left_adjust: float = 0.1
    subplot_right_adjust: float = 0.9
    subplot_top_adjust: float = 0.78
    subplot_bottom_adjust: float = 0.17
    subplot_hspace: float = 0.05
    subplot_wspace: float = 0.3
    subplot_title_x_position: float = 0.45
    subplot_title_y_position: float = 1.0
    title_x_position: float = 0.05
    title_y_position: float = 0.93
    title_fontsize: int = 18
    title_horizontal_alignment: float = "left"
    histogram_bar_transparency: float = 0.5

    @property
    def grid_col_items_match(self) -> bool:
        return (
            len(self.data_cols_to_plot)
            == len(self.bin_counts)
            == len(self.value_ranges)
            == len(self.grid_col_titles)
            == len(self.x_labels)
        )

    def __post_init__(self):
        assert self.grid_col_items_match

    def _plot_histogram_data(
        self,
        ax: plt.Axes,
        data: pd.Series,
        bins: int,
        label: str,
        plot_range: tuple[int, int]):

        data_counts, data_bins = np.histogram(
            a=data,
            bins=bins,
            range=plot_range
        )
        ax.hist(
            data_bins[:-1],
            data_bins,
            alpha=self.histogram_bar_transparency,
            weights=data_counts,
            label=label,
        )

    def _plot_subplot(
        self,
        grid_index: tuple[int, int],
        example_types: tuple[ExampleType],
        bins: int,
        plot_range: tuple[int, int] = None
    ):


    def _plot_histogram(
        self,
        grid_index: tuple[int, int],
        example_types: tuple[ExampleType, ...],
        ax: plt.Axes,
        data: pd.Series,
        bins: int,
        label: str,
        plot_range: tuple[int, int] = None
    ):
        counts, bins = np.histogram(a=data, bins=bins, range=plot_range)
        ax.hist(
            bins[:-1],
            bins,
            alpha=self.histogram_bar_transparency,
            weights=counts,
            label=label,
        )





