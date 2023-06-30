import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm
import lstm_adversarial_attack.resource_io as rio
import lstm_adversarial_attack.config_paths as cfg_paths


class SusceptibilityGridPlotter:
    def __init__(
        self,
        fig_size: tuple[int, int] = (10, 10),
        subplot_num_rows: int = 4,
        color_bar_coords: tuple[float, float, float, float] = (
            0.88,
            0.3,
            0.02,
            0.4,
        ),
        share_xtick_labels: bool = True,
        share_ytick_labels: bool = False,
        xtick_major: int = 5,
        xtick_minor: int = 1,
        ytick_major: int = 5,
        ytick_minor: int = 1,
        color_scheme: str = "RdYlBu_r",
    ):
        # assert 2 <= subplot_num_rows <= 4
        self._fig_size = fig_size
        self._subplot_num_rows = subplot_num_rows
        self._subplot_num_cols = 1  # must have exactly 1 column
        self._subplot_right_adjust: float = 0.85
        self._subplot_left_adjust: float = 0.18
        self._color_bar_coords = color_bar_coords
        self._share_xtick_labels = share_xtick_labels
        self._share_ytick_labels = share_ytick_labels
        self._xtick_major = xtick_major
        self._xtick_minor = xtick_minor
        self._ytick_major = ytick_major
        self._ytick_minor = ytick_minor
        self._color_scheme = color_scheme

    def max_num_samples(self) -> int:
        return self._subplot_num_rows * self._subplot_num_cols

    def _set_subplot_layout(self):
        fig, axes = plt.subplots(
            nrows=self._subplot_num_rows,
            ncols=self._subplot_num_cols,
            sharex=self._share_xtick_labels,
            sharey=self._share_ytick_labels,
            figsize=self._fig_size,
        )
        fig.text(0.5, 0.02, "Time (hours)", ha="center")
        fig.text(0.04, 0.5, "Feature Index", va="center", rotation="vertical")
        plt.subplots_adjust(
            left=self._subplot_left_adjust, right=self._subplot_right_adjust
        )
        colorbar_axes = fig.add_axes(self._color_bar_coords)

        return fig, axes, colorbar_axes

    def _label_axes(
        self,
        ax: plt.Axes,
    ):
        ax.xaxis.set_major_locator(ticker.IndexLocator(4, 0))
        ax.xaxis.set_major_formatter("{x:.0f}")
        ax.xaxis.set_minor_locator(ticker.IndexLocator(2, 0))

        yticks_positions = np.arange(19) + 0.5
        yticks_labels = [
            "potassium",
            "calcium",
            "ph",
            "pco2",
            "lactate",
            "albumin",
            "bun",
            "creatinine",
            "sodium",
            "bicarbonate",
            "platelet",
            "glucose",
            "magnesium",
            "heartrate",
            "sysbp",
            "diasbp",
            "tempc",
            "resprate",
            "spo2",
        ]

        ax.set_yticks(yticks_positions)
        ax.set_yticklabels(yticks_labels, rotation=0)
        # ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

        # ax.set_yticks(np.arange(19))
        # ax.yaxis.set_major_locator(ticker.IndexLocator(self._ytick_major, 0))
        # ax.yaxis.set_major_formatter("{x:.0f}")
        # ax.yaxis.set_minor_locator(ticker.IndexLocator(self._ytick_minor, 0))

    def plot_samples(
        self,
        susceptibilities: list[pd.DataFrame],
    ):
        fig, axes, colorbar_axes = self._set_subplot_layout()

        num_samples_to_plot = self._subplot_num_rows * self._subplot_num_cols

        for plot_num in range(num_samples_to_plot):
            # for i, ax in enumerate(axes):
            sns.heatmap(
                data=susceptibilities[plot_num].T,
                ax=axes[plot_num],
                cmap=self._color_scheme,
                cbar=(plot_num == (num_samples_to_plot - 1)),
                cbar_ax=(
                    colorbar_axes
                    if (plot_num == (num_samples_to_plot - 1))
                    else None
                ),
                norm=LogNorm(),
            )

            # predicted_label = (
            #     predicted_labels[plot_num] if predicted_labels else None
            # )

            self._label_axes(
                ax=axes[plot_num],
                # actual_label=actual_labels[plot_num],
                # predicted_label=predicted_label,
            )
        plt.show()


if __name__ == "__main__":
    plotter = SusceptibilityGridPlotter(subplot_num_rows=2)

    first_examples_s_ij = rio.ResourceImporter().import_pickle_to_object(
        path=cfg_paths.ATTACK_OUTPUT_DIR
        / "plot_practice"
        / "first_s_ij.pickle"
    )

    best_examples_s_ij = rio.ResourceImporter().import_pickle_to_object(
        path=cfg_paths.ATTACK_OUTPUT_DIR / "plot_practice" / "best_s_ij.pickle"
    )

    plotter.plot_samples(
        susceptibilities=[first_examples_s_ij, best_examples_s_ij]
    )

    # plotter.plot_samples(
    #     features=data_features, actual_labels=data_actual_labels
    # )
