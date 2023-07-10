import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm
from typing import Any
import lstm_adversarial_attack.resource_io as rio
import lstm_adversarial_attack.config_paths as cfg_paths


class SusceptibilityPlotter:
    """
    Plots heatmaps of susceptibility metrics for perturbations causing
    adversarial examples
    """

    def __init__(
        self,
        susceptibility_dfs: tuple[tuple[pd.DataFrame, ...], ...],
        main_plot_title: str,
        df_titles: tuple[tuple[str, ...], ...] = (
            (
                "0 \u2192 1 Attack, First Examples",
                "0 \u2192 1 Attack, Best Examples",
            ),
            (
                "1 \u2192 0 Attack, First Examples",
                "1 \u2192 0 Attack, Best Examples",
            ),
        ),
        fig_size: tuple[int, int] = (10, 7),
        xtick_major: int = 12,
        xtick_minor: int = 4,
        ytick_major: int = 2,
        ytick_minor: int = 1,
        color_bar_coords: tuple[float, float, float, float] = (
            0.85,
            0.3,
            0.02,
            0.4,
        ),
        color_scheme: str = "RdYlBu_r",
        subplot_top_adjust: float = 0.80,
        subplot_bottom_adjust: float = 0.15,
        subplot_left_adjust: float = 0.15,
        subplot_right_adjust: float = 0.80,
    ):
        self.susceptibility_dfs = susceptibility_dfs
        self.subplot_num_rows = len(self.susceptibility_dfs)
        self.subplot_num_cols = len(self.susceptibility_dfs[0])
        self.df_titles = df_titles
        self.main_plot_title = main_plot_title
        self.fig_size = fig_size
        self.xtick_major = xtick_major
        self.xtick_minor = xtick_minor
        self.ytick_major = ytick_major
        self.ytick_minor = ytick_minor
        self.color_bar_coords = color_bar_coords
        self.color_scheme = color_scheme
        self.subplot_top_adjust = subplot_top_adjust
        self.subplot_bottom_adjust = subplot_bottom_adjust
        self.subplot_left_adjust = subplot_left_adjust
        self.subplot_right_adjust = subplot_right_adjust

        for row in self.susceptibility_dfs:
            for df in row:
                assert (
                    df.columns == self.susceptibility_dfs[0][0].columns
                ).all()

        self.measurement_names = self.susceptibility_dfs[0][0].columns
        self.yticks_labels = self.susceptibility_dfs[0][0].columns
        self.yticks_positions = np.arange(len(self.yticks_labels) + 0.5)

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
            hspace=0.5,
            wspace=0.35,
        )

        fig.text(
            0.5,
            0.9,
            self.main_plot_title,
            ha="center",
            rotation="horizontal",
            fontsize=18,
        )

        colorbar_axes = fig.add_axes(self.color_bar_coords)

        return fig, axes, colorbar_axes

    def _decorate_subplots(
        self,
        axes: plt.Axes,
    ):
        """
        Adds titles and labels to individual subplots
        """
        # x-axis tick marks and title
        for plot_row in range(self.subplot_num_rows):
            for plot_col in range(self.subplot_num_cols):
                ax = axes[plot_row][plot_col]

                ax.xaxis.set_major_locator(
                    ticker.IndexLocator(self.xtick_major, 0)
                )
                ax.xaxis.set_major_formatter("{x:.0f}")
                ax.xaxis.set_minor_locator(
                    ticker.IndexLocator(self.xtick_minor, 0)
                )
                ax.tick_params(axis="x", labelsize=9)
                ax.set_xlabel("Elapsed Time (hours)")



                # y-axis tick marks and title
                # yticks_positions = np.arange(len(self.yticks_labels)) + 0.5
                # ax.set_yticks(yticks_positions)
                # ax.set_yticklabels(
                #     np.arange(len(self.measurement_names)),
                #     fontsize=8,
                #     rotation=0,
                # )
                ax.yaxis.set_major_locator(
                    ticker.IndexLocator(self.ytick_major, 0)
                )
                ax.yaxis.set_major_formatter("{x:.0f}")
                ax.yaxis.set_minor_locator(
                    ticker.IndexLocator(self.ytick_minor, 9)
                )
                ax.tick_params(axis="y", labelsize=8)
                ax.set_ylabel("Measurement ID")


                # subplot title
                ax.set_title(
                    f"{self.df_titles[plot_row][plot_col]}",
                    x=0.45,
                    y=1.0,
                )

    @staticmethod
    def decorate_heatmap(heatmap: sns.heatmap, source_df: pd.DataFrame):
        heatmap.axhline(y=0, color="k", linewidth=2)
        heatmap.axhline(y=len(source_df.columns), color="k", linewidth=2)
        heatmap.axvline(x=0, color="k", linewidth=2)
        heatmap.axvline(x=source_df.index.max() + 1, color="k", linewidth=2)

    def plot_susceptibilities(self, color_bar_title: str = ""):
        """
        Generates all heatmap susceptibility plots in figure
        """
        fig, axes, colorbar_axes = self._set_figure_layout()

        for plot_row in range(self.subplot_num_rows):
            for plot_col in range(self.subplot_num_cols):
                cur_df = self.susceptibility_dfs[plot_row][plot_col]
                cur_plot = sns.heatmap(
                    data=cur_df.T,
                    ax=axes[plot_row, plot_col],
                    cmap=self.color_scheme,
                    cbar=(plot_row == self.subplot_num_rows - 1)
                    and (plot_col == self.subplot_num_cols - 1),
                    cbar_ax=colorbar_axes,
                    norm=LogNorm(),
                    edgecolor="black",
                )

                self.decorate_heatmap(heatmap=cur_plot, source_df=cur_df)

                if (plot_row == self.subplot_num_rows - 1) and (
                    plot_col == self.subplot_num_cols - 1
                ):
                    cur_plot.collections[0].colorbar.set_label(color_bar_title)

        self._decorate_subplots(
            axes=axes,
            # plot_title=f"{self.df_titles[plot_row][plot_col]}",
        )

        plt.show()


if __name__ == "__main__":
    first_examples_s_ij = rio.ResourceImporter().import_pickle_to_object(
        path=cfg_paths.ATTACK_OUTPUT_DIR
        / "plot_practice"
        / "first_s_ij.pickle"
    )

    best_examples_s_ij = rio.ResourceImporter().import_pickle_to_object(
        path=cfg_paths.ATTACK_OUTPUT_DIR / "plot_practice" / "best_s_ij.pickle"
    )

    plotter = SusceptibilityPlotter(
        susceptibility_dfs=(
            (first_examples_s_ij, best_examples_s_ij),
            (first_examples_s_ij, best_examples_s_ij),
        ),
        df_titles=(
            (
                "0 \u2192 1 Attack, First Examples",
                "0 \u2192 1 Attack, Best Examples",
            ),
            (
                "1 \u2192 0 Attack, First Examples",
                "1 \u2192 0 Attack, Best Examples",
            ),
        ),
        main_plot_title="Perturbation Susceptibility Scores",
    )
    plotter.plot_susceptibilities()
