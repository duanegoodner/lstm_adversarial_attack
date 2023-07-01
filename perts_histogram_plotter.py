import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import lstm_adversarial_attack.config_paths as cfg_paths
import lstm_adversarial_attack.resource_io as rio


class PerturbationHistogramPlotter:
    def __init__(
        self,
        pert_summary_dfs: list[list[pd.DataFrame]],
        fig_size: tuple[int, int] = (10, 7),
    ):
        self.pert_summary_dfs = pert_summary_dfs
        self.subplot_num_rows = len(self.pert_summary_dfs)
        self.subplot_num_cols = len(self.pert_summary_dfs[0])
        self.fig_size = fig_size

    def _set_figure_layout(self):
        fig, axes = plt.subplots(
            nrows=self.subplot_num_rows,
            ncols=self.subplot_num_cols,
            figsize=self.fig_size,
        )

        # plt.subplots_adjust(
        #     left=self.subplot_left_adjust,
        #     right=self.subplot_right_adjust,
        #     hspace=0.45,
        #     wspace=0.35,
        # )

        return fig, axes

    def plot_histograms(self):
        fig, axes = self._set_figure_layout()

        for plot_row in range(self.subplot_num_rows):
            for plot_col in range(self.subplot_num_cols):
                cur_df = self.pert_summary_dfs[plot_row][plot_col]
                counts, bins = np.histogram(
                    cur_df.num_perts, bins=30, range=(0, 30)
                )
                cur_ax = axes[plot_row, plot_col]
                cur_ax.hist(bins[:-1], bins, weights=counts)

        plt.show()


if __name__ == "__main__":
    zto_first_df = rio.ResourceImporter().import_pickle_to_object(
        path=cfg_paths.ATTACK_OUTPUT_DIR
        / "plot_practice"
        / "zto_first_df.pickle"
    )
    zto_best_df = rio.ResourceImporter().import_pickle_to_object(
        path=cfg_paths.ATTACK_OUTPUT_DIR
        / "plot_practice"
        / "zto_best_df.pickle"
    )

    otz_first_df = rio.ResourceImporter().import_pickle_to_object(
        path=cfg_paths.ATTACK_OUTPUT_DIR
        / "plot_practice"
        / "otz_first_df.pickle"
    )
    otz_best_df = rio.ResourceImporter().import_pickle_to_object(
        path=cfg_paths.ATTACK_OUTPUT_DIR
        / "plot_practice"
        / "otz_best_df.pickle"
    )

    plotter = PerturbationHistogramPlotter(
        pert_summary_dfs=[
            [zto_first_df, zto_best_df],
            [otz_first_df, otz_best_df],
        ]
    )

    plotter.plot_histograms()
