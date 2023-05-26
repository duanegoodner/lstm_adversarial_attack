import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import torch
from torch.utils.data import DataLoader

# from cv_trainer import WeightedRandomSamplerBuilder
from weighted_dataloader_builder import WeightedDataLoaderBuilder
from x19_mort_dataset import X19MortalityDataset


class X19MortalitySamplePlotter:
    def __init__(
        self,
        fig_size: tuple[int, int] = (6, 6),
        subplot_num_rows: int = 4,
        color_bar_coords: tuple[float, float, float, float] = (
            0.92,
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
        y2_label_rotation: int | float = 0,
        y2_label_pad: int | float = 15,
        y2_label_font_size: int | float = 9,
        color_scheme: str = "RdYlBu_r",
    ):
        assert 2 <= subplot_num_rows <= 4
        self._fig_size = fig_size
        self._subplot_num_rows = subplot_num_rows
        self._subplot_num_cols = 1  # must have exactly 1 column
        self._subplot_right_adjust: float = 0.8
        self._color_bar_coords = color_bar_coords
        self._share_xtick_labels = share_xtick_labels
        self._share_ytick_labels = share_ytick_labels
        self._xtick_major = xtick_major
        self._xtick_minor = xtick_minor
        self._ytick_major = ytick_major
        self._ytick_minor = ytick_minor
        self._y2_label_rotation = y2_label_rotation
        self._y2_label_pad = y2_label_pad
        self._y2_label_font_size = y2_label_font_size
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
        plt.subplots_adjust(right=self._subplot_right_adjust)
        colorbar_axes = fig.add_axes(self._color_bar_coords)

        return fig, axes, colorbar_axes

    def _label_axes(
        self,
        ax: plt.Axes,
        actual_label: int,
        predicted_label: int = None,
    ):
        ax.xaxis.set_major_locator(ticker.MultipleLocator(self._xtick_major))
        ax.xaxis.set_major_formatter("{x:.0f}")
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(self._xtick_minor))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(self._ytick_major))
        ax.yaxis.set_major_formatter("{x:.0f}")
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(self._ytick_minor))
        ax2 = ax.twinx()
        ax2.tick_params(axis="y", length=0, labelsize=0, labelcolor="white")

        ax2_label = f"$M_{{act}}$ = {actual_label}"

        if predicted_label is not None:
            ax2_label = f"{ax2_label}\n$M_{{pred}}$ = {predicted_label}"

        ax2.set_ylabel(
            ax2_label,
            rotation=self._y2_label_rotation,
            labelpad=self._y2_label_pad,
            fontsize=self._y2_label_font_size,
        )

    def plot_samples(
        self,
        features: torch.tensor,
        actual_labels: torch.tensor,
        predicted_labels: torch.tensor = None,
    ):
        fig, axes, colorbar_axes = self._set_subplot_layout()

        num_samples_to_plot = min(
            self._subplot_num_rows * self._subplot_num_cols, len(actual_labels)
        )

        for plot_num in range(num_samples_to_plot):
            # for i, ax in enumerate(axes):
            sns.heatmap(
                data=torch.squeeze(features[plot_num]),
                ax=axes[plot_num],
                cmap=self._color_scheme,
                cbar=(plot_num == (num_samples_to_plot - 1)),
                cbar_ax=(
                    colorbar_axes
                    if (plot_num == (num_samples_to_plot - 1))
                    else None
                ),
            )

            predicted_label = (
                predicted_labels[plot_num] if predicted_labels else None
            )

            self._label_axes(
                ax=axes[plot_num],
                actual_label=actual_labels[plot_num],
                predicted_label=predicted_label,
            )
        plt.show()


if __name__ == "__main__":
    dataset = X19MortalityDataset()
    data_loader = WeightedDataLoaderBuilder().build(
        dataset=dataset, batch_size=4
    )
    data_iter = iter(data_loader)
    data_features, data_actual_labels = next(data_iter)
    plotter = X19MortalitySamplePlotter()
    plotter.plot_samples(
        features=data_features, actual_labels=data_actual_labels
    )
