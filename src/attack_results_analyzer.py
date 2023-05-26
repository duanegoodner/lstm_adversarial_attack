import argparse

import dill
import matplotlib.pyplot as plt
import numpy as np
import torch as torch
from functools import cached_property
from pathlib import Path
from adv_attack import AdversarialExamplesSummary


def import_pickle_to_adv_example_summary(
    path: Path,
) -> AdversarialExamplesSummary:
    with path.open(mode="rb") as p:
        result = dill.load(p)
    return result


def import_pickle_to_tuple(path: Path) -> tuple:
    with path.open(mode="rb") as p:
        result = dill.load(p)
    return result


class AttackResultsAnalyzer:
    def __init__(self, result_path: Path):
        self.result_path = result_path
        self.result = import_pickle_to_adv_example_summary(result_path)

    @cached_property
    def orig_zeros_indices(self) -> np.ndarray:
        return np.where(self.result.orig_labels == 0)[0]

    @cached_property
    def orig_ones_indices(self) -> np.ndarray:
        return np.where(self.result.orig_labels == 1)[0]

    @cached_property
    def num_oz_perts(self) -> int:
        return len(self.orig_ones_indices)

    @cached_property
    def num_zo_perts(self) -> int:
        return len(self.orig_zeros_indices)

    @cached_property
    def oz_perts(self) -> torch.tensor:
        return self.result.perturbations[self.orig_ones_indices, :, :]

    @cached_property
    def zo_perts(self) -> torch.tensor:
        return self.result.perturbations[self.orig_zeros_indices, :, :]

    @cached_property
    def abs_oz_perts(self) -> torch.tensor:
        return torch.abs(self.oz_perts)

    @cached_property
    def abs_zo_perts(self) -> torch.tensor:
        return torch.abs(self.zo_perts)

    @cached_property
    def mean_abs_oz_perts(self) -> torch.tensor:
        return torch.mean(self.abs_oz_perts, dim=0)

    @cached_property
    def mean_abs_zo_perts(self) -> torch.tensor:
        return torch.mean(self.abs_zo_perts, dim=0)

    @cached_property
    def gmp_ij_oz(self) -> torch.tensor:
        return torch.max(self.abs_oz_perts, dim=0).values

    @cached_property
    def gmp_ij_zo(self) -> torch.tensor:
        return torch.max(self.abs_zo_perts, dim=0).values

    @cached_property
    def gap_ij_oz(self) -> torch.tensor:
        return torch.sum(self.abs_oz_perts, dim=0) / self.num_oz_perts

    @cached_property
    def gap_ij_zo(self) -> torch.tensor:
        return torch.sum(self.abs_zo_perts, dim=0) / self.num_zo_perts

    @cached_property
    def gpp_ij_oz(self) -> torch.tensor:
        return torch.norm(self.abs_oz_perts, p=1, dim=0) / self.num_oz_perts

    @cached_property
    def gpp_ij_zo(self) -> torch.tensor:
        return torch.norm(self.abs_zo_perts, p=1, dim=0) / self.num_zo_perts

    @cached_property
    def s_ij_oz(self) -> torch.tensor:
        return self.gmp_ij_oz * self.gpp_ij_oz

    @cached_property
    def s_ij_zo(self) -> torch.tensor:
        return self.gmp_ij_zo * self.gpp_ij_zo

    @cached_property
    def s_j_oz(self) -> torch.tensor:
        return torch.sum(self.s_ij_oz, dim=1)

    @cached_property
    def s_j_zo(self) -> torch.tensor:
        return torch.sum(self.s_ij_zo, dim=1)

    @property
    def col_names(self) -> list[str]:
        return [
            "K",
            "Ca",
            "PH",
            "PaCO2",
            "Lactate",
            "Albumin",
            "BUN",
            "Cre",
            "Na",
            "HCO3",
            "Platelets",
            "Glc",
            "Mg",
            "HR",
            "SBP",
            "DBP",
            "TEMP",
            "RR",
            "SPO2",
        ]

    def plot_time_series_overlays(
        self,
        features: torch.tensor,
        x_label: str,
        y_label: str,
        plot_title: str,
    ):
        fig, ax = plt.subplots(1, 1)
        meas_times = torch.arange(0, 48)
        for meas_idx in range(features.shape[0]):
            ax.plot(
                meas_times, features[meas_idx], label=self.col_names[meas_idx]
            )
        ax.set_xlabel(xlabel=x_label)
        ax.set_ylabel(ylabel=y_label)
        ax.set_title(plot_title)
        ax.legend(bbox_to_anchor=(1.35, 0.5), loc="right")
        fig.subplots_adjust(right=0.75)

        plt.show()


if __name__ == "__main__":
    cur_parser = argparse.ArgumentParser()
    cur_parser.add_argument(
        "-f",
        "--attack_summary_file",
        type=str,
        nargs="?",
        action="store",
        help=(
            "Filename (not full path) .pickle file containing attack summary"
            " object."
        ),
    )
    args_namespace = cur_parser.parse_args()
    attack_results_dir = (
        Path(__file__).parent.parent / "data" / "attack_results_f48_00"
    )

    summary_file_path = (
        attack_results_dir / args_namespace.attack_summary_file
    )

    analyzer = AttackResultsAnalyzer(result_path=summary_file_path)

    analyzer.plot_time_series_overlays(
        features=analyzer.mean_abs_oz_perts,
        x_label="Elapsed time (hours) after ICU admission",
        y_label="Attack susceptibility",
        plot_title="1-to-0 attack susceptibility",
    )

    analyzer.plot_time_series_overlays(
        features=analyzer.mean_abs_zo_perts,
        x_label="Elapsed time (hours) after ICU admission",
        y_label="Attack susceptibility",
        plot_title="0-to-1 attack susceptibility",
    )
