from functools import cached_property
from pathlib import Path

import pandas as pd

import lstm_adversarial_attack.attack.attack_result_data_structs as ads
import lstm_adversarial_attack.attack.attack_summary as asu
import lstm_adversarial_attack.config_paths as cfg_paths
import lstm_adversarial_attack.resource_io as rio
import lstm_adversarial_attack.attack.attack_results_analyzer as ara
import lstm_adversarial_attack.attack.perts_histogram_plotter as php
import lstm_adversarial_attack.attack.susceptibility_plotter as ssp


class AttackAnalysesBuilder:
    def __init__(self, trainer_result_path: Path, seq_length: int = 48):
        self.trainer_result_path = trainer_result_path
        self.seq_length = seq_length

    @cached_property
    def trainer_result(self) -> ads.TrainerResult:
        return rio.ResourceImporter().import_pickle_to_object(
            path=self.trainer_result_path
        )

    @cached_property
    def full_attack_results(self) -> asu.FullAttackResults:
        return asu.FullAttackResults(trainer_result=self.trainer_result)

    @cached_property
    def standard_attack_analyses(self) -> ara.StandardAttackAnalyses:
        return ara.StandardAttackAnalyses(
            full_attack_results=self.full_attack_results,
            seq_length=self.seq_length,
        )

    @cached_property
    def df_tuple_for_histogram_plotter(
        self,
    ) -> tuple[tuple[pd.DataFrame, ...], ...]:
        return self.standard_attack_analyses.df_tuple_for_histogram_plotter

    def plot_histograms(self, title: str, subtitle: str, **kwargs):
        histogram_plotter = php.PerturbationHistogramPlotter(
            pert_summary_dfs=self.df_tuple_for_histogram_plotter,
            title=title,
            subtitle=subtitle,
            **kwargs
        )
        histogram_plotter.plot_histograms()

    def plot_susceptibility_metric(self, metric: str, title: str):
        susceptibility_dfs = self.standard_attack_analyses.susceptibility_metric_tuple_for_plotting(
            metric=metric
        )
        plotter = ssp.SusceptibilityPlotter(
            susceptibility_dfs=susceptibility_dfs,
            main_plot_title=title
        )
        plotter.plot_susceptibilities()


if __name__ == "__main__":
    max_single_element_result_path = (
        cfg_paths.FROZEN_HYPERPARAMETER_ATTACK
        / "2023-06-28_17_50_46.701620"
        / "2023-06-28_19_02_16.499026_final_attack_result.pickle"
    )
    max_single_element_analyses_builder = AttackAnalysesBuilder(
        trainer_result_path=max_single_element_result_path, seq_length=48
    )

    max_single_element_analyses_builder.plot_histograms(
        title="Perturbation density and magnitude distributions",
        subtitle=(
            "Tuning objective: Maximize # of perturbation elements with "
            "exactly one non-zero element"
        ),
        histogram_num_bins=(912, 50, 50),
        create_insets=((True, False, False), (True, False, False)),
        inset_specs=(
            (
                php.InsetSpec(
                    bounds=[0.2, 0.15, 0.6, 0.82],
                    plot_limits=php.PlotLimits(
                        x_min=1, x_max=20, y_min=0, y_max=2000
                    ),
                ),
                None,
                None,
            ),
            (
                php.InsetSpec(
                    bounds=[0.2, 0.15, 0.6, 0.82],
                    plot_limits=php.PlotLimits(
                        x_min=1, x_max=20, y_min=0, y_max=100
                    ),
                ),
                None,
                None,
            ),
        ),
    )

    max_single_element_analyses_builder.plot_susceptibility_metric(
        metric="s_ganzp_ij",
        title="gpp_ij when tuned to maximize # of single-element perturbations"
    )

    # max_sparsity_result_path = (
    #     cfg_paths.FROZEN_HYPERPARAMETER_ATTACK
    #     / "2023-06-30_12_05_34.834996"
    #     / "2023-06-30_14_20_57.875925_final_attack_result.pickle"
    # )
    # max_sparsity_analyses_builder = AttackAnalysesBuilder(
    #     trainer_result_path=max_sparsity_result_path, seq_length=48
    # )
    # max_sparsity_analyses_builder.plot_histograms(
    #     title="Perturbation density and magnitude distributions",
    #     subtitle="Tuning objective: Maximize sparsity",
    #     histogram_num_bins=(912, 200, 50),
    #     create_insets=((True, False, False), (True, False, False)),
    #     inset_specs=(
    #         (
    #             php.InsetSpec(
    #                 bounds=[0.2, 0.15, 0.6, 0.82],
    #                 plot_limits=php.PlotLimits(
    #                     x_min=1, x_max=20, y_min=0.0, y_max=1500
    #                 ),
    #             ),
    #             None,
    #             None,
    #         ),
    #         (
    #             php.InsetSpec(
    #                 bounds=[0.2, 0.15, 0.6, 0.82],
    #                 plot_limits=php.PlotLimits(
    #                     x_min=1, x_max=20, y_min=0, y_max=80
    #                 ),
    #             ),
    #             None,
    #             None,
    #         ),
    #     ),
    # )
    #
    #
    # max_sparse_small_max_result_path = (
    #     cfg_paths.FROZEN_HYPERPARAMETER_ATTACK
    #     / "2023-07-01_12_01_25.552909"
    #     / "2023-07-01_13_17_06.787795_final_attack_result.pickle"
    # )
    # max_sparse_small_analyses_builder = AttackAnalysesBuilder(
    #     trainer_result_path=max_sparse_small_max_result_path, seq_length=48
    # )
    # max_sparse_small_analyses_builder.plot_histograms(
    #     title="Perturbation density and magnitude distributions",
    #     subtitle="Tuning objective: Maximize sparse-small-max score",
    #     histogram_num_bins = (912, 1000, 50),
    #     create_insets=((False, True, False), (False, True, False)),
    #     inset_specs=(
    #         (
    #             None,
    #             php.InsetSpec(
    #                 bounds=[0.3, 0.15, 0.65, 0.82],
    #                 plot_limits=php.PlotLimits(
    #                     x_min=0, x_max=0.05, y_min=0, y_max=3000
    #                 ),
    #             ),
    #             None,
    #         ),
    #         (
    #             None,
    #             php.InsetSpec(
    #                 bounds=[0.3, 0.15, 0.65, 0.82],
    #                 plot_limits=php.PlotLimits(
    #                     x_min=0, x_max=0.05, y_min=0, y_max=500
    #                 ),
    #             ),
    #             None,
    #         ),
    #     ),
    # )
