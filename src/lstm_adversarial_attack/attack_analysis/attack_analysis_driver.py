from functools import cached_property
from pathlib import Path
import pandas as pd
import lstm_adversarial_attack.attack.attack_result_data_structs as ads
import lstm_adversarial_attack.attack_analysis.attack_analysis as ata
import lstm_adversarial_attack.config_paths as cfg_paths
import lstm_adversarial_attack.config_settings as cfg_settings
import lstm_adversarial_attack.resource_io as rio
import lstm_adversarial_attack.path_searches as ps
import lstm_adversarial_attack.attack_analysis.perts_histogram_plotter as php
import lstm_adversarial_attack.attack_analysis.susceptibility_plotter as ssp


class AttackAnalysesBuilder:
    def __init__(
        self,
        trainer_result_path: Path = None,
        seq_length: int = 48,
        min_num_perts: int = None,
        max_num_perts: int = None,
    ):
        if trainer_result_path is None:
            result_dir = ps.subdir_with_latest_content_modification(
                root_path=cfg_paths.FROZEN_HYPERPARAMETER_ATTACK
            )
            final_results = list(
                result_dir.glob("*final_attack_result.pickle")
            )
            assert len(final_results) == 1
            trainer_result_path = final_results[0]
        self.trainer_result_path = trainer_result_path
        self.seq_length = seq_length
        self.min_num_perts = min_num_perts
        self.max_num_perts = max_num_perts

    @cached_property
    def trainer_result(self) -> ads.TrainerResult:
        return rio.ResourceImporter().import_pickle_to_object(
            path=self.trainer_result_path
        )

    @cached_property
    def success_summary(self) -> ads.TrainerSuccessSummary:
        return ads.TrainerSuccessSummary(trainer_result=self.trainer_result)

    @cached_property
    def attack_results(self) -> ata.FullAttackResults:
        return ata.FullAttackResults(
            success_summary=self.success_summary,
        )

    @cached_property
    def standard_attack_analyses(self) -> ata.StandardAttackAnalyses:
        return ata.StandardAttackAnalyses(
            zero_to_one_first=self.attack_results.get_condition_analysis(
                seq_length=self.seq_length,
                example_type=ata.RecordedExampleType.FIRST,
                orig_label=0,
                min_num_perts=self.min_num_perts,
                max_num_perts=self.max_num_perts,
            ),
            zero_to_one_best=self.attack_results.get_condition_analysis(
                seq_length=self.seq_length,
                example_type=ata.RecordedExampleType.BEST,
                orig_label=0,
                min_num_perts=self.min_num_perts,
                max_num_perts=self.max_num_perts,
            ),
            one_to_zero_first=self.attack_results.get_condition_analysis(
                seq_length=self.seq_length,
                example_type=ata.RecordedExampleType.FIRST,
                orig_label=1,
                min_num_perts=self.min_num_perts,
                max_num_perts=self.max_num_perts,
            ),
            one_to_zero_best=self.attack_results.get_condition_analysis(
                seq_length=self.seq_length,
                example_type=ata.RecordedExampleType.BEST,
                orig_label=1,
                min_num_perts=self.min_num_perts,
                max_num_perts=self.max_num_perts,
            ),
        )

    @cached_property
    def df_tuple_for_histogram_plotter(
        self,
    ) -> tuple[tuple[pd.DataFrame, ...], ...]:
        return self.standard_attack_analyses.data_for_histogram_plotter

    def plot_histograms(self, title: str, subtitle: str, **kwargs):
        histogram_plotter = php.PerturbationHistogramPlotter(
            pert_summary_dfs=self.df_tuple_for_histogram_plotter,
            title=title,
            subtitle=subtitle,
            **kwargs
        )
        histogram_plotter.plot_histograms()

    def plot_susceptibility_metric(
        self, metric: str, title: str, color_bar_title: str = None
    ):
        susceptibility_dfs = (
            self.standard_attack_analyses.data_for_susceptibility_plotter(
                metric=metric
            )
        )
        plotter = ssp.SusceptibilityPlotter(
            susceptibility_dfs=susceptibility_dfs, main_plot_title=title
        )
        plotter.plot_susceptibilities(color_bar_title=color_bar_title)

def plot_latest_result():
    latest_attack_analysis_builder = AttackAnalysesBuilder(
        seq_length=cfg_settings.MAX_OBSERVATION_HOURS
    )

    latest_attack_analysis_builder.plot_susceptibility_metric(
        metric="gpp_ij",
        title="Latest attack: Perturbation probabilities",
        color_bar_title="Perturbation Probability"
    )
    latest_attack_analysis_builder.plot_susceptibility_metric(
        metric="ganzp_ij",
        title="Latest attack: Mean values of non-zero perturbation elements",
        color_bar_title="Mean Perturbation Magnitude (when not zero)"
    )
    latest_attack_analysis_builder.plot_histograms(
        title="Perturbation density and magnitude distributions",
        subtitle="Attack: Latest",
        histogram_num_bins=(912, 1000, 50),
        create_insets=((True, True, False), (True, True, False)),
        inset_specs=(
            (
                php.InsetSpec(
                    bounds=[0.2, 0.15, 0.6, 0.82],
                    plot_limits=php.PlotLimits(
                        x_min=1, x_max=20, y_min=0, y_max=20
                    ),
                ),
                php.InsetSpec(
                    bounds=[0.3, 0.15, 0.65, 0.82],
                    plot_limits=php.PlotLimits(
                        x_min=0, x_max=0.05, y_min=0, y_max=3000
                    ),
                ),
                None,
            ),
            (
                php.InsetSpec(
                    bounds=[0.2, 0.15, 0.6, 0.82],
                    plot_limits=php.PlotLimits(
                        x_min=1, x_max=20, y_min=0, y_max=20
                    ),
                ),
                php.InsetSpec(
                    bounds=[0.3, 0.15, 0.65, 0.82],
                    plot_limits=php.PlotLimits(
                        x_min=0, x_max=0.05, y_min=0, y_max=500
                    ),
                ),
                None,
            ),
        ),
    )

if __name__ == "__main__":
    plot_latest_result()

    # max_single_element_result_path = (
    #     cfg_paths.FROZEN_HYPERPARAMETER_ATTACK
    #     / "2023-06-28_17_50_46.701620"
    #     / "2023-06-28_19_02_16.499026_final_attack_result.pickle"
    # )
    # max_single_element_analyses_builder = AttackAnalysesBuilder(
    #     trainer_result_path=max_single_element_result_path, seq_length=48
    # )
    #
    # max_single_element_analyses_builder.plot_histograms(
    #     title="Perturbation density and magnitude distributions",
    #     subtitle=(
    #         "Tuning objective: Maximize # of perturbation elements with "
    #         "exactly one non-zero element"
    #     ),
    #     histogram_num_bins=(912, 50, 50),
    #     create_insets=((True, False, False), (True, False, False)),
    #     inset_specs=(
    #         (
    #             php.InsetSpec(
    #                 bounds=[0.2, 0.15, 0.6, 0.82],
    #                 plot_limits=php.PlotLimits(
    #                     x_min=1, x_max=20, y_min=0, y_max=2000
    #                 ),
    #             ),
    #             None,
    #             None,
    #         ),
    #         (
    #             php.InsetSpec(
    #                 bounds=[0.2, 0.15, 0.6, 0.82],
    #                 plot_limits=php.PlotLimits(
    #                     x_min=1, x_max=20, y_min=0, y_max=100
    #                 ),
    #             ),
    #             None,
    #             None,
    #         ),
    #     ),
    # )
    #
    # max_single_element_analyses_builder.plot_susceptibility_metric(
    #     metric="ganzp_ij",
    #     title=(
    #         "ganzp_ij when tuned to maximize # of single-element perturbations"
    #     ),
    # )
    #
    # max_single_element_analyses_builder.plot_susceptibility_metric(
    #     metric="gpp_ij",
    #     title=(
    #         "gpp_ij when tuned to maximize # of single-element perturbations"
    #     ),
    # )
    #
    # max_single_element_analyses_builder.plot_susceptibility_metric(
    #     metric="sensitivity_ij",
    #     title=(
    #         "sensitivity_ij when tuned to maximize # of single-element"
    #         " perturbations"
    #     ),
    # )

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
    #     create_insets=((True, True, False), (True, True, False)),
    #     inset_specs=(
    #         (
    #             php.InsetSpec(
    #                 bounds=[0.2, 0.15, 0.6, 0.82],
    #                 plot_limits=php.PlotLimits(
    #                     x_min=1, x_max=20, y_min=0, y_max=20
    #                 ),
    #             ),
    #             php.InsetSpec(
    #                 bounds=[0.3, 0.15, 0.65, 0.82],
    #                 plot_limits=php.PlotLimits(
    #                     x_min=0, x_max=0.05, y_min=0, y_max=3000
    #                 ),
    #             ),
    #             None,
    #         ),
    #         (
    #             php.InsetSpec(
    #                 bounds=[0.2, 0.15, 0.6, 0.82],
    #                 plot_limits=php.PlotLimits(
    #                     x_min=1, x_max=20, y_min=0, y_max=20
    #                 ),
    #             ),
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
