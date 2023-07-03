from functools import cached_property
from pathlib import Path
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
            seq_length=self.seq_length
        )


if __name__ == "__main__":
    # max_single_element_result_path = (
    #     cfg_paths.FROZEN_HYPERPARAMETER_ATTACK
    #     / "2023-06-28_17_50_46.701620"
    #     / "2023-06-28_19_02_16.499026_final_attack_result.pickle"
    # )
    # max_single_element_analyses_builder = AttackAnalysesBuilder(
    #     trainer_result_path=max_single_element_result_path,
    #     seq_length=48
    # )
    # max_single_element_hist_plotter = php.PerturbationHistogramPlotter.from_standard_attack_analyses(
    #     attack_analyses=max_single_element_analyses_builder.standard_attack_analyses,
    #     title="Perturbation density and magnitude distributions",
    #     subtitle="Tuning objective: Maximize # of perturbation elements with "
    #              "exactly one non-zero element",
    # )


    # max_sparsity_result_path = (
    #     cfg_paths.FROZEN_HYPERPARAMETER_ATTACK
    #     / "2023-06-30_12_05_34.834996"
    #     / "2023-06-30_14_20_57.875925_final_attack_result.pickle"
    # )
    # max_sparsity_analyses_builder = AttackAnalysesBuilder(
    #     trainer_result_path=max_sparsity_result_path,
    #     seq_length=48
    # )
    # max_sparsity_hist_plotter = php.PerturbationHistogramPlotter.from_standard_attack_analyses(
    #     attack_analyses=max_sparsity_analyses_builder.standard_attack_analyses,
    #     title="Perturbation density and magnitude distributions",
    #     subtitle="Tuning objective: Maximize sparsity",
    #     histogram_num_bins=(912, 50, 50),
    #     histogram_plot_ranges=((0, 912), (0, 1.), (0, 1.))
    #
    # )


    max_sparse_small_max_result_path = (
        cfg_paths.FROZEN_HYPERPARAMETER_ATTACK
        / "2023-07-01_12_01_25.552909"
        / "2023-07-01_13_17_06.787795_final_attack_result.pickle"
    )
    max_sparse_small_analyses_builder = AttackAnalysesBuilder(
        trainer_result_path=max_sparse_small_max_result_path,
        seq_length=48
    )
    max_sparse_small_hist_plotter = php.PerturbationHistogramPlotter.from_standard_attack_analyses(
        attack_analyses=max_sparse_small_analyses_builder.standard_attack_analyses,
        title="Perturbation density and magnitude distributions",
        subtitle="Tuning objective: Maximize sparse-small-max score",
        histogram_num_bins=(912, 50, 50),
        histogram_plot_ranges=((0, 912), (0, 0.05), (0, 1.))
    )

