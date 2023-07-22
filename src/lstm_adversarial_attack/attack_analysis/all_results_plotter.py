from functools import cached_property
from pathlib import Path
import lstm_adversarial_attack.attack_analysis.attack_analysis as ata
import lstm_adversarial_attack.attack_analysis.perts_histogram_plotter as php
import lstm_adversarial_attack.attack_analysis.susceptibility_plotter as ssp
import lstm_adversarial_attack.config_paths as cfg_paths
import lstm_adversarial_attack.config_settings as cfg_settings
import lstm_adversarial_attack.path_searches as ps


class AllResultsPlotter:
    def __init__(
        self,
        attack_result_path: Path = None,
        seq_length: int = cfg_settings.ATTACK_ANALYSIS_DEFAULT_SEQ_LENGTH,
        label: str = None,
    ):
        if attack_result_path is None:
            attack_result_path = ps.latest_modified_file_with_name_condition(
                component_string="attack_result.pickle",
                root_dir=cfg_paths.FROZEN_HYPERPARAMETER_ATTACK,
                comparison_type=ps.StringComparisonType.SUFFIX,
            )
        self.attack_result_path = attack_result_path
        self.seq_length = seq_length
        self.label = label

    @cached_property
    def full_attack_results(self) -> ata.FullAttackResults:
        return ata.FullAttackResults.from_trainer_result_path(
            trainer_result_path=self.attack_result_path
        )

    @cached_property
    def attack_condition_summaries(self):
        return (
            self.full_attack_results.get_standard_attack_condition_summaries(
                seq_length=self.seq_length
            )
        )

    @cached_property
    def histogram_plotter(self) -> php.HistogramPlotter:
        return php.HistogramPlotter(
            title=f"Perturbation Element Histograms for {self.label}",
            perts_dfs=self.attack_condition_summaries.data_for_histogram_plotter,
        )

    # @cached_property
    # def susceptibility_plotter(self):
