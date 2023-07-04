from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd

import lstm_adversarial_attack.attack.attack_summary as asu
import lstm_adversarial_attack.attack.attack_susceptibility_metrics as asm
import lstm_adversarial_attack.config_paths as cfg_paths
import lstm_adversarial_attack.resource_io as rio


class AttackConditionAnalysis:
    def __init__(
        self,
        full_attack_results: asu.FullAttackResults,
        seq_length: int,
        example_type: asu.RecordedExampleType = None,
        orig_label: int = None,
        min_num_perts: int = None,
        max_num_perts: int = None,
    ):
        self._full_attack_results = full_attack_results
        self.seq_length = seq_length
        self.example_type = example_type
        self.orig_label = orig_label
        self.min_num_perts = min_num_perts
        self.max_num_perts = max_num_perts

    @property
    def _examples_df_dispatch(self) -> dict:
        return {
            asu.RecordedExampleType.FIRST: (
                self._full_attack_results.first_examples_df
            ),
            asu.RecordedExampleType.BEST: (
                self._full_attack_results.best_examples_df
            ),
        }

    @property
    def _perts_summary_dispatch(self) -> dict:
        return {
            asu.RecordedExampleType.FIRST: (
                self._full_attack_results.first_perts_summary
            ),
            asu.RecordedExampleType.BEST: (
                self._full_attack_results.best_perts_summary
            ),
        }

    @cached_property
    def filtered_examples_df(self) -> pd.DataFrame:
        orig_df = self._examples_df_dispatch[self.example_type]
        filtered_df = orig_df[
            (orig_df["seq_length"] == self.seq_length)
            & (orig_df["orig_label"] == self.orig_label)
        ]
        if self.min_num_perts is not None:
            filtered_df = filtered_df[
                filtered_df["num_perts"] >= self.min_num_perts
            ]

        if self.max_num_perts is not None:
            filtered_df = filtered_df[
                filtered_df["num_perts"] <= self.max_num_perts
            ]

        return filtered_df

    @cached_property
    def susceptibility_metrics(self) -> asm.AttackSusceptibilityMetrics:
        perts_summary = self._perts_summary_dispatch[self.example_type]
        return asm.AttackSusceptibilityMetrics(
            perts=perts_summary.padded_perts[
                self.filtered_examples_df.index, :, :
            ]
        )


class StandardAttackAnalyses:
    def __init__(
        self,
        full_attack_results: asu.FullAttackResults,
        seq_length: int,
        min_num_perts: int = None,
        max_num_perts: int = None,
    ):
        self._full_attack_results = full_attack_results
        self.seq_length = seq_length
        self.min_num_perts = min_num_perts
        self.max_num_perts = max_num_perts

    @cached_property
    def zero_to_one_first(self) -> AttackConditionAnalysis:
        return AttackConditionAnalysis(
            full_attack_results=self._full_attack_results,
            seq_length=self.seq_length,
            example_type=asu.RecordedExampleType.FIRST,
            orig_label=0,
            min_num_perts=self.min_num_perts,
            max_num_perts=self.max_num_perts,
        )

    @cached_property
    def zero_to_one_best(self) -> AttackConditionAnalysis:
        return AttackConditionAnalysis(
            full_attack_results=self._full_attack_results,
            seq_length=self.seq_length,
            example_type=asu.RecordedExampleType.BEST,
            orig_label=0,
            min_num_perts=self.min_num_perts,
            max_num_perts=self.max_num_perts,
        )

    @cached_property
    def one_to_zero_first(self) -> AttackConditionAnalysis:
        return AttackConditionAnalysis(
            full_attack_results=self._full_attack_results,
            seq_length=self.seq_length,
            example_type=asu.RecordedExampleType.FIRST,
            orig_label=1,
            min_num_perts=self.min_num_perts,
            max_num_perts=self.max_num_perts,
        )

    @cached_property
    def one_to_zero_best(self) -> AttackConditionAnalysis:
        return AttackConditionAnalysis(
            full_attack_results=self._full_attack_results,
            seq_length=self.seq_length,
            example_type=asu.RecordedExampleType.BEST,
            orig_label=1,
            min_num_perts=self.min_num_perts,
            max_num_perts=self.max_num_perts,
        )

    @cached_property
    def df_tuple_for_histogram_plotter(
        self,
    ) -> tuple[tuple[pd.DataFrame, ...], ...]:
        return (
            (
                self.zero_to_one_first.filtered_examples_df,
                self.zero_to_one_best.filtered_examples_df,
            ),
            (
                self.one_to_zero_first.filtered_examples_df,
                self.one_to_zero_best.filtered_examples_df,
            ),
        )

    def susceptibility_metric_tuple_for_plotting(
        self, metric: str
    ) -> tuple[tuple[pd.DataFrame | pd.Series, ...], ...]:
        return (
            (
                getattr(self.zero_to_one_first.susceptibility_metrics, metric),
                getattr(self.zero_to_one_best.susceptibility_metrics, metric),
            ),
            (
                getattr(self.one_to_zero_first.susceptibility_metrics, metric),
                getattr(self.one_to_zero_best.susceptibility_metrics, metric),
            ),
        )
