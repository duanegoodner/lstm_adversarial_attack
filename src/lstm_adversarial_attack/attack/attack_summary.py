from dataclasses import dataclass
from enum import Enum, auto
from functools import cached_property

import numpy as np
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import lstm_adversarial_attack.attack.attack_result_data_structs as ads
import lstm_adversarial_attack.attack.attack_susceptibility_metrics as asm


class RecordedExampleType(Enum):
    FIRST = auto()
    BEST = auto()


@dataclass
class AttackConditionAnalysisNew:
    examples_df: pd.DataFrame
    perts: np.array

    def __post_init__(self):
        assert len(self.examples_df == self.perts.shape[0])

    @cached_property
    def susceptibility_metrics(self) -> asm.AttackSusceptibilityMetrics:
        return asm.AttackSusceptibilityMetrics(perts=self.perts)


@dataclass
class StandardAttackAnalysesNew:
    zero_to_one_first: AttackConditionAnalysisNew
    zero_to_one_best: AttackConditionAnalysisNew
    one_to_zero_first: AttackConditionAnalysisNew
    one_to_zero_best: AttackConditionAnalysisNew

    @cached_property
    def data_struct_for_histogram_plotter(
        self,
    ) -> tuple[tuple[pd.DataFrame, ...], ...]:
        return (
            (
                self.zero_to_one_first.examples_df,
                self.zero_to_one_best.examples_df,
            ),
            (
                self.one_to_zero_first.examples_df,
                self.one_to_zero_best.examples_df,
            ),
        )

    def data_struct_for_susceptibility_plotter(
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


class FullAttackResults:
    def __init__(
        self,
        success_summary: ads.TrainerSuccessSummary,
    ):
        self._dataset = success_summary.dataset
        self.all_attacks_df = success_summary.all_attacks_df
        self.successful_attack_df = success_summary.successful_attacks_df
        self._first_examples_raw = success_summary.first_examples
        self._best_examples_raw = success_summary.best_examples
        self.first_perts_summary = success_summary.perts_summary_first
        self.best_perts_summary = success_summary.perts_summary_best

    @cached_property
    def attacked_features_padded(self) -> np.array:
        torch_attacked_features_padded = pad_sequence(
            self._dataset[:][1], batch_first=True
        )[self.all_attacks_df.dataset_index]
        return np.array(torch_attacked_features_padded)

    @cached_property
    def padded_features_success(self) -> np.array:
        return self.attacked_features_padded[
            self.successful_attack_df["attacked_samples_index"], :, :
        ]

    @cached_property
    def _orig_labels(self) -> np.array:
        return np.array(self._dataset[:][2])[self.all_attacks_df.dataset_index]

    # TODO replace all downstream refs to self.attacked_samples_df w/
    #  self.all_attacks_df then remove self.attacked_samples_df
    # @cached_property
    # def attacked_samples_df(self) -> pd.DataFrame:
    #     return self.all_attacks_df

    @cached_property
    def first_examples_padded_perts(self) -> np.array:
        return np.array(self._first_examples_raw.perturbations)[
            self.successful_attack_df["attacked_samples_index"]
        ]

    @cached_property
    def best_examples_padded_perts(self) -> np.array:
        return np.array(self._best_examples_raw.perturbations)[
            self.successful_attack_df["attacked_samples_index"]
        ]

    def build_examples_summary_df(
        self,
        example_type: RecordedExampleType,
    ):
        examples_dispatch = {
            RecordedExampleType.FIRST: self._first_examples_raw,
            RecordedExampleType.BEST: self._best_examples_raw,
        }

        perts_summary_dispatch = {
            RecordedExampleType.FIRST: self.first_perts_summary,
            RecordedExampleType.BEST: self.best_perts_summary,
        }

        trainer_examples = examples_dispatch[example_type]
        perts_summary_df = perts_summary_dispatch[example_type].df

        example_summary_data_array = np.stack(
            (
                self.all_attacks_df.dataset_index[
                    self.successful_attack_df["attacked_samples_index"]
                ],
                self.successful_attack_df["attacked_samples_index"],
                self._orig_labels[
                    self.successful_attack_df["attacked_samples_index"]
                ],
                np.array(trainer_examples.epochs)[
                    self.successful_attack_df["attacked_samples_index"]
                ],
                np.array(trainer_examples.losses)[
                    self.successful_attack_df["attacked_samples_index"]
                ],
            ),
            axis=1,
        )

        example_summary_df = pd.DataFrame(
            data=example_summary_data_array,
            columns=[
                "dataset_index",
                "attacked_samples_index",
                "orig_label",
                "epoch_found",
                "loss",
            ],
        ).astype(
            dtype={
                "dataset_index": "int",
                "attacked_samples_index": "int",
                "orig_label": "int",
                "epoch_found": "int",
                "loss": "float32",
            }
        )

        return pd.concat([example_summary_df, perts_summary_df], axis=1)

    @cached_property
    def first_examples_df(self) -> pd.DataFrame:
        return self.build_examples_summary_df(
            example_type=RecordedExampleType.FIRST
        )

    @cached_property
    def best_examples_df(self) -> pd.DataFrame:
        return self.build_examples_summary_df(
            example_type=RecordedExampleType.BEST
        )

    def get_attack_condition_data(
        self,
        seq_length: int,
        example_type: RecordedExampleType,
        orig_label: int = None,
        min_num_perts: int = None,
        max_num_perts: int = None,
    ) -> AttackConditionAnalysisNew:
        example_type_dispatch = {
            RecordedExampleType.FIRST: self.first_examples_df,
            RecordedExampleType.BEST: self.best_examples_df,
        }

        pert_summary_dispatch = {
            RecordedExampleType.FIRST: self.first_perts_summary,
            RecordedExampleType.BEST: self.best_perts_summary,
        }

        starting_df = example_type_dispatch[example_type]
        filtered_df = starting_df[starting_df["seq_length"] == seq_length]
        if orig_label is not None:
            filtered_df = filtered_df[filtered_df["orig_label"] == orig_label]
        if min_num_perts is not None:
            filtered_df = filtered_df[
                filtered_df["num_perts"] >= min_num_perts
            ]
        if max_num_perts is not None:
            filtered_df = filtered_df[
                filtered_df["num_perts"] <= max_num_perts
            ]

        perts = pert_summary_dispatch[example_type].padded_perts[
            filtered_df.index, :, :
        ]

        return AttackConditionAnalysisNew(examples_df=filtered_df, perts=perts)

    @cached_property
    def zto_first_examples_df(self) -> pd.DataFrame:
        return self.first_examples_df[
            self.first_examples_df["orig_label"] == 0
        ]

    @cached_property
    def otz_first_examples_df(self) -> pd.DataFrame:
        return self.first_examples_df[
            self.first_examples_df["orig_label"] == 1
        ]

    @cached_property
    def zto_best_examples_df(self) -> pd.DataFrame:
        return self.best_examples_df[self.best_examples_df["orig_label"] == 0]

    @cached_property
    def otz_best_examples_df(self) -> pd.DataFrame:
        return self.best_examples_df[self.best_examples_df["orig_label"] == 1]
