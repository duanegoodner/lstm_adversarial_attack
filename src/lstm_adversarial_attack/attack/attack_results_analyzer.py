import numpy as np
import pandas as pd
import torch
from pathlib import Path
import lstm_adversarial_attack.attack.attack_result_data_structs as ads
import lstm_adversarial_attack.attack.attack_summary as asu
import lstm_adversarial_attack.config_settings as cfg_set
import lstm_adversarial_attack.config_paths as cfg_paths
import lstm_adversarial_attack.resource_io as rio


# def calc_gmp_ij(perts: torch.tensor) -> torch.tensor:
#     return torch.max(torch.abs(perts), dim=0).values


def calc_gmp_ij(perts: np.array) -> np.array:
    return np.max(np.abs(perts), axis=0)


# def calc_gap_ij(perts: torch.tensor) -> torch.tensor:
#     return torch.sum(torch.abs(perts), dim=0) / perts.shape[0]


def calc_gap_ij(perts: np.array) -> np.array:
    return np.sum(np.abs(perts), axis=0) / perts.shape[0]


# def calc_gpp_ij(perts: torch.tensor) -> torch.tensor:
#     return torch.count_nonzero(perts, dim=0) / perts.shape[0]


def calc_gpp_ij(perts: np.array) -> np.array:
    return np.count_nonzero(perts, axis=0) / perts.shape[0]


# def calc_s_ij(gmp: torch.tensor, gpp: torch.tensor) -> torch.tensor:
#     return torch.mul(gmp, gpp)


def calc_s_ij(gmp: np.array, gpp: np.array) -> np.array:
    return gmp * gpp


# def calc_s_j(s_ij: torch.tensor) -> torch.tensor:
#     return torch.sum(s_ij, dim=0)


def calc_s_j(s_ij: np.array) -> np.array:
    return np.sum(s_ij, axis=0)


class AttackSusceptibilityMetrics:
    def __init__(self, perts: np.array, measurement_labels: tuple[str] = None):
        self.perts = perts
        if measurement_labels is None:
            measurement_labels = (
                rio.ResourceImporter().import_pickle_to_object(
                    path=cfg_paths.PREPROCESS_OUTPUT_DIR
                    / "measurement_col_names.pickle"
                )
            )
        self.measurement_labels = measurement_labels
        if perts.shape[0] != 0:
            self._gmp_ij = calc_gmp_ij(perts=perts)
            self._gap_ij = calc_gap_ij(perts=perts)
            self._gpp_ij = calc_gpp_ij(perts=perts)
            self._s_ij = calc_s_ij(gmp=self._gmp_ij, gpp=self._gpp_ij)
            self._s_j = calc_s_j(s_ij=self._s_ij)
            self.gmp_ij = pd.DataFrame(
                data=self._gmp_ij, columns=self.measurement_labels
            )
            self.gap_ij = pd.DataFrame(
                data=self._gap_ij, columns=self.measurement_labels
            )
            self.gpp_ij = pd.DataFrame(
                data=self._gpp_ij, columns=self.measurement_labels
            )
            self.s_ij = pd.DataFrame(
                data=self._s_ij, columns=self.measurement_labels
            )
            self.s_j = pd.Series(data=self._s_j, index=self.measurement_labels)
        else:
            self._gmp_ij = None
            self._gap_ij = None
            self._gpp_ij = None
            self._s_ij = None
            self._s_j = None

    # @property
    # def gap_ij(self) -> pd.DataFrame:
    #     if self._gap_ij is not None:
    #         return pd.DataFrame(
    #             data=self._gap_ij, columns=self.measurement_labels
    #         )


class SusceptibilityCalculator:
    def __init__(self, attack_summary: asu.AttackResults):
        self._attack_summary = attack_summary

    @property
    def _examples_df_dispatch(self) -> dict:
        return {
            asu.RecordedExampleType.FIRST: (
                self._attack_summary.best_examples_df
            ),
            asu.RecordedExampleType.BEST: (
                self._attack_summary.first_examples_df
            ),
        }

    @property
    def _perts_summary_dispatch(self) -> dict:
        return {
            asu.RecordedExampleType.FIRST: (
                self._attack_summary.best_perts_summary
            ),
            asu.RecordedExampleType.BEST: (
                self._attack_summary.first_perts_summary
            ),
        }

    def _get_filtered_examples_df(
        self,
        example_type: asu.RecordedExampleType,
        seq_length: int,
        orig_label: int,
        min_num_perts: int = None,
        max_num_perts: int = None,
    ):
        orig_df = self._examples_df_dispatch[example_type]
        filtered_df = orig_df[
            (orig_df["seq_length"] == seq_length)
            & (orig_df["orig_label"] == orig_label)
        ]

        if min_num_perts is not None:
            filtered_df = filtered_df[
                filtered_df["num_perts"] >= min_num_perts
            ]

        if max_num_perts is not None:
            filtered_df = filtered_df[
                filtered_df["num_perts"] <= max_num_perts
            ]

        return filtered_df

    def get_susceptibility_metrics(
        self,
        example_type: asu.RecordedExampleType,
        seq_length: int,
        orig_label: int,
        min_num_perts: int = None,
        max_num_perts: int = None,
    ) -> AttackSusceptibilityMetrics:
        filtered_df = self._get_filtered_examples_df(
            example_type=example_type,
            seq_length=seq_length,
            orig_label=orig_label,
            min_num_perts=min_num_perts,
            max_num_perts=max_num_perts,
        )

        perts_summary = self._perts_summary_dispatch[example_type]

        metrics = AttackSusceptibilityMetrics(
            perts=perts_summary.padded_perts[filtered_df.index, :, :]
        )

        return metrics
