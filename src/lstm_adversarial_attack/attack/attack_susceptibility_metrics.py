from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd
import lstm_adversarial_attack.attack.attack_summary as asu
import lstm_adversarial_attack.config_paths as cfg_paths
import lstm_adversarial_attack.resource_io as rio


def calc_gmp_ij(perts: np.array) -> np.array:
    return np.max(np.abs(perts), axis=0)


def calc_gap_ij(perts: np.array) -> np.array:
    return np.sum(np.abs(perts), axis=0) / perts.shape[0]


def calc_ganzp_ij(perts: np.array) -> np.ma.MaskedArray:
    masked_perts = np.ma.masked_equal(np.abs(perts), 0)
    return masked_perts.mean(axis=0)


def calc_gpp_ij(perts: np.array) -> np.array:
    return np.count_nonzero(perts, axis=0) / perts.shape[0]


def calc_s_ij(gmp: np.array, gpp: np.array) -> np.array:
    return gmp * gpp


def calc_s_ganzp_ij(
    ganzp: np.ma.MaskedArray, gpp: np.array
) -> np.ma.MaskedArray:
    return gpp / ganzp


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
            self._ganzp_ij = calc_ganzp_ij(perts=perts)
            self._gpp_ij = calc_gpp_ij(perts=perts)
            self._s_ij = calc_s_ij(gmp=self._gmp_ij, gpp=self._gpp_ij)
            self._s_j = calc_s_j(s_ij=self._s_ij)
            self.gmp_ij = pd.DataFrame(
                data=self._gmp_ij, columns=self.measurement_labels
            )
            self.gap_ij = pd.DataFrame(
                data=self._gap_ij, columns=self.measurement_labels
            )
            self.ganzp_ij = pd.DataFrame(
                data=np.ma.filled(self._ganzp_ij, 0),
                columns=self.measurement_labels,
            )
            self.gpp_ij = pd.DataFrame(
                data=self._gpp_ij, columns=self.measurement_labels
            )
            self.s_ij = pd.DataFrame(
                data=self._s_ij, columns=self.measurement_labels
            )
            self._s_ganzp_ij = calc_s_ganzp_ij(
                ganzp=self._ganzp_ij, gpp=self._gpp_ij
            )
            self.s_ganzp_ij = pd.DataFrame(
                data=np.ma.filled(self._s_ganzp_ij, 0),
                columns=self.measurement_labels,
            )
            self.s_j = pd.Series(data=self._s_j, index=self.measurement_labels)
        else:
            self._gmp_ij = None
            self._gap_ij = None
            self._gpp_ij = None
            self._s_ij = None
            self._s_j = None