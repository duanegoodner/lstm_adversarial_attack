from functools import cached_property

import numpy as np
import pandas as pd
import lstm_adversarial_attack.config_paths as cfg_paths
import lstm_adversarial_attack.resource_io as rio


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

    @cached_property
    def _abs_perts(self) -> np.array:
        return np.abs(self.perts)

    @cached_property
    def _gmp_ij(self) -> np.array:
        return np.max(self._abs_perts, axis=0)

    @cached_property
    def _gap_ij(self) -> np.array:
        return np.sum(self._abs_perts) / self.perts.shape[0]
        # return self._abs_perts.mean(axis=0)

    @cached_property
    def _ganzp_ij(self) -> np.ma.MaskedArray:
        masked_perts = np.ma.masked_equal(self._abs_perts, 0)
        return masked_perts.mean(axis=0)

    @cached_property
    def ganzp_ij(self) -> pd.DataFrame:
        return pd.DataFrame(
            data=np.ma.filled(self._ganzp_ij, 0),
            columns=self.measurement_labels,
        )

    @cached_property
    def _gpp_ij(self) -> np.array:
        return np.count_nonzero(self.perts, axis=0) / self.perts.shape[0]

    @cached_property
    def gpp_ij(self) -> pd.DataFrame:
        return pd.DataFrame(data=self._gpp_ij, columns=self.measurement_labels)

    @cached_property
    def _s_ij(self) -> np.array:
        return self._gmp_ij * self._gpp_ij

    @cached_property
    def _s_j(self) -> np.array:
        return np.sum(self._s_ij, axis=0)

    @cached_property
    def _sensitivity_ij(self) -> np.ma.masked_array:
        return self._gpp_ij / self._ganzp_ij

    @cached_property
    def sensitivity_ij(self) -> pd.DataFrame:
        unmasked_data = np.ma.filled(self._sensitivity_ij, 0)
        return pd.DataFrame(
            data=unmasked_data, columns=self.measurement_labels
        )

    @cached_property
    def _sensitivity_j(self) -> np.ma.masked_array:
        return np.sum(self._sensitivity_ij, axis=0)

    @cached_property
    def sensitivity_j(self) -> pd.Series:
        unmasked_data = np.ma.filled(self._sensitivity_j, 0)
        return pd.Series(data=unmasked_data, index=self.measurement_labels)

    @cached_property
    def _num_nonzero_elements(self) -> np.array:
        return np.sum(self.perts != 0, axis=(1, 2))
