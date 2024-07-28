from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
import lstm_adversarial_attack.msgspec_io as mio
import lstm_adversarial_attack.preprocess.encode_decode_structs as eds
from lstm_adversarial_attack.config import CONFIG_READER
import lstm_adversarial_attack.path_searches as ps


class AttackSusceptibilityMetrics:
    """
    Computes susceptibiliy metrics using perturbations that generated
    adversarial example_data. All perturbations must be from samples of same
    input sequence length.
    """

    def __init__(
        self,
        perts: np.array,
        measurement_labels: tuple[str] = None,
        preprocess_id: str = None,
    ):
        """
        :param perts: perturbations that resulted in adversarial example_data.
        Changing value alog axis 0 corresponds to changing sample.
        :param measurement_labels: decodes measurement id to measurement name
        (length of measurement_labels must be same as size of perts along
        axis 2)
        """
        self.perts = perts

        # TODO clean up measurement labels retrieval
        preprocess_output_root = Path(
            CONFIG_READER.read_path("preprocess.output_root")
        )
        if preprocess_id is None:
            preprocess_id = ps.get_latest_sequential_child_dirname(
                root_dir=preprocess_output_root
            )

        if measurement_labels is None:
            measurement_labels_path = (
                preprocess_output_root
                / preprocess_id
                / "5_feature_finalizer"
                / "measurement_col_names.json"
            )
            measurement_labels_dto = mio.MsgspecIO(
                msgspec_struct_type=eds.MeasurementColumnNames
            ).import_to_struct(path=measurement_labels_path)

        self.measurement_labels = measurement_labels_dto.data

    @cached_property
    def _abs_perts(self) -> np.array:
        """
        Absolute value of each element in original .perts array
        :return: 3d array of floats
        """
        return np.abs(self.perts)

    @cached_property
    def _gmp_ij(self) -> np.array:
        """
        Global Maximum Perturbation as defined by Sun et al.
        :return: 2d array of floats
        """
        return np.max(self._abs_perts, axis=0)

    @cached_property
    def _gap_ij(self) -> np.array:
        """
        Global Average Perturbation as defined by Sun et al.
        :return: 2d array of floats
        """
        return np.sum(self._abs_perts) / self.perts.shape[0]
        # return self._abs_perts.mean(axis=0)

    @cached_property
    def _ganzp_ij(self) -> np.ma.MaskedArray:
        """
        Global Average Non Zero Perturbation magnitude. Seems like a better
        metric than Sun et al's GAP.
        :return: 2d array of floats
        """
        masked_perts = np.ma.masked_equal(self._abs_perts, 0)
        return masked_perts.mean(axis=0)

    @cached_property
    def ganzp_ij(self) -> pd.DataFrame:
        """
        self._ganzp_ij in Dataframe form, w/ self.measurement_labels as col
        names
        :return:  2d array of floats
        """
        return pd.DataFrame(
            data=np.ma.filled(self._ganzp_ij, 0),
            columns=self.measurement_labels,
        )

    @cached_property
    def _gpp_ij(self) -> np.array:
        """
        Global Perturbation Probability as defined by Sun et al
        :return: 2d array of floats
        """
        return np.count_nonzero(self.perts, axis=0) / self.perts.shape[0]

    @cached_property
    def gpp_ij(self) -> pd.DataFrame:
        """
        self._gpp_ij in dataframe form w/ self.measurement_labels as col names
        :return: Pandas Dataframe
        """

        return pd.DataFrame(data=self._gpp_ij, columns=self.measurement_labels)

    @cached_property
    def _s_ij(self) -> np.array:
        """
        Susceptibility scores as defined by Sun et al. NOTE: Gives high score
        to elements with large perturbations
        :return: 2d array of floats
        """
        return self._gmp_ij * self._gpp_ij

    @cached_property
    def _s_j(self) -> np.array:
        """
        Cumulative Susceptibility scores as defined by Sun et al. (sum of
        susceptibility score for each lab / vital sign measurement)
        :return: 1d array of floats
        """
        return np.sum(self._s_ij, axis=0)

    @cached_property
    def _sensitivity_ij(self) -> np.ma.masked_array:
        """
        Sensitivity score. Unlike Susceptibility, this is higher for elements
        with small example-producing perturbations.
        :rtype: 2d masked array of floats
        """
        return self._gpp_ij / self._ganzp_ij

    @cached_property
    def sensitivity_ij(self) -> pd.DataFrame:
        """
        self._sensitivity_ij as a dataframe w/ self.measurement labels as col
        names
        :return: Pandas dataframe
        """
        unmasked_data = np.ma.filled(self._sensitivity_ij, 0)
        return pd.DataFrame(
            data=unmasked_data, columns=self.measurement_labels
        )

    @cached_property
    def _sensitivity_j(self) -> np.ma.masked_array:
        """
        Cumulative sensitivity
        :return: 1d array of floats
        """
        return np.sum(self._sensitivity_ij, axis=0)

    @cached_property
    def sensitivity_j(self) -> pd.Series:
        """
        self._sensitivity_j as Pandas Series w/ self.measurement_labels as
        value names.
        :return: Pandas Series
        """
        unmasked_data = np.ma.filled(self._sensitivity_j, 0)
        return pd.Series(data=unmasked_data, index=self.measurement_labels)

    @cached_property
    def _num_nonzero_elements(self) -> np.array:
        """
        Number of non-zero elements in the perturbation subarray corresponding
        to each example
        :return: 1d array of ints
        """
        return np.sum(self.perts != 0, axis=(1, 2))
