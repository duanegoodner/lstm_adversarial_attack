import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import lstm_aa.preprocess.prefilter as prf
import lstm_aa.preprocess.icustay_measurement_combiner as imc
import lstm_aa.preprocess.sample_list_builder as slb
import lstm_aa.preprocess.feature_builder as fb
import lstm_aa.preprocess.feature_finalizer as ff
import lstm_aa.preprocess.preprocess_resource as pr


class Preprocessor:
    def __init__(
        self,
        prefilter=prf.Prefilter(),
        measurement_combiner=imc.ICUStayMeasurementCombiner(),
        hadm_list_builder=slb.FullAdmissionListBuilder(),
        feature_builder=fb.FeatureBuilder(),
        feature_finalizer=ff.FeatureFinalizer(),
    ):
        self._prefilter = prefilter
        self._measurement_combiner = measurement_combiner
        self._hadm_list_builder = hadm_list_builder
        self._feature_builder = feature_builder
        self._feature_finalizer = feature_finalizer
        self._saved_files = []

    def preprocess(self) -> dict[str, pr.DataResource]:
        start = time.time()

        print("Starting Prefilter")
        prefilter_exports = self._prefilter()
        self._saved_files.append(prefilter_exports)
        print("Done with Prefilter\n")

        print("Starting ICUStatyMeasurementCombiner")
        measurement_combiner_exports = self._measurement_combiner()
        self._saved_files.append(measurement_combiner_exports)
        print("Done with ICUStatyMeasurementCombiner\n")

        print("Starting FullAdmissionListBuilder")
        hadm_list_builder_exports = self._hadm_list_builder()
        self._saved_files.append(hadm_list_builder_exports)
        print("Done with FullAdmissionListBuilder\n")

        print("Starting FeatureBuilder")
        feature_builder_exports = self._feature_builder()
        self._saved_files.append(feature_builder_exports)
        print("Done with FeatureBuilder\n")

        print("Starting FeatureFinalizer")
        feature_finalizer_exports = self._feature_finalizer()
        self._saved_files.append(feature_finalizer_exports)
        print("Done with FeatureFinalizer\n")

        end = time.time()
        print(
            "All Done!\nTotal preprocessing time ="
            f" {(end - start):.2f} seconds"
        )

        preprocessed_resources = {}
        for key, val in feature_finalizer_exports.items():
            preprocessed_resources[key] = pr.DataResource(
                path=val.path, py_object_type=val.data_type
            )

        return preprocessed_resources


if __name__ == "__main__":
    preprocessor = Preprocessor()
    preprocessor.preprocess()
