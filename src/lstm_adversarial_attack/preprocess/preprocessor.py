import pprint
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import lstm_adversarial_attack.preprocess.prefilter as prf
import lstm_adversarial_attack.preprocess.icustay_measurement_combiner as imc
import lstm_adversarial_attack.preprocess.sample_list_builder as slb
import lstm_adversarial_attack.preprocess.feature_builder as fb
import lstm_adversarial_attack.preprocess.feature_finalizer as ff
import lstm_adversarial_attack.preprocess.preprocess_resource as pr


class Preprocessor:
    def __init__(
        self,
        prefilter=prf.Prefilter(),
        measurement_combiner=imc.ICUStayMeasurementCombiner(),
        hadm_list_builder=slb.FullAdmissionListBuilder(),
        feature_builder=fb.FeatureBuilder(),
        feature_finalizer=ff.FeatureFinalizer(),
    ):
        self.preprocess_modules = [
            prefilter,
            measurement_combiner,
            hadm_list_builder,
            feature_builder,
            feature_finalizer,
        ]
        self._prefilter = prefilter
        self._measurement_combiner = measurement_combiner
        self._hadm_list_builder = hadm_list_builder
        self._feature_builder = feature_builder
        self._feature_finalizer = feature_finalizer
        self._saved_files = []

    def preprocess(self) -> list[dict[str, pr.DataResource]]:
        start = time.time()

        for module_idx in range(len(self.preprocess_modules)):
            module = self.preprocess_modules[module_idx]
            print(
                f"\nRunning preprocess module {module_idx + 1} of"
                f" {len(self.preprocess_modules)}: {module.name}"
            )
            print("Incoming resources:")
            for ref in module.incoming_resource_refs.__dict__.values():
                print(f"{ref}")
            exported_resources = module()
            self._saved_files.append(exported_resources)
            print(f"Done with {module.name}. Results saved to:")
            for item in exported_resources.values():
                print(f"{item}")

        end = time.time()
        print(
            "\nAll preprocess modules complete.\nTotal preprocessing time ="
            f" {(end - start):.2f} seconds"
        )

        return self._saved_files
