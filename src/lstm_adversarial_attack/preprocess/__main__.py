import sys
from pathlib import Path
import time

sys.path.append(str(Path(__file__).parent.parent.parent))

import lstm_adversarial_attack.config_paths as cfp
import lstm_adversarial_attack.preprocess.admission_list_builder as alb
import lstm_adversarial_attack.preprocess.feature_builder as fb
import lstm_adversarial_attack.preprocess.feature_finalizer as ff
import lstm_adversarial_attack.preprocess.icustay_measurement_merger as imm
import lstm_adversarial_attack.preprocess.prefilter as prf
import lstm_adversarial_attack.preprocess.preprocessor as ppr
import lstm_adversarial_attack.preprocess.resource_data_structs as rds


def main():
    prefilter_info = ppr.ModuleInfo(
        module_name="prefilter",
        module_constructor=prf.Prefilter,
        resources_constructor=rds.PrefilterResources,
        settings_constructor=prf.PrefilterSettings,
        individual_resources_info=[
            rds.FileResourceInfo(
                key="icustay",
                path=cfp.PREFILTER_INPUT_FILES["icustay"],
                # path=cfp.DB_OUTPUT_DIR / "icustay_500.csv",
                constructor=rds.IncomingDataFrame,
            ),
            rds.FileResourceInfo(
                key="bg",
                path=cfp.PREFILTER_INPUT_FILES["bg"],
                constructor=rds.IncomingDataFrame,
            ),
            rds.FileResourceInfo(
                key="vital",
                path=cfp.PREFILTER_INPUT_FILES["vital"],
                constructor=rds.IncomingDataFrame,
            ),
            rds.FileResourceInfo(
                key="lab",
                path=cfp.PREFILTER_INPUT_FILES["lab"],
                constructor=rds.IncomingDataFrame,
            ),
        ],
    )

    combiner_info = ppr.ModuleInfo(
        module_name="measurement_merger",
        module_constructor=imm.ICUStayMeasurementMerger,
        resources_constructor=rds.ICUStayMeasurementMergerResources,
        settings_constructor=imm.ICUStayMeasurementMergerSettings,
        individual_resources_info=[
            rds.PoolResourceInfo(
                key="prefiltered_icustay", constructor=rds.IncomingDataFrame
            ),
            rds.PoolResourceInfo(
                key="prefiltered_bg", constructor=rds.IncomingDataFrame
            ),
            rds.PoolResourceInfo(
                key="prefiltered_vital", constructor=rds.IncomingDataFrame
            ),
            rds.PoolResourceInfo(
                key="prefiltered_lab", constructor=rds.IncomingDataFrame
            ),
        ],
    )

    list_builder_info = ppr.ModuleInfo(
        module_name="admission_list_builder",
        module_constructor=alb.AdmissionListBuilder,
        resources_constructor=rds.AdmissionListBuilderResources,
        settings_constructor=alb.AdmissionListBuilderSettings,
        individual_resources_info=[
            rds.PoolResourceInfo(
                key="icustay_bg_lab_vital", constructor=rds.IncomingDataFrame
            )
        ],
    )

    feature_builder_info = ppr.ModuleInfo(
        module_name="feature_builder",
        module_constructor=fb.FeatureBuilder,
        resources_constructor=rds.FeatureBuilderResources,
        settings_constructor=fb.FeatureBuilderSettings,
        individual_resources_info=[
            rds.PoolResourceInfo(
                key="full_admission_list", constructor=rds.IncomingFullAdmissionData
            ),
            rds.PoolResourceInfo(
                key="bg_lab_vital_summary_stats",
                constructor=rds.IncomingDataFrame,
            ),
        ],
        # output_info=rds.FeatureBuilderOutputConstructors(
        #     processed_admission_list=rds.OutgoingPreprocessPickle
        # )
    )

    feature_finalizer_info = ppr.ModuleInfo(
        module_name="feature_finalizer",
        module_constructor=ff.FeatureFinalizer,
        resources_constructor=rds.FeatureFinalizerResources,
        settings_constructor=ff.FeatureFinalizerSettings,
        individual_resources_info=[
            rds.PoolResourceInfo(
                key="processed_admission_list",
                constructor=rds.IncomingFullAdmissionData,
            )
        ],
        save_output=True,
    )
    modules_info = [
        prefilter_info,
        combiner_info,
        list_builder_info,
        feature_builder_info,
        feature_finalizer_info,
    ]

    preprocessor = ppr.Preprocessor(modules_info=modules_info, save_checkpoints=True)
    return preprocessor.run_all_modules()


if __name__ == "__main__":
    start = time.time()
    result = main()
    end = time.time()

    print(f"total time = {end - start}")
