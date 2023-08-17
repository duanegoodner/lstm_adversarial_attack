import time

import lstm_adversarial_attack.config_paths as cfp
import lstm_adversarial_attack.preprocess.new_admission_list_builder as alb
import lstm_adversarial_attack.preprocess.new_feature_builder as fb
import lstm_adversarial_attack.preprocess.new_feature_finalizer as ff
import lstm_adversarial_attack.preprocess.new_icustay_measurement_merger as imm
import lstm_adversarial_attack.preprocess.new_prefilter as prf
import lstm_adversarial_attack.preprocess.new_preprocessor as ppr
import lstm_adversarial_attack.preprocess.resource_data_structs as rds


def main():
    prefilter_info = ppr.ModuleInfo(
        module_constructor=prf.NewPrefilter,
        resources_constructor=rds.NewPrefilterResources,
        individual_resources_info=[
            rds.FileResourceInfo(
                key="icustay",
                path=cfp.PREFILTER_INPUT_FILES["icustay"],
                constructor=rds.IncomingCSVDataFrame
            ),
            rds.FileResourceInfo(
                key="bg",
                path=cfp.PREFILTER_INPUT_FILES["bg"],
                constructor=rds.IncomingCSVDataFrame
            ),
            rds.FileResourceInfo(
                key="vital",
                path=cfp.PREFILTER_INPUT_FILES["vital"],
                constructor=rds.IncomingCSVDataFrame
            ),
            rds.FileResourceInfo(
                key="lab",
                path=cfp.PREFILTER_INPUT_FILES["lab"],
                constructor=rds.IncomingCSVDataFrame
            )
        ]
    )

    combiner_info = ppr.ModuleInfo(
        module_constructor=imm.NewICUStayMeasurementMerger,
        resources_constructor=rds.NewICUStayMeasurementMergerResources,
        individual_resources_info=[
            rds.PoolResourceInfo(
                key="prefiltered_icustay",
                constructor=rds.IncomingFeatherDataFrame
            ),
            rds.PoolResourceInfo(
                key="prefiltered_bg",
                constructor=rds.IncomingFeatherDataFrame
            ),
            rds.PoolResourceInfo(
                key="prefiltered_vital",
                constructor=rds.IncomingFeatherDataFrame
            ),
            rds.PoolResourceInfo(
                key="prefiltered_lab",
                constructor=rds.IncomingFeatherDataFrame
            )
        ]
    )

    list_builder_info = ppr.ModuleInfo(
        module_constructor=alb.NewAdmissionListBuilder,
        resources_constructor=rds.NewAdmissionListBuilderResources,
        individual_resources_info=[
            rds.PoolResourceInfo(
                key="icustay_bg_lab_vital",
                constructor=rds.IncomingFeatherDataFrame
            )
        ],

    )

    feature_builder_info = ppr.ModuleInfo(
        module_constructor=fb.NewFeatureBuilder,
        resources_constructor=rds.NewFeatureBuilderResources,
        individual_resources_info=[
            rds.PoolResourceInfo(
                key="full_admission_list",
                constructor=rds.IncomingFullAdmissionData
            ),
            rds.PoolResourceInfo(
                key="bg_lab_vital_summary_stats",
                constructor=rds.IncomingFeatherDataFrame
            )
        ],
        # output_info=rds.NewFeatureBuilderOutputConstructors(
        #     processed_admission_list=rds.OutgoingPreprocessPickle
        # )
    )

    feature_finalizer_info = ppr.ModuleInfo(
        module_constructor=ff.NewFeatureFinalizer,
        resources_constructor=rds.NewFeatureFinalizerResources,
        individual_resources_info=[
            rds.PoolResourceInfo(
                key="processed_admission_list",
                constructor=rds.IncomingFullAdmissionData
            )
        ],
        save_output=True
    )
    modules_info = [
        prefilter_info,
        combiner_info,
        list_builder_info,
        feature_builder_info,
        feature_finalizer_info
    ]

    preprocessor = ppr.NewPreprocessor(
        modules_info=modules_info,
        save_checkpoints=True
    )
    return preprocessor.run_all_modules()

if __name__ == "__main__":
    start = time.time()
    result = main()
    end = time.time()

    print(f"total time = {end - start}")
