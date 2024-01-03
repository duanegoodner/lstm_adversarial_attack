import sys
from pathlib import Path
import time

sys.path.append(str(Path(__file__).parent.parent.parent))

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
        default_data_source_type=rds.DataSourceType.FILE
    )

    combiner_info = ppr.ModuleInfo(
        module_name="measurement_merger",
        module_constructor=imm.ICUStayMeasurementMerger,
        resources_constructor=rds.ICUStayMeasurementMergerResources,
        settings_constructor=imm.ICUStayMeasurementMergerSettings,
        default_data_source_type=rds.DataSourceType.POOL
    )

    list_builder_info = ppr.ModuleInfo(
        module_name="admission_list_builder",
        module_constructor=alb.AdmissionListBuilder,
        resources_constructor=rds.AdmissionListBuilderResources,
        settings_constructor=alb.AdmissionListBuilderSettings,
        default_data_source_type=rds.DataSourceType.POOL
    )

    feature_builder_info = ppr.ModuleInfo(
        module_name="feature_builder",
        module_constructor=fb.FeatureBuilder,
        resources_constructor=rds.FeatureBuilderResources,
        settings_constructor=fb.FeatureBuilderSettings,
        default_data_source_type=rds.DataSourceType.POOL
    )

    feature_finalizer_info = ppr.ModuleInfo(
        module_name="feature_finalizer",
        module_constructor=ff.FeatureFinalizer,
        resources_constructor=rds.FeatureFinalizerResources,
        settings_constructor=ff.FeatureFinalizerSettings,
        default_data_source_type=rds.DataSourceType.POOL
    )
    modules_info = [
        prefilter_info,
        combiner_info,
        list_builder_info,
        feature_builder_info,
        feature_finalizer_info,
    ]

    preprocessor = ppr.Preprocessor(modules_info=modules_info,
                                    save_checkpoints=True)
    return preprocessor.run_all_modules()


if __name__ == "__main__":
    start = time.time()
    result = main()
    end = time.time()

    print(f"total time = {end - start}")
