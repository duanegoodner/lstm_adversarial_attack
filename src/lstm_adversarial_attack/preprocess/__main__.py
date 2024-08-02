import argparse
import sys
from pathlib import Path
import time
from typing import Any

sys.path.append(str(Path(__file__).parent.parent.parent))

import lstm_adversarial_attack.preprocess.admission_list_builder as alb
import lstm_adversarial_attack.preprocess.feature_builder as fb
import lstm_adversarial_attack.preprocess.feature_finalizer as ff
import lstm_adversarial_attack.preprocess.icustay_measurement_merger as imm
import lstm_adversarial_attack.preprocess.prefilter as prf
import lstm_adversarial_attack.preprocess.preprocessor as ppr
import lstm_adversarial_attack.preprocess.resource_data_structs as rds
import lstm_adversarial_attack.utils.session_id_generator as sig
from lstm_adversarial_attack.config import CONFIG_READER


def main(db_result_id: str = None) -> dict | dict[str, Any]:
    preprocess_id = sig.generate_session_id()

    if db_result_id is None:
        db_output_parent = CONFIG_READER.read_path("db.output_root")
        db_result_id = str(
            max(
                [
                    int(item.name)
                    for item in list(Path(db_output_parent).iterdir())
                    if item.is_dir()
                ]
            )
        )


    prefilter_info = ppr.ModuleInfo(
        resource_collection_ids={"db": db_result_id, "preprocess": preprocess_id},
        module_name="prefilter",
        module_constructor=prf.Prefilter,
        resources_constructor=rds.PrefilterResources,
        settings_constructor=prf.PrefilterSettings,
        default_data_source_type=rds.DataSourceType.FILE,
    )

    combiner_info = ppr.ModuleInfo(
        resource_collection_ids={"preprocess": preprocess_id},
        module_name="measurement_merger",
        module_constructor=imm.ICUStayMeasurementMerger,
        resources_constructor=rds.ICUStayMeasurementMergerResources,
        settings_constructor=imm.ICUStayMeasurementMergerSettings,
        default_data_source_type=rds.DataSourceType.POOL,
    )

    list_builder_info = ppr.ModuleInfo(
        resource_collection_ids={"preprocess": preprocess_id},
        module_name="admission_list_builder",
        module_constructor=alb.AdmissionListBuilder,
        resources_constructor=rds.AdmissionListBuilderResources,
        settings_constructor=alb.AdmissionListBuilderSettings,
        default_data_source_type=rds.DataSourceType.POOL,
    )

    feature_builder_info = ppr.ModuleInfo(
        resource_collection_ids={"preprocess": preprocess_id},
        module_name="feature_builder",
        module_constructor=fb.FeatureBuilder,
        resources_constructor=rds.FeatureBuilderResources,
        settings_constructor=fb.FeatureBuilderSettings,
        default_data_source_type=rds.DataSourceType.POOL,
    )

    feature_finalizer_info = ppr.ModuleInfo(
        resource_collection_ids={"preprocess": preprocess_id},
        module_name="feature_finalizer",
        module_constructor=ff.FeatureFinalizer,
        resources_constructor=rds.FeatureFinalizerResources,
        settings_constructor=ff.FeatureFinalizerSettings,
        default_data_source_type=rds.DataSourceType.POOL,
    )
    modules_info = [
        prefilter_info,
        combiner_info,
        list_builder_info,
        feature_builder_info,
        feature_finalizer_info,
    ]

    preprocessor = ppr.Preprocessor(
        preprocess_id=preprocess_id, modules_info=modules_info, save_checkpoints=True
    )
    return preprocessor.run_all_modules()


if __name__ == "__main__":
    start = time.time()

    parser = argparse.ArgumentParser(
        description="Takes data output from database query, and runs through preprocess modules."
    )
    parser.add_argument(
        "-d",
        "--db_result_id",
        type=str,
        action="store",
        nargs="?",
        help="ID of database query session to take output from. Defaults to latest query.",
    )

    args_namespace = parser.parse_args()

    result = main(**args_namespace.__dict__)
    end = time.time()

    print(f"Total preprocess time = {end - start}")
