import time

import lstm_adversarial_attack.preprocess.new_admission_list_builder as alb
import lstm_adversarial_attack.preprocess.new_feature_builder as fb
import lstm_adversarial_attack.preprocess.new_feature_finalizer as ff
import lstm_adversarial_attack.preprocess.new_icustay_measurement_merger as imm
import lstm_adversarial_attack.preprocess.new_prefilter as prf
import lstm_adversarial_attack.preprocess.new_preprocessor as ppr
import lstm_adversarial_attack.preprocess.preprocess_input_classes as pic


def main():
    preprocessor = ppr.NewPreprocessor(
        prefilter=prf.NewPrefilter,
        icustay_measurement_combiner=imm.NewICUStayMeasurementMerger,
        admission_list_builder=alb.NewAdmissionListBuilder,
        feature_builder=fb.NewFeatureBuilder,
        feature_finalizer=ff.NewFeatureFinalizer,
        inputs=pic.PrefilterResourceRefs(),
        save_checkpoints=False
    )
    return preprocessor.preprocess()


if __name__ == "__main__":
    result = main()
