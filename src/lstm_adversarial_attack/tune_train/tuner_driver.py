import optuna
import sys
import torch
from pathlib import Path
from typing import Callable

sys.path.append(str(Path(__file__).parent.parent.parent))
import lstm_adversarial_attack.config_paths as cfg_path
import lstm_adversarial_attack.resource_io as rio
import lstm_adversarial_attack.tune_train.hyperparameter_tuner as htu
import lstm_adversarial_attack.tune_train.tuner_helpers as tuh
import lstm_adversarial_attack.x19_mort_general_dataset as xmd


class TunerDriver:
    """
    Instantiates and runs a HyperparameterTuner
    """

    def __init__(
        self,
        device: torch.device,
        collate_fn: Callable = xmd.x19m_collate_fn,
        continue_study_path: Path = None,
        output_dir: Path = None,
    ):
        self.device = device
        self.collate_fn = collate_fn
        self.continue_study_path = continue_study_path
        if output_dir is None:
            output_dir = rio.create_timestamped_dir(parent_path=cfg_path.HYPERPARAMETER_OUTPUT_DIR)
        self.output_dir = output_dir
        self.tuning_ranges = tuh.X19MLSTMTuningRanges()

        # self.tuner = htu.HyperParameterTuner(
        #     device=device,
        #     dataset=xmd.X19MGeneralDataset.from_feature_finalizer_output(),
        #     collate_fn=collate_fn,
        #     tuning_ranges=tuh.X19MLSTMTuningRanges(),
        #     continue_study_path=continue_study_path,
        #     output_dir=output_dir
        # )

    def run(self, num_trials: int) -> optuna.Study:
        """
        Runs an optuna study for num_trials
        :param num_trials: number of trials to run
        :return: completed optuna study
        """
        driver_dict_output_path = rio.create_timestamped_filepath(
            parent_path=self.output_dir,
            file_extension="pickle",
            prefix="tuner_driver_dict_",
        )
        rio.ResourceExporter().export(
            resource=self.__dict__, path=driver_dict_output_path
        )
        tuner = htu.HyperParameterTuner(
            device=self.device,
            dataset=xmd.X19MGeneralDataset.from_feature_finalizer_output(),
            collate_fn=self.collate_fn,
            tuning_ranges=self.tuning_ranges,
            continue_study_path=self.continue_study_path,
            output_dir=self.output_dir,
        )
        # completed_study = self.tuner.tune(num_trials=num_trials)
        completed_study = tuner.tune(num_trials=num_trials)
        return completed_study

    def __call__(self, num_trials: int) -> optuna.Study:
        completed_study = self.run(num_trials=num_trials)
        return completed_study
