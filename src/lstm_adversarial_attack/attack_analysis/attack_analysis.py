from dataclasses import dataclass
from enum import Enum, auto
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import lstm_adversarial_attack.attack.attack_result_data_structs as ads
import lstm_adversarial_attack.attack_analysis.attack_susceptibility_metrics as asm
import lstm_adversarial_attack.config_paths as cfg_paths
import lstm_adversarial_attack.utils.path_searches as ps
from lstm_adversarial_attack.config import CONFIG_READER


class RecordedExampleType(Enum):
    FIRST = auto()
    BEST = auto()


@dataclass
class StandardDataFramesForPlotter:
    zero_to_one_first: pd.DataFrame
    zero_to_one_best: pd.DataFrame
    one_to_zero_first: pd.DataFrame
    one_to_zero_best: pd.DataFrame

    def __post_init__(self):
        assert (
            self.zero_to_one_first.columns == self.one_to_zero_best.columns
        ).all()
        assert (
            self.zero_to_one_first.columns == self.one_to_zero_first.columns
        ).all()
        assert (
            self.zero_to_one_first.columns == self.one_to_zero_best.columns
        ).all()


@dataclass
class ZerosOnesDFPair:
    zero_to_one: pd.DataFrame
    one_to_zero: pd.DataFrame


@dataclass
class AttackConditionSummary:
    """
    Container to dataframe with summary of first or best example_data and the
    perturbations corresponding to those example_data.
    :param examples_df: Pandas dataframe of form returned by
    FullAttackResults.get_condition_analysis()
    :param perts: perturbations that resulted in the adversarial example_data
    summarized in examples_df.
    """

    examples_df: pd.DataFrame
    perts: np.array

    def __post_init__(self):
        """
        Make sure number of example_data matches number of perturbation subarrays.
        """
        assert len(self.examples_df == self.perts.shape[0])

    @cached_property
    def susceptibility_metrics(self) -> asm.AttackSusceptibilityMetrics:
        """
        Calculates AttackSusceptibilityMetrics for the adversarial example_data.
        :return: an AttackSusceptibilityMetrics object
        """
        return asm.AttackSusceptibilityMetrics(perts=self.perts)


@dataclass
class StandardAttackConditionSummaries:
    """
    Holds results for standard adversarial example categories (from attacks)
    on single dataset, and provides methods to convert results into format
    expected by plotting classes.
    """

    zero_to_one_first: AttackConditionSummary
    zero_to_one_best: AttackConditionSummary
    one_to_zero_first: AttackConditionSummary
    one_to_zero_best: AttackConditionSummary
    all_attacked_zeros_df: pd.DataFrame
    all_attacked_ones_df: pd.DataFrame

    def __post_init__(self):
        unique_zeros_num_epochs = np.unique(
            self.all_attacked_zeros_df["num_epochs_run"]
        )
        unique_ones_num_epochs = np.unique(
            self.all_attacked_ones_df["num_epochs_run"]
        )
        assert len(unique_zeros_num_epochs) == len(unique_ones_num_epochs) == 1
        assert unique_zeros_num_epochs.item() == unique_ones_num_epochs.item()

    @property
    def num_epochs_run(self) -> int:
        return self.all_attacked_zeros_df["num_epochs_run"][0]

    @property
    def data_for_histogram_plotter(self) -> StandardDataFramesForPlotter:
        return StandardDataFramesForPlotter(
            zero_to_one_first=self.zero_to_one_first.examples_df,
            zero_to_one_best=self.zero_to_one_best.examples_df,
            one_to_zero_first=self.one_to_zero_first.examples_df,
            one_to_zero_best=self.one_to_zero_best.examples_df,
        )

    def data_for_susceptibility_plotter(
        self, metric
    ) -> StandardDataFramesForPlotter:
        return StandardDataFramesForPlotter(
            zero_to_one_first=getattr(
                self.zero_to_one_first.susceptibility_metrics, metric
            ),
            zero_to_one_best=getattr(
                self.zero_to_one_best.susceptibility_metrics, metric
            ),
            one_to_zero_first=getattr(
                self.one_to_zero_first.susceptibility_metrics, metric
            ),
            one_to_zero_best=getattr(
                self.one_to_zero_best.susceptibility_metrics, metric
            ),
        )

    @property
    def condition_dispatch(
        self,
    ) -> dict[tuple[RecordedExampleType, int], AttackConditionSummary]:
        return {
            (RecordedExampleType.FIRST, 0): self.zero_to_one_first,
            (RecordedExampleType.BEST, 0): self.zero_to_one_best,
            (RecordedExampleType.FIRST, 1): self.one_to_zero_first,
            (RecordedExampleType.BEST, 1): self.one_to_zero_best,
        }

    @property
    def all_attack_dispatch(self):
        return {0: self.all_attacked_zeros_df, 1: self.all_attacked_ones_df}

    def epochfound_cdf(
        self, orig_label: int, example_type: RecordedExampleType
    ) -> np.array:
        df = self.condition_dispatch[(example_type, orig_label)].examples_df
        counts_arr, bins_arr = np.histogram(
            a=df["epoch_found"],
            bins=self.num_epochs_run,
            range=(1, self.num_epochs_run),
        )
        cumsum_arr = np.cumsum(counts_arr) / len(
            self.all_attack_dispatch[orig_label]
        )

        return cumsum_arr

    def epochfound_cdf_first_best_df(self, orig_label: int) -> pd.DataFrame:
        cumsum_first = self.epochfound_cdf(
            orig_label=orig_label, example_type=RecordedExampleType.FIRST
        )
        cumsum_best = self.epochfound_cdf(
            orig_label=orig_label, example_type=RecordedExampleType.BEST
        )
        assert len(cumsum_first) == len(cumsum_best)
        data_arr = np.stack(
            (np.arange(len(cumsum_first)), cumsum_first, cumsum_best),
            axis=1,
        )
        return pd.DataFrame(
            data=data_arr,
            columns=[
                "epoch_found",
                "first_example_cumsum",
                "best_example_cumsum",
            ],
        ).astype(
            dtype={
                "epoch_found": "int",
                "first_example_cumsum": "float32",
                "best_example_cumsum": "float32",
            },
        )

    @property
    def data_for_epochfound_cdf_plotter(self) -> ZerosOnesDFPair:
        return ZerosOnesDFPair(
            zero_to_one=self.epochfound_cdf_first_best_df(orig_label=0),
            one_to_zero=self.epochfound_cdf_first_best_df(orig_label=1),
        )


class FullAttackResults:
    """
    Provides full summary of results of attack on a model & dataset
    """

    def __init__(
        self,
        attack_trainer_result: ads.AttackTrainerResult,
        # success_summary: ads.TrainerSuccessSummary,
    ):
        """
        :param success_summary: a TrainerSuccessSummary produced by attack
        """
        success_summary = ads.TrainerSuccessSummary(
            attack_trainer_result=attack_trainer_result
        )
        self._dataset = success_summary.dataset
        self.all_attacks_df = success_summary.all_attacks_df
        self.successful_attack_df = success_summary.successful_attacks_df
        self._first_examples_raw = success_summary.first_examples
        self._best_examples_raw = success_summary.best_examples
        self.first_perts_summary = success_summary.perts_summary_first
        self.best_perts_summary = success_summary.perts_summary_best

    @classmethod
    def from_attack_trainer_result_path(cls, attack_trainer_result_path):
        # attack_trainer_result = rio.ResourceImporter().import_pickle_to_object(
        #     path=attack_trainer_result_path
        # )
        attack_trainer_result_dto = (
            ads.ATTACK_TRAINER_RESULT_IO.import_to_struct(
                path=attack_trainer_result_path
            )
        )
        attack_trainer_result = ads.AttackTrainerResult.from_dto(
            dto=attack_trainer_result_dto
        )
        return cls(attack_trainer_result=attack_trainer_result)

    @classmethod
    def from_most_recent_attack(cls):

        attack_output_root = Path(
            CONFIG_READER.read_path("attack.attack_driver.output_dir")
        )
        attack_id = ps.get_latest_sequential_child_dirname(
            root_dir=attack_output_root
        )
        latest_attack_result_path = (
            cfg_paths / attack_id / f"final_attack_result_{attack_id}.json"
        )

        return cls.from_attack_trainer_result_path(
            attack_trainer_result_path=latest_attack_result_path
        )

    @cached_property
    def padded_features_attacked(self) -> np.array:
        """
        padded array of input features of attacked samples
        :return: array of floats
        """
        torch_padded_features_attacked = pad_sequence(
            self._dataset[:][1], batch_first=True
        )[self.all_attacks_df.dataset_index]
        return np.array(torch_padded_features_attacked)

    @cached_property
    def padded_features_success(self) -> np.array:
        """
        padded array of input features of successfully attacked samples
        :return: array of floats
        """
        return self.padded_features_attacked[
            self.successful_attack_df["attacked_samples_index"], :, :
        ]

    @cached_property
    def _orig_labels(self) -> np.array:
        """
        Originally predicted labels of all attacked samples
        :return: array of ints
        """
        return np.array(self._dataset[:][2])[self.all_attacks_df.dataset_index]

    @cached_property
    def first_examples_padded_perts(self) -> np.array:
        """
        Padded perturbations for example_data that are first example found for
        corresponding sample
        :return: array of floats
        """
        return np.array(self._first_examples_raw.perturbations)[
            self.successful_attack_df["attacked_samples_index"]
        ]

    @cached_property
    def best_examples_padded_perts(self) -> np.array:
        """
        Padded perturbations for example_data that are best example found for
        corresponding sample
        :return: array of floats
        """
        return np.array(self._best_examples_raw.perturbations)[
            self.successful_attack_df["attacked_samples_index"]
        ]

    def build_examples_summary_df(
        self,
        example_type: RecordedExampleType,
    ):
        """
        Builds a summary dataframe first example_data or best example_data
        :param example_type: type of example to summarize
        :return: a Pandas dataframe
        """
        examples_dispatch = {
            RecordedExampleType.FIRST: self._first_examples_raw,
            RecordedExampleType.BEST: self._best_examples_raw,
        }

        perts_summary_dispatch = {
            RecordedExampleType.FIRST: self.first_perts_summary,
            RecordedExampleType.BEST: self.best_perts_summary,
        }

        trainer_examples = examples_dispatch[example_type]
        perts_summary_df = perts_summary_dispatch[example_type].df

        example_summary_data_array = np.stack(
            (
                self.all_attacks_df.dataset_index[
                    self.successful_attack_df["attacked_samples_index"]
                ],
                self.successful_attack_df["attacked_samples_index"],
                self._orig_labels[
                    self.successful_attack_df["attacked_samples_index"]
                ],
                np.array(trainer_examples.epochs)[
                    self.successful_attack_df["attacked_samples_index"]
                ],
                np.array(trainer_examples.losses)[
                    self.successful_attack_df["attacked_samples_index"]
                ],
            ),
            axis=1,
        )

        example_summary_df = pd.DataFrame(
            data=example_summary_data_array,
            columns=[
                "dataset_index",
                "attacked_samples_index",
                "orig_label",
                "epoch_found",
                "loss",
            ],
        ).astype(
            dtype={
                "dataset_index": "int",
                "attacked_samples_index": "int",
                "orig_label": "int",
                "epoch_found": "int",
                "loss": "float32",
            }
        )

        return pd.concat([example_summary_df, perts_summary_df], axis=1)

    @cached_property
    def first_examples_df(self) -> pd.DataFrame:
        """
        Summary dataframe for all first example_data
        :return: Pandas Dataframe
        """
        return self.build_examples_summary_df(
            example_type=RecordedExampleType.FIRST
        )

    @cached_property
    def best_examples_df(self) -> pd.DataFrame:
        """
        Summary dataframe for all best (lowest loss for sample) example_data
        :return: Pandas dataframe
        """
        return self.build_examples_summary_df(
            example_type=RecordedExampleType.BEST
        )

    def get_summary_df_for_condition(
        self,
        seq_length: int,
        example_type: RecordedExampleType,
        orig_label: int = None,
        min_num_perts: int = None,
        max_num_perts: int = None,
    ) -> pd.DataFrame:
        example_type_dispatch = {
            RecordedExampleType.FIRST: self.first_examples_df,
            RecordedExampleType.BEST: self.best_examples_df,
        }
        starting_df = example_type_dispatch[example_type]
        filtered_df = starting_df[starting_df["seq_length"] == seq_length]
        if orig_label is not None:
            filtered_df = filtered_df[filtered_df["orig_label"] == orig_label]
        if min_num_perts is not None:
            filtered_df = filtered_df[
                filtered_df["num_perts"] >= min_num_perts
            ]
        if max_num_perts is not None:
            filtered_df = filtered_df[
                filtered_df["num_perts"] <= max_num_perts
            ]

        return filtered_df

    def get_condition_analysis(
        self,
        seq_length: int,
        example_type: RecordedExampleType,
        orig_label: int = None,
        min_num_perts: int = None,
        max_num_perts: int = None,
    ) -> AttackConditionSummary:
        """
        Builds an AttackConditionSummary object for example_data with specified
        sequence length and example type. Data can be further filtered using
        other optional params
        :param seq_length: input seq length of example_data to summarize
        :param example_type: type of example_data to summarize
        :param orig_label: (optional) original label of example_data to summarize
        :param min_num_perts: (optional) min number of nonzero perturbation
        elements required of example_data to summarize
        :param max_num_perts: (optional) max number of nonzero perturbation
        elements required of example_data to summarize
        :return:
        """

        pert_summary_dispatch = {
            RecordedExampleType.FIRST: self.first_perts_summary,
            RecordedExampleType.BEST: self.best_perts_summary,
        }

        summary_df = self.get_summary_df_for_condition(
            seq_length=seq_length,
            example_type=example_type,
            orig_label=orig_label,
            min_num_perts=min_num_perts,
            max_num_perts=max_num_perts,
        )

        perts = pert_summary_dispatch[example_type].padded_perts[
            summary_df.index, :, :
        ]

        return AttackConditionSummary(examples_df=summary_df, perts=perts)

    def get_standard_attack_condition_summaries(
        self,
        seq_length: int,
        min_num_perts: int = None,
        max_num_perts: int = None,
    ) -> StandardAttackConditionSummaries:
        return StandardAttackConditionSummaries(
            zero_to_one_first=self.get_condition_analysis(
                seq_length=seq_length,
                example_type=RecordedExampleType.FIRST,
                orig_label=0,
                min_num_perts=min_num_perts,
                max_num_perts=max_num_perts,
            ),
            zero_to_one_best=self.get_condition_analysis(
                seq_length=seq_length,
                example_type=RecordedExampleType.BEST,
                orig_label=0,
                min_num_perts=min_num_perts,
                max_num_perts=max_num_perts,
            ),
            one_to_zero_first=self.get_condition_analysis(
                seq_length=seq_length,
                example_type=RecordedExampleType.FIRST,
                orig_label=1,
                min_num_perts=min_num_perts,
                max_num_perts=max_num_perts,
            ),
            one_to_zero_best=self.get_condition_analysis(
                seq_length=seq_length,
                example_type=RecordedExampleType.BEST,
                orig_label=1,
                min_num_perts=min_num_perts,
                max_num_perts=max_num_perts,
            ),
            all_attacked_zeros_df=self.all_attacks_df[
                (self.all_attacks_df["seq_length"] == seq_length)
                & (self.all_attacks_df["orig_label"] == 0)
            ],
            all_attacked_ones_df=self.all_attacks_df[
                (self.all_attacks_df["seq_length"] == seq_length)
                & (self.all_attacks_df["orig_label"] == 1)
            ],
        )
