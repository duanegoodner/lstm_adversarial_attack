from abc import ABC, abstractmethod
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from typing import Callable


class DataLoaderBuilder(ABC):
    @abstractmethod
    def build(self, dataset: Dataset, batch_size: int) -> DataLoader:
        pass


class WeightedRandomSamplerBuilder:
    def __init__(self, labels: torch.tensor):
        self.labels = labels

    def build(self) -> WeightedRandomSampler:
        class_sample_counts = np.unique(self.labels, return_counts=True)[1]
        class_weights = 1.0 / class_sample_counts
        sample_weights = np.choose(self.labels, class_weights)
        return WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights)
        )


class WeightedDataLoaderBuilder:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        labels_idx: int = 1,
        collate_fn: Callable = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.labels_idx = labels_idx
        self.collate_fn = collate_fn

    # def __call__(
    #     self,
    #     dataset: Dataset,
    #     batch_size: int,
    #     labels_idx: int = 1,
    # ):
    #     return self.build(dataset=dataset, batch_size=batch_size)

    # @staticmethod
    # def _build_weighted_random_sampler(
    #     labels: torch.tensor,
    # ) -> WeightedRandomSampler:
    #     class_sample_counts = np.unique(labels, return_counts=True)[1]
    #     class_weights = 1.0 / class_sample_counts
    #     sample_weights = np.choose(labels, class_weights)
    #     return WeightedRandomSampler(
    #         weights=sample_weights, num_samples=len(sample_weights)
    #     )

    def build(
        self,
        # dataset: Dataset,
        # batch_size: int,
        # labels_idx: int = 1,
        # collate_fn: Callable = None,
    ) -> DataLoader:
        labels = torch.tensor(
            [
                self.dataset[i][self.labels_idx].item()
                for i in range(len(self.dataset))
            ],
            dtype=torch.long,
        )
        assert labels.dim() == 1
        # assert not torch.is_floating_point(labels)
        # assert not torch.is_complex(labels)
        # weighted_random_sampler = self._build_weighted_random_sampler(
        #     labels=labels
        # )
        weighted_random_sampler = WeightedRandomSamplerBuilder(
            labels=labels
        ).build()

        if self.collate_fn is not None:
            return DataLoader(
                dataset=self.dataset,
                batch_size=self.batch_size,
                sampler=weighted_random_sampler,
                collate_fn=self.collate_fn,
            )
        else:
            # not if OK to pass collate_fn = None; use if/else to be safe
            return DataLoader(
                dataset=self.dataset,
                batch_size=self.batch_size,
                sampler=weighted_random_sampler,
            )
