import numpy as np
import sys
import torch
import torch.nn as nn
import torch.utils.data as ud
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


from dataset_with_index import DatasetWithIndex


@dataclass
class StandardInferenceResults:
    y_pred: torch.tensor
    y_score: torch.tensor
    y_true: torch.tensor

    @property
    def correct_prediction_indices(self) -> np.ndarray:
        return np.where(self.y_pred == self.y_true)


class StandardModelInferrer:
    def __init__(
        self,
        model: nn.Module,
        dataset: DatasetWithIndex,
        collate_fn: Callable,
        batch_size: int = 128,
    ):
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.data_loader = ud.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=False,
        )

    # extract this from evaluate function so it can be easily overridden
    @staticmethod
    def interpret_output(model_output: torch.tensor) -> torch.tensor:
        return torch.argmax(input=model_output, dim=1)

    @torch.no_grad()
    def infer(self):
        self.model.eval()
        all_y_true = torch.LongTensor()
        all_y_pred = torch.LongTensor()
        all_y_score = torch.FloatTensor()
        for index, x, y in self.data_loader:
            x, y = x.to(self.model.model_device), y.to(self.model.model_device)
            y_hat = self.model(x)
            y_pred = self.interpret_output(model_output=y_hat)
            all_y_true = torch.cat((all_y_true, y.to("cpu")), dim=0)
            all_y_pred = torch.cat((all_y_pred, y_pred.to("cpu")), dim=0)
            all_y_score = torch.cat((all_y_score, y_hat.to("cpu")), dim=0)
        return StandardInferenceResults(
            y_pred=all_y_pred, y_score=all_y_score, y_true=all_y_true
        )

    def get_correctly_predicted_samples(self) -> ud.Dataset:
        inference_results = self.infer()
        return ud.Subset(
            dataset=self.dataset,
            indices=inference_results.correct_prediction_indices[0],
        )
