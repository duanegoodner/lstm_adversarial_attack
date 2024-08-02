from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as ud
from sklearn.metrics import confusion_matrix

import lstm_adversarial_attack.dataset.dataset_with_index as dsi


@dataclass
class StandardInferenceResults:
    """
    Data container to hold results of running all samples in dataset through
    forward pass of a classification model.
    :param dataset: dataset with samples that ran through model
    :param y_pred: predicted classes
    :param y_score: values output by final activation layer of model
    :param y_true: actual classes
    """
    dataset: dsi.DatasetWithIndex
    y_pred: torch.tensor
    y_score: torch.tensor
    y_true: torch.tensor

    @property
    def correct_prediction_indices(self) -> np.ndarray:
        """
        Indices of correctly classified samples
        :return: array of ints
        """
        return np.where(self.y_pred == self.y_true)[0]

    @property
    def incorrect_prediction_indices(self) -> np.ndarray:
        """
        Indices of misclassified samples
        :return: array of ints
        """
        return np.where(self.y_pred != self.y_true)[0]

    @property
    def confusion_matrix(self) -> np.ndarray:
        """
        Confusion matrix based on y_pred and y_true
        :return: sklearn.metrics.confusion_matrix
        """
        return confusion_matrix(y_true=self.y_true, y_pred=self.y_pred)

    def indices_true_of_predicted_class(self, class_num: int) -> np.ndarray:
        """
        Gets indices of samples that are correctly predicted and belong to
        particular class.
        :param class_num: a sample class (0 or 1 for binary classification)
        :return: array of ints
        """
        return np.where((self.y_pred == class_num == self.y_true))[0]

    def indices_false_of_predicted_class(self, class_num: int) -> np.ndarray:
        """
        Indices of incorrectly classified samples of particular class.
        :param class_num: a sample class
        :return: array of ints
        """
        return np.where((self.y_pred == class_num != self.y_true))[0]


class StandardModelInferrer:
    """
    Runs all samples in dataset through forward pass of a model and obtains
    inference results
    """
    def __init__(
        self,
        device: torch.device,
        model: nn.Module,
        dataset: dsi.DatasetWithIndex,
        collate_fn: Callable,
        batch_size: int = 128,
    ):
        """
        :param device: device to run on
        :param model: model used for inference
        :param dataset: dataset used for inference
        :param collate_fn: converts dataset elements to batches
        :param batch_size: num samples per batch
        """
        self.device = device
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
        """
        Converts model outputs to predicted class
        :param model_output: final output tensor from forward pass
        :return: integer corresponding to predicted class
        """
        return torch.argmax(input=model_output, dim=1)

    @torch.no_grad()
    def infer(self):
        """
        Runs all samples in dataloader through forward pass, obtains class
        prediction, and calcs StandardInferenceResults
        :return: StandardInferenceResults
        """
        self.model.to(self.device)
        self.model.eval()
        all_indices = torch.LongTensor()
        all_y_true = torch.LongTensor()
        all_y_pred = torch.LongTensor()
        all_y_score = torch.FloatTensor()
        for indices, inputs, y in self.data_loader:
            inputs.features, y = inputs.features.to(self.device), y.to(
                self.device
            )
            y_hat = self.model(inputs)
            y_pred = self.interpret_output(model_output=y_hat)
            all_indices = torch.cat((all_indices, indices), dim=0)
            all_y_true = torch.cat((all_y_true, y.to("cpu")), dim=0)
            all_y_pred = torch.cat((all_y_pred, y_pred.to("cpu")), dim=0)
            all_y_score = torch.cat((all_y_score, y_hat.to("cpu")), dim=0)

        # make sure samples have not been shuffled
        assert torch.all(all_indices == torch.arange(len(all_indices))).item()
        return StandardInferenceResults(
            dataset=self.dataset,
            y_pred=all_y_pred,
            y_score=all_y_score,
            y_true=all_y_true,
        )
