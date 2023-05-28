from enum import Enum, auto
from data_structures import EvalEpochResult, OptimizeDirection
from typing import Iterable, TypeVar

_T = TypeVar("_T")


class PerformanceSelector:

    _selection_dispatch = {
        OptimizeDirection.MIN: min,
        OptimizeDirection.MAX: max
    }

    def __init__(self, optimize_direction: OptimizeDirection):
        self._optimize_direction = optimize_direction

    def choose_best_val(self, values: Iterable[_T]) -> _T:
        return self._selection_dispatch[self._optimize_direction](values)


class EarlyStopper:
    def __init__(
        self,
        performance_metric: str,
        optimize_direction: OptimizeDirection,
        patience: int,
    ):
        self._optimize_direction = optimize_direction
        self._performance_metric = performance_metric
        self._patience = patience
        self._num_deteriorations = 0
        if optimize_direction == OptimizeDirection.MIN:
            self._previous_result = float("inf")
        if optimize_direction == OptimizeDirection.MAX:
            self._previous_result = float("-inf")

    def _is_greater_than_previous_result(self, result: EvalEpochResult) -> bool:
        return getattr(result, self._performance_metric) > self._previous_result

    def _is_less_than_previous_result(self, result: EvalEpochResult) -> bool:
        return getattr(result, self._performance_metric) < self._previous_result

    def _reset_num_deteriorations(self):
        self._num_deteriorations = 0

    def _update_previous_result(self, result: EvalEpochResult):
        self._previous_result = getattr(result, self._performance_metric)

    def _is_worse_than_best_result(self, result: EvalEpochResult) -> bool:

        dispatch_table = {
            OptimizeDirection.MIN: self._is_greater_than_previous_result,
            OptimizeDirection.MAX: self._is_less_than_previous_result
        }

        return dispatch_table[self._optimize_direction](result)

    def _check_and_update(self, result: EvalEpochResult):
        if self._is_worse_than_best_result(result=result):
            self._num_deteriorations += 1
        else:
            self._reset_num_deteriorations()
        self._update_previous_result(result=result)

    def indicates_early_stop(self, result: EvalEpochResult):
        self._check_and_update(result=result)
        return self._num_deteriorations > self._patience




