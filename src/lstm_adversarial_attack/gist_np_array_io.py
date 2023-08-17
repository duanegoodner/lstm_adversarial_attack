from typing import Any, Type

import msgspec
import numpy as np


class FeatureArrays(msgspec.Struct):
    data: list[np.ndarray]


class ClassLabels(msgspec.Struct):
    data: list[int]


class MeasurementColumnNames(msgspec.Struct):
    data: tuple[str, ...]



def enc_hook(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()


def dec_hook(type: Type, obj: Any) -> Any:
    if type is np.ndarray:
        return np.array(obj)
    else:
        raise NotImplementedError(f"Objects of type {type} are not supported")


my_arrays = [
    np.array([[1.1, 2.2, 3.3]]),
    np.array([[4.4, 5.5, 6.6], [7.7, 8.8, 9.9]]),
]
my_encoder = msgspec.json.Encoder(enc_hook=enc_hook)
my_encoded_arrays = my_encoder.encode(my_arrays)
my_decoder = msgspec.json.Decoder(list[np.ndarray], dec_hook=dec_hook)
simple_decoder = msgspec.json.Decoder()
simple_decoded_arrays = simple_decoder.decode(my_encoded_arrays)

my_decoded_arrays = my_decoder.decode(my_encoded_arrays)

