from functools import cached_property
from typing import Any, Type

import msgspec


class JsonReadyDataWriter:
    @cached_property
    def encoder(self) -> msgspec.json.Encoder:
        return msgspec.json.Encoder()

    def encode(self, obj: Any) -> bytes:
        return self.encoder.encode(obj)





class TupleStringDataReader:

    @staticmethod
    def dec_hook(type: Type, obj: Any) -> Any:
        if type is tuple[str]:
            return tuple(obj)

    @cached_property
    def decoder(self) -> msgspec.json.Decoder:
        return msgspec.json.Decoder(tuple[str, ...], dec_hook=self.dec_hook)

    def decode(self, obj: Any) -> tuple[str]:
        return self.decoder.decode(obj)


class DirectDataReader:
    @cached_property
    def decoder(self) -> msgspec.json.Decoder:
        return msgspec.json.Decoder()

    def decode(self, data: bytes) -> Any:
        return self.decoder.decode(data)


data_writer = JsonReadyDataWriter()
encoded_str_tuple = data_writer.encode(("hello", "there"))

tuple_str_data_reader = TupleStringDataReader()
decoded_str_tuple = tuple_str_data_reader.decode(encoded_str_tuple)


# tuple_str_data_reader = TupleStringDataReader()
# tuple_str_decoded_str_tuple = tuple_str_data_reader.decode(encoded_str_tuple)
