from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path
from typing import Callable, TypeVar, Type, Any

import msgspec

MsgSpecStructType = TypeVar("MsgSpecStructType", bound=msgspec.Struct)


class MsgspecIO:
    def __init__(self, msgspec_struct_type: Callable[..., MsgSpecStructType]):
        self.struct_type = msgspec_struct_type

    @staticmethod
    def dec_hook(decode_type: Type, obj: Any) -> Any:
        pass

    @property
    def decoder(self) -> msgspec.json.Decoder:
        return msgspec.json.Decoder(self.struct_type, dec_hook=self.dec_hook)

    def decode(self, obj: bytes) -> MsgSpecStructType:
        return self.decoder.decode(obj)

    def import_to_struct(self, path: Path) -> MsgSpecStructType:
        with path.open(mode="rb") as in_file:
            encoded_data = in_file.read()
        return self.decode(obj=encoded_data)

    @staticmethod
    def enc_hook(obj: Any) -> Any:
        pass

    @property
    def encoder(self) -> msgspec.json.Encoder:
        return msgspec.json.Encoder(enc_hook=self.enc_hook)

    def encode(self, obj: MsgSpecStructType) -> bytes:
        return self.encoder.encode(obj)

    def export(self, obj: MsgSpecStructType, path: Path):
        encoded_data = self.encode(obj)
        with path.open(mode="wb") as out_file:
            out_file.write(encoded_data)



class StandardStructWriter(ABC):
    def __init__(self, struct_type: Callable[..., MsgSpecStructType]):
        self._struct_type = struct_type

    @staticmethod
    @abstractmethod
    def enc_hook(obj: Any) -> Any:
        pass

    @cached_property
    def encoder(self) -> msgspec.json.Encoder:
        return msgspec.json.Encoder(enc_hook=self.enc_hook)

    def encode(self, obj: MsgSpecStructType) -> bytes:
        return self.encoder.encode(obj)

    def export(self, obj: MsgSpecStructType, path: Path):
        encoded_data = self.encode(obj)
        with path.open(mode="wb") as out_file:
            out_file.write(encoded_data)


class StandardStructReader(ABC):
    def __init__(self, struct_type: Callable[..., MsgSpecStructType]):
        self._struct_type = struct_type

    @staticmethod
    @abstractmethod
    def dec_hook(decode_type: Type, obj: Any) -> Any:
        pass

    @cached_property
    def decoder(self) -> msgspec.json.Decoder:
        return msgspec.json.Decoder(self._struct_type, dec_hook=self.dec_hook)

    def decode(self, obj: bytes) -> MsgSpecStructType:
        return self.decoder.decode(obj)

    def import_struct(self, path: Path) -> MsgSpecStructType:
        with path.open(mode="rb") as in_file:
            encoded_struct = in_file.read()
        return self.decode(obj=encoded_struct)
