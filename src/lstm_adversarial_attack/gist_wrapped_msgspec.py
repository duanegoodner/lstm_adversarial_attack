from functools import cached_property
from pathlib import Path
from typing import Callable, TypeVar, Any, Type

import msgspec
import torch

EncodeType = TypeVar("EncodeType", bound=msgspec.Struct)
DecodeType = TypeVar("DecodeType", bound=msgspec.Struct)
DTOType = TypeVar("DTOType", bound=msgspec.Struct)



class WrappedMsgspec:
    def __init__(
        self,
        dto_constructor: Callable,
        enc_hook: Callable = None,
        dec_hook: Callable = None,
        **kwargs
    ):
        self.dto_type = dto_constructor
        self.enc_hook = enc_hook
        self.dec_hook = dec_hook
        self.data = dto_constructor(**kwargs) if kwargs else None


    @cached_property
    def encoder(self) -> msgspec.json.Encoder:
        return msgspec.json.Encoder(enc_hook=self.enc_hook)

    def encode(self, obj: EncodeType) -> bytes:
        return self.encoder.encode(obj)

    def export(self, path: Path):
        encoded_data = self.encode(self.data)
        with path.open("wb") as out_file:
            out_file.write(encoded_data)

    @cached_property
    def decoder(self) -> msgspec.json.Decoder:
        return msgspec.json.Decoder(self.dto_type, dec_hook=self.dec_hook)

    def decode(self, obj: bytes) -> DecodeType:
        return self.decoder.decode(obj)

    def import_to_struct(self, path: Path):
        with path.open("rb") as in_file:
            encoded_data = in_file.read()
        self.data = self.decode(obj=encoded_data)


def import_to_msgspec(path: Path, dto_constructor: Callable) -> WrappedMsgspec:
    new_wrapped_msgspec = WrappedMsgspec(dto_constructor, enc_hook=my_dto_enc_hook, dec_hook=my_dto_dec_hook)
    new_wrapped_msgspec.import_to_struct(path=path)
    return new_wrapped_msgspec


class MyDTO(msgspec.Struct):
    data: torch.Tensor

def my_dto_dec_hook(decode_type: Type, obj: Any) -> Any:
    if decode_type is torch.Tensor:
        return torch.tensor(obj)
    else:
        raise NotImplementedError(f"Objects of type {decode_type} not supported.")

def my_dto_enc_hook(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    else:
        raise NotImplementedError(f"Objects of type {type(obj)} not supported.")


if __name__ == "__main__":
    my_dto = MyDTO(data=torch.rand(3, 3, 3))

    my_wrapped_dto = WrappedMsgspec(
        dto_constructor=MyDTO,
        enc_hook=my_dto_enc_hook,
        dec_hook=my_dto_dec_hook,
        data=torch.rand(3, 3, 3),
    )

    my_wrapped_dto.export(path=Path("test_output.json"))

    imported_wrapped_dto = import_to_msgspec(path=Path("test_output.json"), dto_constructor=MyDTO)

    print("pause")



