from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Type, TypeVar

import msgspec
import torch

MsgSpecStructType = TypeVar("MsgSpecStructType", bound=msgspec.Struct)

class MsgSpecIO:
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


class MySimpleStruct(msgspec.Struct):
    item: int

class MyStructWithTensor(msgspec.Struct):
    data: torch.Tensor


class MySimpleStructIO(MsgSpecIO):
    def __init__(self):
        super().__init__(msgspec_struct_type=MySimpleStruct)


class MyStructWithTensorIO(MsgSpecIO):
    def __init__(self):
        super().__init__(msgspec_struct_type=MyStructWithTensor)


    @staticmethod
    def enc_hook(obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        else:
            raise NotImplementedError(f"Objects of type {type(obj)} not supported.")

    @staticmethod
    def dec_hook(decode_type: Type, obj: Any) -> Any:
        if decode_type is torch.Tensor:
            return torch.tensor(obj)
        else:
            raise NotImplementedError(f"Objects of type {decode_type} not supported.")


if __name__ == "__main__":
    my_simple_struct = MySimpleStruct(item=1)
    print(my_simple_struct)
    my_simple_struct_io = MySimpleStructIO()
    my_simple_struct_io.export(obj=my_simple_struct, path=Path("my_simple_struct.json"))
    imported_simple_struct = my_simple_struct_io.import_to_struct(path=Path("my_simple_struct.json"))
    print(imported_simple_struct)

    my_tensor_struct = MyStructWithTensor(data=torch.rand(3, 3, 3))
    print(my_tensor_struct)
    my_tensor_struct_io = MyStructWithTensorIO()
    my_tensor_struct_io.export(obj=my_tensor_struct, path=Path("my_tensor_struct.json"))
    imported_tensor_struct = my_tensor_struct_io.import_to_struct(path=Path("my_tensor_struct.json"))
    print(imported_tensor_struct)