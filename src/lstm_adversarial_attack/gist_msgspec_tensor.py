from typing import Type, Any

import msgspec
import torch


class MyDTO(msgspec.Struct):
    data: torch.Tensor


def tensor_dec_hook(type: Type, obj: Any) -> Any:
    if type is torch.Tensor:
        return torch.tensor(obj)
    else:
        raise NotImplementedError(f"Objects of type {type} not supported.")

def tensor_enc_hook(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    else:
        raise NotImplementedError(f"Objects of type {type(obj)} not supported.")

encoder = msgspec.json.Encoder(enc_hook=tensor_enc_hook)
decoder = msgspec.json.Decoder(MyDTO, dec_hook=tensor_dec_hook)


if __name__ == "__main__":
    my_data = torch.rand(3, 3, 3)
    my_dto = MyDTO(data=my_data)
    print(my_dto)

    buf = encoder.encode(my_dto)
    decoded_dto = decoder.decode(buf)
    print(decoded_dto)
