from typing import Any, Type

import msgspec
import torch


class DataContainer(msgspec.Struct):
    data: torch.Tensor


def replace_inf(nested_list):
    if isinstance(nested_list, list):
        return [replace_inf(item) for item in nested_list]
    elif nested_list == "inf":
        return float("inf")
    else:
        return nested_list

# Example usage:

def enc_hook(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        if obj.dim() > 0:
            return list(obj)
        else:
            if obj.item() == float("inf"):
                return "inf"
            else:
                return obj.item()
    else:
        raise NotImplementedError(
            f"Objects of type {type(obj)} not supported yet."
        )


def dec_hook(decode_type: Type, obj: Any) -> Any:
    if decode_type is torch.Tensor:
        obj = replace_inf(obj)
        return torch.tensor(obj)
    else:
        raise NotImplementedError(
            f"Objects of type {decode_type} not supported yet."
        )


container_encoder = msgspec.json.Encoder(enc_hook=enc_hook)
container_decoder = msgspec.json.Decoder(DataContainer, dec_hook=dec_hook)


if __name__ == "__main__":
    my_data = torch.tensor([[0.12, float("inf")], [0.14, 0.15]])
    print(f"my_data = {my_data}")
    my_data_container = DataContainer(data=my_data)

    my_encoded_data_container = container_encoder.encode(my_data_container)
    print(my_encoded_data_container)

    my_decoded_data_container = container_decoder.decode(
        my_encoded_data_container
    )
    print(my_decoded_data_container.data)
