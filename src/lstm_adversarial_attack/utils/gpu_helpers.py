import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        cur_device = torch.device("cuda:0")
    else:
        cur_device = torch.device("cpu")
    return cur_device