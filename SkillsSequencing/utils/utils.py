import torch


def prepare_torch():
    device = "cpu"
    torch.set_default_dtype(torch.float64)
    return device

