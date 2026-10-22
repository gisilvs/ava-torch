import torch


class SineLFO:
    """Rate is Hz, depth is octaves; both tensors have shape [batch, 1]."""

    def __call__(self, time, rate, depth):
        # Initial phase is zero when time starts at zero.
        return depth * torch.sin(2 * torch.pi * rate * time)
