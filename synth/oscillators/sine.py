import torch


class SineOscillator:
    """Hz [batch, 1], seconds [samples] -> waveform [batch, samples]."""

    def __call__(self, freq, time):
        # Initial phase is zero when time starts at zero; no phase-offset control yet.
        return torch.sin(2 * torch.pi * freq * time)
