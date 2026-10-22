import torch


class SawtoothOscillator:
    """Approximate a rising sawtooth with harmonics 1 through 31."""

    def __init__(self, sample_rate=24000):
        self.sample_rate = sample_rate

    def __call__(self, freq, time):
        harmonics = torch.arange(1, 32, device=freq.device, dtype=freq.dtype)
        harmonics = harmonics.view(1, -1, 1)
        freq = freq.unsqueeze(1)
        time = time.unsqueeze(-2)
        allowed = freq * harmonics < self.sample_rate / 2
        amplitudes = (-1) ** (harmonics + 1) / harmonics
        # Initial phase is zero when time starts at zero.
        waves = torch.sin(2 * torch.pi * freq * harmonics * time)
        return 2 / torch.pi * (allowed * amplitudes * waves).sum(dim=1)
