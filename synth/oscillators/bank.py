import torch

from .sine import SineOscillator
from .square import SquareOscillator
from .sawtooth import SawtoothOscillator
from .triangle import TriangleOscillator


class OscillatorBank:
    """Select by integer ID, or combine using weights shaped [batch, 4]."""

    names = ('sine', 'square', 'sawtooth', 'triangle')

    def __init__(self, sample_rate=24000):
        self.oscillators = [
            SineOscillator(),
            SquareOscillator(sample_rate),
            SawtoothOscillator(sample_rate),
            TriangleOscillator(sample_rate),
        ]

    def __call__(self, freq, time, selection=0):
        if isinstance(selection, int):
            return self.oscillators[selection](freq, time)
        waves = torch.stack([osc(freq, time) for osc in self.oscillators], dim=1)
        return (selection.unsqueeze(-1) * waves).sum(dim=1)
