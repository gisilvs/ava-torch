import json
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from synth.controls import encode_controls


class SynthDataset(Dataset):
    """Return a latent, normalized continuous controls and two oscillator labels."""

    def __init__(self, data_dir, split, parameters=None):
        data_dir = Path(data_dir)
        config = OmegaConf.load(data_dir / 'config.yaml')
        self.folder = data_dir / split
        self.labels = json.loads((self.folder / 'labels.json').read_text())
        self.parameters = list(config.dataset.ranges) if parameters is None else list(parameters)

        # Compute training targets once, rather than repeating this every epoch.
        controls = {
            name: torch.tensor([[row['preset'][name]] for row in self.labels])
            for name in config.dataset.ranges
        }
        # MIDI 69 is A4 at 440 Hz; each 12 semitones doubles the frequency.
        # Pitch is needed to normalize cutoff/depth labels, not as a network input.
        freq = torch.tensor([[440 * 2 ** ((row['midi_pitch'] - 69) / 12)]
                             for row in self.labels])
        self.targets = encode_controls(controls, self.parameters, freq, config.synth.sample_rate)
        self.selectors = torch.tensor([
            [row['preset']['oscillator'], row['preset']['oscillator_b']]
            for row in self.labels
        ], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        latent = torch.load(self.folder / self.labels[index]['latent'],
                            map_location='cpu', weights_only=True)
        return latent, self.targets[index], self.selectors[index]
