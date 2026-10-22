import json
from pathlib import Path

import soundfile as sf
import torch
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    """Load latent/audio pairs and their note settings, regardless of origin."""

    def __init__(self, data_dir, split):
        self.folder = Path(data_dir) / split
        self.labels = json.loads((self.folder / 'labels.json').read_text())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        row = self.labels[index]
        latent = torch.load(self.folder / row['latent'], map_location='cpu', weights_only=True)
        audio, _ = sf.read(self.folder / row['audio'], dtype='float32', always_2d=True)
        audio = torch.from_numpy(audio.mean(axis=1)).unsqueeze(0)
        note = torch.tensor([row['frequency'], row['hold'], row['velocity']], dtype=torch.float32)
        return latent, audio, note
