import json
import math
from pathlib import Path

import hydra
import soundfile as sf
import torch
from omegaconf import OmegaConf

from synth.controls import linear, logarithmic
from synth.synth import Synth


def sample_preset(ranges, freq, sample_rate, generator):
    """Sample physical parameters, leaving room for the full LFO sweep."""
    preset = {}
    for name, (low, high, scale) in ranges.items():
        if name == 'cutoff_octaves':
            depth = ranges['lfo_depth'][1]
            low = max(low, math.log2(20 / freq) + depth)
            high = min(high, math.log2(min(10000, 0.45 * sample_rate) / freq) - depth)
        value = torch.rand((), generator=generator)
        mapping = logarithmic if scale == 'log' else linear
        preset[name] = mapping(value, low, high).item()
    return preset


@torch.inference_mode()
def generate(cfg):
    torch.set_num_threads(cfg.threads)
    folder = Path(cfg.data_dir)
    folder.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, folder / 'config.yaml', resolve=True)
    synth = Synth(**cfg.synth)
    dataset = cfg.dataset

    for split_index, split in enumerate(('train', 'val', 'test')):
        output = folder / split
        output.mkdir(exist_ok=True)
        generator = torch.Generator().manual_seed(cfg.seed + split_index)
        count = dataset.counts[split]
        # Shuffle a balanced list of the 16 ordered oscillator pairs.
        pairs = torch.arange(count) % 16
        pairs = pairs[torch.randperm(count, generator=generator)]
        labels = []
        for index, pair in enumerate(pairs.tolist()):
            midi = torch.randint(dataset.midi_min, dataset.midi_max + 1,
                                 (), generator=generator).item()
            freq = torch.tensor([[440 * 2 ** ((midi - 69) / 12)]], device=cfg.device)
            preset = sample_preset(dataset.ranges, freq.item(), synth.sample_rate, generator)
            preset['oscillator'] = pair // 4
            preset['oscillator_b'] = pair % 4
            audio = synth(freq, dataset.duration, dataset.note_press_duration,
                          velocity=dataset.velocity, **preset)
            filename = f'{index:06d}.wav'
            sf.write(output / filename, audio.cpu().reshape(-1).numpy(),
                     synth.sample_rate, subtype='FLOAT')
            labels.append({'latent': f'latents/{index:06d}.pt', 'audio': filename,
                           'frequency': freq.item(), 'hold': dataset.note_press_duration,
                           'velocity': dataset.velocity, 'midi_pitch': midi, 'preset': preset})
            if (index + 1) % 100 == 0:
                print(f'{split}: {index + 1}/{count}', flush=True)
        (output / 'labels.json').write_text(json.dumps(labels, indent=2) + '\n')
        print(f'Saved {count} {split} sounds to {output}')


@hydra.main(version_base=None, config_path='../configs', config_name='dataset')
def main(cfg):
    generate(cfg)


if __name__ == '__main__':
    main()
