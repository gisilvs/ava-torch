import json
import random
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from preparation.text_prompts import build_prompt


def replacement_prompt(cfg, seen, sample_id, seed):
    """Draw a fresh prompt using the saved dataset's descriptions and pitch ranges."""
    rng = random.Random(seed)
    for _ in range(10000):
        instrument = rng.choice(list(cfg.instrument_descriptions))
        description = rng.choice(cfg.instrument_descriptions[instrument])
        modifier = rng.choice(cfg.modifiers)
        if modifier:
            description += f', {modifier}'
        low, high = cfg.pitch_ranges[instrument]
        midi = rng.randint(low, high)
        prompt = build_prompt(instrument, description, midi, cfg.duration)
        if prompt not in seen:
            return {'id': sample_id, 'instrument': instrument,
                    'description': description, 'midi_pitch': midi,
                    'prompt': prompt, 'seed': seed}
    raise RuntimeError('Could not find an unused prompt; add more descriptions or modifiers')


def sample_prompts(cfg):
    """Sample the fixed prompt list; audio generation retries each prompt later."""
    rng = random.Random(cfg.seed)
    instruments = list(cfg.instrument_descriptions)
    splits = list(cfg.counts)
    seen = set()
    samples = {split: [] for split in splits}
    sample_id = 0

    # Fail early if the requested size exceeds the available unique prompts.
    available = 0
    for instrument in instruments:
        descriptions = set()
        for description in cfg.instrument_descriptions[instrument]:
            for modifier in cfg.modifiers:
                text = f'{description}, {modifier}' if modifier else description
                descriptions.add(text.strip())
        low, high = cfg.pitch_ranges[instrument]
        available += len(descriptions) * (high - low + 1)
    if sum(cfg.counts.values()) > available:
        raise ValueError(f'Requested too many prompts: these descriptions and pitches allow {available}')

    for split in splits:
        while len(samples[split]) < cfg.counts[split]:
            instrument = rng.choice(instruments)
            description = rng.choice(cfg.instrument_descriptions[instrument])
            modifier = rng.choice(cfg.modifiers)
            if modifier:
                description += f', {modifier}'
            low, high = cfg.pitch_ranges[instrument]
            midi = rng.randint(low, high)
            prompt = build_prompt(instrument, description, midi, cfg.duration)

            # One global set prevents duplicates within and across all splits.
            if prompt in seen:
                continue
            seen.add(prompt)

            samples[split].append({
                'id': f'{len(samples[split]):06d}', 'instrument': instrument,
                'description': description, 'midi_pitch': midi, 'prompt': prompt,
                'seed': cfg.seed + sample_id,
            })
            sample_id += 1
    return samples


@hydra.main(version_base=None, config_path='../configs', config_name='prompt_generation')
def main(cfg):
    samples = sample_prompts(cfg)
    folder = Path(cfg.data_dir)
    folder.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, folder / 'prompt_config.yaml', resolve=True)
    for split, rows in samples.items():
        output = folder / split
        output.mkdir(exist_ok=True)
        (output / 'prompts.json').write_text(json.dumps(rows, indent=2) + '\n')
        print(f'Saved {len(rows)} {split} candidate prompts to {output}')


if __name__ == '__main__':
    main()
