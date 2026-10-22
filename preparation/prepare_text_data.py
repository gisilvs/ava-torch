import json
import shutil
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf


@hydra.main(version_base=None, config_path='../configs', config_name='text_data')
def main(cfg):
    """Convert an existing text-generated dataset to the common latent/audio layout."""
    source = Path(cfg.source_dir)
    manifest = json.loads((source / 'manifest.json').read_text())
    folder = Path(cfg.data_dir)
    folder.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, folder / 'config.yaml', resolve=True)
    for split in ('train', 'val', 'test'):
        output = folder / split
        (output / 'latents').mkdir(parents=True, exist_ok=True)
        rows = [row for row in manifest['records']
                if row['split'] == split and row['accepted'] and row['pitch']['accepted']]
        labels = []
        for index, row in enumerate(rows):
            filename = f'{index:06d}'
            latent_path = f'latents/{filename}.pt'
            latent = torch.from_numpy(np.load(source / row['latents'])).float()
            torch.save(latent, output / latent_path)
            shutil.copyfile(source / row['target'], output / f'{filename}.wav')
            labels.append({'latent': latent_path, 'audio': f'{filename}.wav',
                           'frequency': row['pitch']['median_hz'],
                           'hold': cfg.hold, 'velocity': 1.0, 'prompt': row['prompt']})
        (output / 'labels.json').write_text(json.dumps(labels, indent=2) + '\n')
        print(f'Prepared {len(labels)} {split} examples in {output}')


if __name__ == '__main__':
    main()
