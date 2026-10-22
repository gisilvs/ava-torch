import json
from pathlib import Path

import hydra
import soundfile as sf
import torch
from omegaconf import OmegaConf
from stable_audio_3 import AutoencoderModel
from stable_audio_3.loading_utils import load_autoencoder
from stable_audio_3.model_configs import models


@torch.inference_mode()
def encode(cfg):
    config_path, weights = models[cfg.model].resolve()
    config = json.loads(Path(config_path).read_text())
    # Autoencoder settings from the pretrained model, not data augmentation.
    transform = config['model']['pretransform']
    # Match the latent units used by the previous encoder and mapper.
    scale = transform.get('scale', 1.0)
    # Encode overlapping chunks to reduce memory, when requested by the model config.
    chunked = transform.get('chunked', False)
    autoencoder = load_autoencoder(config_path, weights, cfg.device)
    autoencoder.eval().requires_grad_(False)
    encoder = AutoencoderModel(autoencoder, config['sample_rate'], cfg.device)

    folder = Path(cfg.data_dir)
    OmegaConf.save(cfg, folder / 'encoder.yaml', resolve=True)
    for split in ('train', 'val', 'test'):
        source = folder / split
        output = source / 'latents'
        output.mkdir(exist_ok=True)
        labels = json.loads((source / 'labels.json').read_text())
        for index, label in enumerate(labels):
            path = source / label['audio']
            audio, sample_rate = sf.read(path, dtype='float32', always_2d=True)
            # SAME expects [channels, samples] and handles resampling/stereo/padding.
            waveform = torch.from_numpy(audio.T.copy())
            latent = encoder.encode(waveform, sample_rate, chunked=chunked) / scale
            # Drop the batch dimension, matching the previous training features.
            feature = latent[0].float().cpu()
            torch.save(feature, source / label['latent'])
            if index == 0:
                print(f'Latent shape: {tuple(feature.shape)}', flush=True)
            if (index + 1) % 100 == 0:
                print(f'{split}: {index + 1}/{len(labels)}', flush=True)
        print(f'Saved {len(labels)} {split} latents to {output}')


@hydra.main(version_base=None, config_path='../configs', config_name='encoding')
def main(cfg):
    encode(cfg)


if __name__ == '__main__':
    main()
