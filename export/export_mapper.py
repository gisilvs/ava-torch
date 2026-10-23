"""Export a clean parameter- or audio-trained mapper for the existing Ava plugin."""
import json
from pathlib import Path

import hydra
import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from export.utils import checksum, copy_model_notices, save_json
from networks.mapper import ParameterTransformer


class Mapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, latents, valid_frames):
        # Keep Ava's two-input interface. Training uses all 66 frames; for shorter
        # inputs, zero the missing tail without changing transformer attention.
        frames = torch.arange(66, device=latents.device)
        valid = frames[None, :] < valid_frames[:, None]
        return self.model(latents * valid[:, None, :])


@torch.inference_mode()
def verify(model, mapper, session, latent):
    frames = torch.tensor([66], dtype=torch.int64)
    expected = model(latent)
    actual = session.run(None, {'latents': latent.numpy(), 'valid_frames': frames.numpy()})
    for target, result in zip(expected, actual):
        np.testing.assert_allclose(result, target.numpy(), rtol=1e-4, atol=1e-5)
    # Also check that the compatibility adapter exports its tail mask correctly.
    frames = torch.tensor([33], dtype=torch.int64)
    expected = mapper(latent, frames)
    actual = session.run(None, {'latents': latent.numpy(), 'valid_frames': frames.numpy()})
    for target, result in zip(expected, actual):
        np.testing.assert_allclose(result, target.numpy(), rtol=1e-4, atol=1e-5)


@hydra.main(version_base=None, config_path='../configs', config_name='mapper_export')
def main(cfg):
    checkpoint = torch.load(cfg.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint.get('latent_normalization') != 'none':
        raise ValueError('Use a checkpoint trained without latent normalization')
    if checkpoint['control_mapping'] != 'synth_v1' or checkpoint['feature_shape'] != [256, 66]:
        raise ValueError('Ava requires synth_v1 controls and SAME [256, 66] features')
    parameters = checkpoint['parameters']
    required = {'gain', 'mix', 'attack', 'decay', 'sustain', 'release', 'cutoff_octaves',
                'q', 'lfo_rate', 'lfo_depth', 'detune_cents', 'noise_mix', 'drive'}
    if len(parameters) != 13 or set(parameters) != required:
        raise ValueError('Ava requires all 13 continuous controls')
    model = ParameterTransformer(checkpoint['feature_shape'], controls=13,
                                 **checkpoint['config']['network']).eval()
    model.load_state_dict(checkpoint['model'])
    mapper = Mapper(model).eval()
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'mapper.onnx'
    latent = torch.zeros(1, 256, 66)
    frames = torch.tensor([66], dtype=torch.int64)

    # Export ordinary attention operations rather than PyTorch's fused CPU path.
    fastpath = torch.backends.mha.get_fastpath_enabled()
    try:
        torch.backends.mha.set_fastpath_enabled(False)
        torch.onnx.export(mapper, (latent, frames), str(path), opset_version=17,
                          input_names=['latents', 'valid_frames'],
                          output_names=['controls', 'oscillator_logits'],
                          dynamo=False, external_data=False)
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath)
    onnx.checker.check_model(str(path))
    session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    verify(model, mapper, session, latent)
    generator = torch.Generator().manual_seed(123)
    verify(model, mapper, session, torch.randn(1, 256, 66, generator=generator))
    if cfg.latent is not None:
        # Optional raw PT latent from our dataset, without normalization.
        latent = torch.load(cfg.latent, map_location='cpu', weights_only=True).float().unsqueeze(0)
        verify(model, mapper, session, latent)

    # Ava uses these two source hashes to check that generator and mapper share
    # the same latent space. Choose the bundle used for the training data.
    sampler = json.loads((Path(cfg.generator_dir) / 'sampler.json').read_text())
    save_json(output / 'mapper.json', {
        'format': 'ava-latent-mapper', 'version': 1, 'control_mapping': 'synth_v1',
        'feature_shape': [256, 66], 'controls': parameters, 'normalization': 'none',
        'onnx_sha256': checksum(path),
        'encoder': {'checkpoint_sha256': sampler['checkpoint_sha256'],
                    'config_sha256': sampler['config_sha256']},
    })
    copy_model_notices(output)
    print(f'Exported {path}; ONNX outputs match PyTorch reference cases')


if __name__ == '__main__':
    main()
