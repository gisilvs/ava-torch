"""Run Ava's exported text encoder and denoiser from Python."""
import json
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer


class LatentGenerator:
    def __init__(self, assets, device='cpu'):
        assets = Path(assets)
        self.settings = json.loads((assets / 'sampler.json').read_text())
        self.manifest = json.loads((assets / 'manifest.json').read_text())
        tokenizer_path = assets / 'tokenizer_hf.json'
        if not tokenizer_path.exists():
            # Older installed bundles contain only the native tokenizer binary.
            # Fetch its original Hugging Face JSON, not another text encoder.
            tokenizer_path = hf_hub_download(
                f'stabilityai/{self.manifest["model"]}', 't5gemma-b-b-ul2/tokenizer.json')
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_truncation(max_length=self.settings['text_length'])
        self.tokenizer.enable_padding(
            length=self.settings['text_length'], pad_id=self.tokenizer.token_to_id('<pad>'),
            pad_token='<pad>')
        provider = {'cpu': 'CPUExecutionProvider', 'cuda': 'CUDAExecutionProvider'}[device]
        if provider not in ort.get_available_providers():
            raise ValueError(f'{provider} is unavailable; install the matching ONNX Runtime package')
        if device == 'cuda':
            # Make CUDA/cuDNN libraries installed with PyTorch visible to ORT.
            ort.preload_dlls()
        options = ort.SessionOptions()
        options.enable_cpu_mem_arena = False
        if 'SLURM_CPUS_PER_TASK' in os.environ:
            # An explicit count disables ORT's automatic CPU-affinity settings.
            options.intra_op_num_threads = int(os.environ['SLURM_CPUS_PER_TASK'])
        self.text_encoder = ort.InferenceSession(
            str(assets / 'text_encoder.onnx'), options, providers=[provider])
        self.denoiser = ort.InferenceSession(
            str(assets / 'denoiser.onnx'), options, providers=[provider])
        # Provider initialization failures can otherwise silently revert to CPU.
        for session in (self.text_encoder, self.denoiser):
            if provider not in session.get_providers():
                raise RuntimeError(f'ONNX Runtime could not initialize {provider}')

    def encode(self, prompt):
        tokens = self.tokenizer.encode(prompt)
        context, global_cond = self.text_encoder.run(None, {
            'input_ids': np.array([tokens.ids], dtype=np.int64),
            'attention_mask': np.array([tokens.attention_mask], dtype=np.int64),
        })
        return {'cross_attention': context, 'global_cond': global_cond}

    def __call__(self, context, seed):
        # Use the plugin's saved schedule and ping-pong update. NumPy and C++
        # draw noise differently, so equal integer seeds do not imply equal audio.
        steps = np.asarray(self.settings['schedule'], dtype=np.float32)
        shape = (len(steps), 1, self.settings['channels'], self.settings['frames'])
        noise = np.random.default_rng(seed).standard_normal(shape).astype(np.float32)
        latent = noise[0].copy()
        for index in range(len(steps) - 1):
            velocity = self.denoiser.run(None, context | {
                'latent': latent, 'timestep': steps[index:index + 1],
            })[0]
            denoised = latent - steps[index] * velocity
            latent = (1 - steps[index + 1]) * denoised + steps[index + 1] * noise[index + 1]
        if not np.isfinite(latent).all():
            raise ValueError('Generator returned non-finite latents')
        return latent
