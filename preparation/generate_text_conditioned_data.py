import json
from collections import deque
from pathlib import Path

import hydra
import soundfile as sf
import torch
import torchcrepe
import torch.nn.functional as F
import torchaudio.functional as AF
from omegaconf import OmegaConf
from stable_audio_3.loading_utils import load_autoencoder
from stable_audio_3.model_configs import models

from networks.latent_generator import LatentGenerator
from preparation.generate_prompts import replacement_prompt


@torch.inference_mode()
def check_pitch(audio, sample_rate, midi_pitch, cfg):
    frequency = 440 * 2 ** ((midi_pitch - 69) / 12)
    if not cfg.fmin <= frequency <= cfg.fmax:
        raise ValueError('Requested pitch is outside the pitch tracker range')
    waveform = AF.resample(audio.reshape(1, -1).float().cpu(), sample_rate, 16000)
    pitch, confidence = torchcrepe.predict(
        waveform, 16000, hop_length=cfg.hop_samples, fmin=cfg.fmin, fmax=cfg.fmax,
        model='full', device=cfg.device, batch_size=cfg.batch_size, return_periodicity=True)
    pitch, confidence = pitch[0].cpu(), confidence[0].cpu()

    # Measure loudness on the same time grid as the pitch tracker. Ignore the
    # quiet tail of a pluck rather than requiring pitch throughout all six seconds.
    frames = F.pad(waveform, (512, 512)).unfold(1, 1024, cfg.hop_samples)
    rms = frames.square().mean(-1).sqrt()[0]
    # Defaults: active frames are within 40 dB of the loudest frame AND have
    # RMS >= 1e-5. The absolute floor also rejects an entirely silent clip.
    active = (rms >= rms.max() * 10 ** (cfg.relative_silence_db / 20)) & (rms >= cfg.rms_floor)
    # "Voiced" means active with a finite positive pitch and periodicity >= 0.8.
    # Periodicity is CREPE's confidence measure, not a calibrated probability.
    voiced = active & torch.isfinite(pitch) & (pitch > 0) & (confidence >= cfg.min_periodicity)
    voiced_frames = int(voiced.sum())
    active_frames = int(active.sum())

    # Cents measure musical distance: 100 cents is one semitone. Check the
    # fraction of voiced frames near the requested note, not just the median.
    cents = 1200 * torch.log2(pitch[voiced] / frequency)
    matched = float((cents.abs() <= cfg.tolerance_cents).float().mean()) if voiced_frames else 0.0
    coverage = voiced_frames / max(1, active_frames)
    # Accept only if ALL three checks pass (defaults below):
    # 1. At least 10 voiced frames: roughly 100 ms with a 160-sample hop at 16 kHz.
    # 2. At least 50% of active frames are voiced (voiced / active).
    # 3. At least 90% of voiced frames match the requested note within +/-50 cents.
    #    Matching is measured over voiced frames, not over the entire clip.
    accepted = (voiced_frames >= cfg.min_voiced_frames
                and coverage >= cfg.min_voiced_fraction
                and matched >= cfg.min_match_fraction)
    # Once accepted, render the synth at the median detected voiced pitch.
    # Acceptance uses the voiced pitch trajectory, not just this median.
    median = float(pitch[voiced].median()) if voiced_frames else None
    measurements = {'requested_hz': frequency, 'detected_hz': median,
                    'voiced_frames': voiced_frames, 'active_frames': active_frames,
                    'voiced_fraction': coverage, 'match_fraction': matched}
    return median if accepted else None, measurements


@torch.inference_mode()
def generate_sample(generator, context, decoder, config, sample, cfg):
    # The same quantized ONNX models produce latents here and in the plugin.
    latent = torch.from_numpy(generator(context, sample['seed']))
    transform = config['model']['pretransform']
    # SAME expects the original latent scale. Only the full-precision autoencoder
    # is loaded in Torch; the PyTorch text encoder and denoiser are not loaded.
    decoded = decoder.decode_audio(
        latent.to(cfg.device) * transform.get('scale', 1.0),
        chunked=transform.get('chunked', False)).float().cpu()
    native_samples = round(cfg.duration * config['sample_rate'])
    mono = decoded[0, :, :native_samples].mean(dim=0)
    audio = AF.resample(mono, config['sample_rate'], cfg.sample_rate)
    audio = audio[:round(cfg.duration * cfg.sample_rate)]
    if audio.numel() != round(cfg.duration * cfg.sample_rate) or not torch.isfinite(audio).all():
        raise ValueError('Decoder returned incomplete or non-finite audio')
    return latent[0], audio


@hydra.main(version_base=None, config_path='../configs', config_name='text_generation')
def main(cfg):
    folder = Path(cfg.data_dir)
    prompts = {
        split: json.loads((folder / split / 'prompts.json').read_text())
        for split in ('train', 'val', 'test')
    }
    prompt_cfg = OmegaConf.load(folder / 'prompt_config.yaml')
    rejected_path = folder / 'rejected_prompts.json'
    rejected = json.loads(rejected_path.read_text()) if rejected_path.exists() else []
    # Replacements must be unique across splits, including previously rejected text.
    seen = {row['prompt'] for rows in prompts.values() for row in rows}
    seen.update(row['prompt'] for row in rejected)
    # Retain accepted samples when restarting after a failed pitch check.
    saved = {}
    for split, rows in prompts.items():
        path = folder / split / 'labels.json'
        saved[split] = json.loads(path.read_text()) if path.exists() else []
        planned = {row['id']: row['prompt'] for row in rows}
        for label in saved[split]:
            stem = Path(label['audio']).stem
            if planned.get(stem) != label['prompt']:
                raise ValueError('Prompts changed; use a new data_dir')
            if not all((folder / split / label[key]).exists() for key in ('audio', 'latent')):
                raise ValueError(f'Missing saved audio or latent for {split}/{stem}')
    failure_path = folder / 'failed_sample.json'
    previous_failure = json.loads(failure_path.read_text()) if failure_path.exists() else {}
    OmegaConf.save(cfg, folder / 'config.yaml', resolve=True)
    # Reserve a separate sequence for retries so seeds never collide with the
    # original prompt list or with earlier retries.
    used_seeds = [row['seed'] for rows in prompts.values() for row in rows]
    used_seeds += [label['seed'] for labels in saved.values() for label in labels]
    used_seeds.append(previous_failure.get('seed', 0))
    retry_seed = max(used_seeds) + 1
    resuming = any(saved.values()) or bool(previous_failure)
    if resuming and not previous_failure:
        # Older runs did not save the final rejected seed. Step past that budget.
        retry_seed += cfg.max_attempts_per_sample
    generator = LatentGenerator(cfg.generator_dir, cfg.generator_device)
    if cfg.duration != generator.settings['duration']:
        raise ValueError('Duration must match the exported generator')
    # Resolve the original autoencoder from the model named by the asset bundle.
    model_name = generator.manifest['model'].removeprefix('stable-audio-3-')
    config_path, weights = models[model_name].resolve()
    config = json.loads(Path(config_path).read_text())
    decoder = load_autoencoder(config_path, weights, cfg.device)
    decoder.eval().requires_grad_(False)

    for split, rows in prompts.items():
        output = folder / split
        (output / 'latents').mkdir(exist_ok=True)
        labels = saved[split]
        completed = {Path(label['audio']).stem for label in labels}
        print(f'{split}: {len(labels)}/{len(rows)} already accepted', flush=True)
        pending = deque(row for row in rows if row['id'] not in completed)
        next_id = max(int(row['id']) for row in rows) + 1 if rows else 0
        while pending:
            row = pending.popleft()
            sample = dict(row)
            # Reuse text conditioning when retrying the same prompt.
            context = generator.encode(sample['prompt'])
            for attempt in range(cfg.max_attempts_per_sample):
                if attempt > 0 or resuming:
                    sample['seed'] = retry_seed
                    retry_seed += 1
                latent, audio = generate_sample(generator, context, decoder, config, sample, cfg)
                frequency, measurements = check_pitch(audio, cfg.sample_rate, sample['midi_pitch'], cfg.pitch)
                if frequency is not None:
                    break
            else:
                # Keep the failure for inspection, replace this prompt, and try
                # its replacement after the remaining queue. Split size stays fixed.
                (output / 'labels.json').write_text(json.dumps(labels, indent=2) + '\n')
                failure = sample | {'split': split, 'measurements': measurements,
                                    'pitch_settings': OmegaConf.to_container(cfg.pitch, resolve=True)}
                failure_path.write_text(json.dumps(failure, indent=2) + '\n')
                sf.write(folder / 'failed_sample.wav', audio.numpy(), cfg.sample_rate, subtype='FLOAT')
                rejected.append(failure)
                rejected_path.write_text(json.dumps(rejected, indent=2) + '\n')
                replacement = replacement_prompt(prompt_cfg, seen, f'{next_id:06d}', retry_seed)
                next_id += 1
                retry_seed += 1
                seen.add(replacement['prompt'])
                rows.remove(row)
                rows.append(replacement)
                pending.append(replacement)
                (output / 'prompts.json').write_text(json.dumps(rows, indent=2) + '\n')
                print(f'{split}/{sample["id"]}: pitch check failed; '
                      f'queued new prompt {replacement["id"]} at the end', flush=True)
                continue

            stem = sample['id']
            torch.save(latent, output / 'latents' / f'{stem}.pt')
            sf.write(output / f'{stem}.wav', audio.numpy(), cfg.sample_rate, subtype='FLOAT')
            labels.append({
                'latent': f'latents/{stem}.pt', 'audio': f'{stem}.wav',
                'frequency': frequency, 'hold': cfg.hold, 'velocity': 1.0,
                'prompt': sample['prompt'], 'seed': sample['seed'],
            })
            if len(labels) % 100 == 0 or len(labels) == len(rows):
                (output / 'labels.json').write_text(json.dumps(labels, indent=2) + '\n')
            if len(labels) % 10 == 0 or len(labels) == len(rows):
                print(f'{split}: {len(labels)}/{len(rows)} accepted', flush=True)


if __name__ == '__main__':
    main()
