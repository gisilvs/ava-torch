import csv
import json
from pathlib import Path

import hydra
import soundfile as sf
import torch
from omegaconf import OmegaConf

from synth.controls import cutoff_bounds, decode_controls
from synth.synth import Synth


def fit(synth, preset, names, freq, duration, hold, velocity, steps, restarts, lr, seed,
        selector_training='hard', criterion=None):
    """Recover selected parameters from synthetic audio; other parameters stay fixed."""
    if selector_training not in ('hard', 'soft'):
        raise ValueError('selector_training must be hard or soft')
    generator = torch.Generator().manual_seed(seed)
    continuous = [name for name in names if name not in ('oscillator', 'oscillator_b')]
    selectors = [name for name in names if name in ('oscillator', 'oscillator_b')]
    logits = torch.nn.Parameter(torch.randn(restarts, len(continuous), generator=generator))
    selector_logits = {
        name: torch.nn.Parameter(torch.randn(restarts, 4, generator=generator))
        for name in selectors
    }
    trainable = ([logits] if continuous else []) + list(selector_logits.values())
    optimizer = torch.optim.Adam(trainable, lr=lr)

    # A fixed nonzero LFO depth narrows the cutoff range that can be fitted safely.
    lower = torch.zeros(1, len(continuous))
    upper = torch.ones_like(lower)
    if ('cutoff_octaves' in continuous and 'lfo_depth' not in continuous
            and preset['lfo_rate'] is not None):
        bounds = cutoff_bounds(freq, synth.sample_rate)
        margin = preset['lfo_depth'] / (bounds[1] - bounds[0])
        index = continuous.index('cutoff_octaves')
        lower[:, index:index + 1] = margin
        upper[:, index:index + 1] = 1 - margin

    with torch.no_grad():
        target = synth(freq, duration, hold, velocity=velocity, **preset)

    best_loss = float('inf')
    best_preset = None
    history = []
    for step in range(steps + 1):
        values = lower + (upper - lower) * logits.sigmoid()
        controls = decode_controls(values, continuous, freq, synth.sample_rate, preset)
        for name, raw in selector_logits.items():
            probability = raw.softmax(dim=-1)
            if selector_training == 'soft':
                controls[name] = probability
            else:
                hard = torch.nn.functional.one_hot(probability.argmax(-1), 4).to(probability)
                # Forward uses one oscillator; backward uses the softmax derivative.
                controls[name] = hard + (probability - probability.detach())
        current = preset | controls
        audio = synth(freq, duration, hold, velocity=velocity, **current)
        if criterion is None:
            losses = (audio - target).square().mean(dim=(1, 2))
        else:
            # Auraloss returns a batch scalar. Score restarts separately so they
            # stay independent and we can select the best one.
            losses = torch.stack([criterion(wave.unsqueeze(0), target) for wave in audio])
        loss, restart = losses.min(dim=0)
        history.append([step, loss.item(), losses.mean().item()])
        if step % 50 == 0 or step == steps:
            print(f'Step {step}: best restart loss = {loss.item():.6g}', flush=True)
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_preset = dict(preset)
            for name, value in controls.items():
                best_preset[name] = (value[restart].argmax().item() if name in selectors
                                     else value[restart].item())
        if step == steps:
            break
        optimizer.zero_grad()
        # Sum keeps each restart's gradient independent of the number of restarts.
        losses.sum().backward()
        optimizer.step()
    return best_preset, history


@hydra.main(version_base=None, config_path='../configs', config_name='fitting')
def main(cfg):
    torch.set_num_threads(cfg.threads)
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output / 'config.yaml', resolve=True)
    synth = Synth(**cfg.synth)
    if cfg.loss == 'mse':
        criterion = None
    elif cfg.loss == 'stft':
        criterion = hydra.utils.instantiate(cfg.stft).eval()
    else:
        raise ValueError('Choose loss mse or stft')
    preset = OmegaConf.to_container(cfg.preset, resolve=True)
    names = list(cfg.fit.parameters)
    if not names or len(names) != len(set(names)):
        raise ValueError('Choose at least one parameter, without duplicates')
    if any(preset[name] is None for name in names):
        raise ValueError('Enable a component in the preset before fitting its parameters')
    if preset['oscillator_b'] is None and any(name in names for name in ('mix', 'detune_cents')):
        raise ValueError('Mix and detuning fitting require oscillator_b')
    if preset['cutoff_octaves'] is None and any(name in names for name in ('q', 'lfo_rate', 'lfo_depth')):
        raise ValueError('Filter and LFO fitting require cutoff_octaves')
    if preset['lfo_rate'] is None and 'lfo_depth' in names:
        raise ValueError('LFO depth fitting requires lfo_rate')
    freq = torch.tensor([[440 * 2 ** ((cfg.note.midi_pitch - 69) / 12)]])
    recovered, history = fit(
        synth, preset, names, freq, cfg.note.duration, cfg.note.note_press_duration,
        cfg.note.velocity, cfg.fit.steps, cfg.fit.restarts, cfg.fit.lr, cfg.seed,
        cfg.fit.selector_training, criterion)

    with torch.no_grad():
        target = synth(freq, cfg.note.duration, cfg.note.note_press_duration,
                       velocity=cfg.note.velocity, **preset)
        audio = synth(freq, cfg.note.duration, cfg.note.note_press_duration,
                      velocity=cfg.note.velocity, **recovered)
        audio_loss = ((audio - target).square().mean().item() if criterion is None
                      else criterion(audio, target).item())
    mse = (audio - target).square().mean().item()
    report = {
        'target': preset,
        'recovered': recovered,
        'absolute_error': {name: abs(recovered[name] - preset[name]) for name in names
                           if name not in ('oscillator', 'oscillator_b')},
        'mse': mse,
        'selector_training': cfg.fit.selector_training,
        'loss': cfg.loss,
        'audio_loss': audio_loss,
        'best_training_loss': min(row[1] for row in history),
        'snr_db': (10 * torch.log10((target.square().mean() + 1e-12) / (mse + 1e-12))).item(),
        'torch_version': torch.__version__,
    }
    for name in ('oscillator', 'oscillator_b'):
        if name in names:
            report.setdefault('selector_correct', {})[name] = recovered[name] == preset[name]
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    with (output / 'loss.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['step', 'best_restart_loss', 'mean_restart_loss'])
        writer.writerows(history)
    for name, wave in (('target', target), ('recovered', audio)):
        sf.write(output / f'{name}.wav', wave.detach().reshape(-1).numpy(),
                 synth.sample_rate, subtype='FLOAT')
    print(json.dumps(report, indent=2))
    print(f'Results saved to {output.resolve()}')


if __name__ == '__main__':
    main()
