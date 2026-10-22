import json
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from datasets.audio_data import AudioDataset
from datasets.data import SynthDataset
from networks.mapper import ParameterTransformer
from synth.synth import Synth
from training.train_params import run_epoch as parameter_metrics
from training.utils import save_audio_examples


class AudioLoss(torch.nn.Module):
    """Use waveform MSE, or multi-resolution STFT with optional log-mel L1."""

    def __init__(self, cfg):
        super().__init__()
        if cfg.loss not in ('mse', 'stft', 'stft_mel'):
            raise ValueError('Choose loss mse, stft or stft_mel')
        self.base = (torch.nn.MSELoss() if cfg.loss == 'mse'
                     else hydra.utils.instantiate(cfg.stft))
        self.mel = hydra.utils.instantiate(cfg.mel) if cfg.loss == 'stft_mel' else None
        self.mel_weight = cfg.mel_weight

    def forward(self, audio, target):
        loss = self.base(audio, target)
        if self.mel is not None:
            loss = loss + self.mel_weight * self.mel(audio, target)
        return loss


def render_prediction(synth, controls, logits, notes, parameters, duration,
                      training=False, selector_training='soft'):
    # Evaluation always selects one oscillator per head.
    weights = F.one_hot(logits.argmax(-1), 4).to(controls)
    if training:
        probability = logits.softmax(dim=-1)
        if selector_training == 'soft':
            weights = probability
        else:
            # Hard forward choices, with gradients through soft probabilities.
            weights = weights + (probability - probability.detach())
    return synth.from_controls(
        controls, parameters, notes[:, :1], duration, notes[:, 1:2],
        velocity=notes[:, 2:3], oscillator=weights[:, 0], oscillator_b=weights[:, 1])


def run_audio_epoch(model, loader, device, parameters, synth, criterion,
                    selector_training, gradient_clip, optimizer=None, ema=None):
    training = optimizer is not None
    model.train(training)
    # Copies between devices preserve gradients; MPS cannot run the float64 filter.
    render_device = 'cpu' if device == 'mps' else device
    total_loss, total_mse = 0.0, 0.0
    with torch.set_grad_enabled(training):
        for latent, target, notes in loader:
            controls, logits = model(latent.to(device))
            target, notes = target.to(render_device), notes.to(render_device)
            audio = render_prediction(
                synth, controls.to(render_device), logits.to(render_device), notes,
                parameters, target.shape[-1] / synth.sample_rate, training, selector_training)
            loss = criterion(audio, target)
            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
                if ema is not None:
                    ema.update_parameters(model)
            total_loss += loss.detach().item() * len(latent)
            # Diagnostic only: waveform MSE is not added to the STFT objective.
            total_mse += F.mse_loss(audio.detach(), target).item() * len(latent)
    return {'loss': total_loss / len(loader.dataset),
            'audio_mse': total_mse / len(loader.dataset)}


@hydra.main(version_base=None, config_path='../configs', config_name='audio_training')
def main(cfg):
    initial = None
    if cfg.init_checkpoint is not None:
        initial = torch.load(cfg.init_checkpoint, map_location='cpu', weights_only=True)
        if initial.get('latent_normalization') != 'none':
            raise ValueError('Use a checkpoint trained without latent normalization')
        # Preserve the pretrained architecture and output-column meaning.
        cfg.network = initial['config']['network']
        cfg.parameters = initial['parameters']
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    datasets = {split: AudioDataset(cfg.data_dir, split) for split in ('train', 'val', 'test')}
    if cfg.epochs is None:
        # Synthetic labels contain a preset; text-generated pairs do not.
        cfg.epochs = 50 if 'preset' in datasets['train'].labels[0] else 200
    OmegaConf.save(cfg, output / 'config.yaml', resolve=True)
    parameters = list(cfg.parameters)
    if len(parameters) != 13:
        raise ValueError('Audio training requires all 13 continuous controls')
    if cfg.selector_training not in ('hard', 'soft'):
        raise ValueError('Choose selector_training hard or soft')
    synth = Synth(**cfg.synth)
    render_device = 'cpu' if cfg.device == 'mps' else cfg.device
    criterion = AudioLoss(cfg).to(render_device)

    # Only training is shuffled, with its own reproducible random generator.
    shuffle_generator = torch.Generator().manual_seed(cfg.seed)
    loaders = {
        split: DataLoader(data, batch_size=cfg.batch_size, shuffle=split == 'train',
                          generator=shuffle_generator if split == 'train' else None)
        for split, data in datasets.items()
    }
    validation_parameters = None
    if 'preset' in datasets['val'].labels[0]:
        # This extra pass runs only the mapper, not the synth. Its labels never
        # enter backpropagation or change which audio checkpoint is selected.
        validation_parameters = DataLoader(
            SynthDataset(cfg.data_dir, 'val', parameters), batch_size=cfg.batch_size,
            generator=torch.Generator().manual_seed(cfg.seed))
    feature_shape = list(datasets['train'][0][0].shape)
    # Loss construction must not change the mapper's random initialization.
    torch.manual_seed(cfg.seed)
    model = ParameterTransformer(feature_shape, controls=len(parameters), **cfg.network).to(cfg.device)
    if initial is not None:
        model.load_state_dict(initial['model'])
    # Keep a separate moving average; only the live model receives gradients.
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(cfg.ema_decay))
    ema.module.requires_grad_(False)
    ema.update_parameters(model)  # Start the average from the initialized weights.
    # A new optimizer is intentional: checkpoint initialization is fine-tuning, not resume.
    optimizer = torch.optim.AdamW(model.parameters(), **cfg.optimizer)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')

    def epoch_metrics(network, loader, optimizer=None):
        return run_audio_epoch(network, loader, cfg.device, parameters, synth, criterion,
                               cfg.selector_training, cfg.gradient_clip, optimizer, ema)

    best_loss = float('inf')
    history = []
    for epoch in range(1, cfg.epochs + 1):
        train = epoch_metrics(model, loaders['train'], optimizer)
        val = epoch_metrics(ema.module, loaders['val'])
        if validation_parameters is not None:
            recovery = parameter_metrics(
                ema.module, validation_parameters, cfg.device, cfg.selector_weight)
            val['parameter_recovery'] = {
                key: recovery[key] for key in
                ('mse', 'accuracy_a', 'accuracy_b', 'pair_accuracy', 'parameter_rmse')}
        history.append({'epoch': epoch, 'train': train, 'val': val})
        (output / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
        checkpoint = {'model': ema.module.state_dict(), 'weights': 'ema',
                      'latent_normalization': 'none',
                      'feature_shape': feature_shape, 'parameters': parameters,
                      'control_mapping': 'synth_v1', 'epoch': epoch, 'validation': val,
                      'config': OmegaConf.to_container(cfg, resolve=True)}
        if val['loss'] < best_loss:
            best_loss = val['loss']
            torch.save(checkpoint, output / 'best.pt')
        print(f'Epoch {epoch}: train audio loss {train["loss"]:.6f}, '
              f'val EMA audio loss {val["loss"]:.6f}, val waveform MSE {val["audio_mse"]:.6f}', flush=True)

    # Audio models are selected by their hard-choice validation audio loss.
    # Save the final EMA too, even if an earlier epoch was best.
    torch.save(checkpoint, output / 'ema.pt')
    selected = 'best.pt'
    checkpoint = torch.load(output / selected, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model'])
    test = epoch_metrics(model, loaders['test'])
    report = {'weights': 'ema', 'objective': 'audio', 'checkpoint': selected, 'epoch': checkpoint['epoch'],
              'parameters': parameters, 'validation': checkpoint['validation'], 'test': test}
    if 'preset' in datasets['test'].labels[0]:
        # Known synthetic labels are evaluation only, never part of the audio objective.
        labeled = SynthDataset(cfg.data_dir, 'test', parameters)
        report['parameter_recovery'] = parameter_metrics(
            model, DataLoader(labeled, batch_size=cfg.batch_size), cfg.device, cfg.selector_weight)
    report['audio_examples'] = save_audio_examples(
        model, datasets['test'], cfg.device, cfg.audio_examples, output / 'audio', parameters, synth)
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
