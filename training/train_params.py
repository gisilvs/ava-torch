import json
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from datasets.data import SynthDataset
from networks.mapper import ParameterTransformer
from synth.synth import Synth
from training.utils import save_audio_examples


def run_epoch(model, loader, device, selector_weight, optimizer=None, ema=None):
    training = optimizer is not None
    model.train(training)
    totals = dict.fromkeys(('loss', 'mse', 'ce', 'accuracy_a', 'accuracy_b', 'pair_accuracy'), 0.0)
    parameter_error = torch.zeros(len(loader.dataset.parameters))
    with torch.set_grad_enabled(training):
        for latent, target, labels in loader:
            latent = latent.to(device)
            target, labels = target.to(device), labels.to(device)
            controls, logits = model(latent)
            mse = F.mse_loss(controls, target)
            # Flattening averages CE over both selectors and all batch examples.
            ce = F.cross_entropy(logits.reshape(-1, 4), labels.reshape(-1))
            loss = mse + selector_weight * ce
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if ema is not None:
                    ema.update_parameters(model)
            correct = logits.argmax(-1) == labels
            metrics = {'loss': loss, 'mse': mse, 'ce': ce,
                       'accuracy_a': correct[:, 0].float().mean(),
                       'accuracy_b': correct[:, 1].float().mean(),
                       'pair_accuracy': correct.all(dim=1).float().mean()}
            for name, value in metrics.items():
                totals[name] += value.detach().item() * len(latent)
            if not training:
                parameter_error += (controls.detach() - target).square().sum(dim=0).cpu()
    result = {name: value / len(loader.dataset) for name, value in totals.items()}
    if not training:
        rmse = (parameter_error / len(loader.dataset)).sqrt()
        result['parameter_rmse'] = dict(zip(loader.dataset.parameters, rmse.tolist()))
    return result


@hydra.main(version_base=None, config_path='../configs', config_name='training')
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
    OmegaConf.save(cfg, output / 'config.yaml', resolve=True)

    datasets = {split: SynthDataset(cfg.data_dir, split, cfg.parameters)
                for split in ('train', 'val', 'test')}
    parameters = datasets['train'].parameters
    data_config = OmegaConf.load(Path(cfg.data_dir) / 'config.yaml')
    synth = Synth(**data_config.synth)

    # Only training is shuffled, with its own reproducible random generator.
    shuffle_generator = torch.Generator().manual_seed(cfg.seed)
    loaders = {
        split: DataLoader(data, batch_size=cfg.batch_size, shuffle=split == 'train',
                          generator=shuffle_generator if split == 'train' else None)
        for split, data in datasets.items()
    }
    feature_shape = list(datasets['train'][0][0].shape)
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

    best_loss, best_mse = float('inf'), float('inf')
    history = []
    for epoch in range(1, cfg.epochs + 1):
        train = run_epoch(model, loaders['train'], cfg.device, cfg.selector_weight, optimizer, ema)
        val = run_epoch(ema.module, loaders['val'], cfg.device, cfg.selector_weight)
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
        if val['mse'] < best_mse:
            best_mse = val['mse']
            torch.save(checkpoint, output / 'best_parameter_mse.pt')
        print(f'Epoch {epoch}: train MSE {train["mse"]:.6f}, val EMA MSE {val["mse"]:.6f}, '
              f'val CE {val["ce"]:.4f}, val pair accuracy {val["pair_accuracy"]:.1%}', flush=True)

    # Save the final EMA too, even if an earlier epoch was best.
    torch.save(checkpoint, output / 'ema.pt')
    selected = 'best_parameter_mse.pt'
    checkpoint = torch.load(output / selected, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model'])
    test = run_epoch(model, loaders['test'], cfg.device, cfg.selector_weight)
    report = {'weights': 'ema', 'objective': 'parameters', 'checkpoint': selected, 'epoch': checkpoint['epoch'],
              'parameters': parameters, 'validation': checkpoint['validation'], 'test': test}
    report['audio_examples'] = save_audio_examples(
        model, datasets['test'], cfg.device, cfg.audio_examples, output / 'audio', parameters, synth)
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
