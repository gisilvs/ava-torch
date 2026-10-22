import shutil

import soundfile as sf
import torch

from synth.controls import decode_controls


@torch.inference_mode()
def save_audio_examples(model, dataset, device, count, output, parameters, synth):
    """Render the first held-out predictions and copy their reference WAVs."""
    output.mkdir(exist_ok=True)
    model.eval()
    examples = []
    for index in range(min(count, len(dataset))):
        latent, _, _ = dataset[index]
        controls, logits = model(latent.unsqueeze(0).to(device))
        row = dataset.labels[index]
        freq = torch.tensor([[row['frequency']]])
        preset = row.get('preset', {})
        physical = decode_controls(controls.cpu(), parameters, freq, synth.sample_rate, preset)
        recovered = preset | {name: value.item() for name, value in physical.items()}
        recovered['oscillator'], recovered['oscillator_b'] = logits.argmax(-1)[0].tolist()
        reference = dataset.folder / row['audio']
        duration = sf.info(reference).frames / synth.sample_rate
        # Preview rendering always uses CPU and hard oscillator choices.
        audio = synth(freq, duration, row['hold'], velocity=row['velocity'], **recovered)
        stem = f'{index:03d}'
        shutil.copyfile(reference, output / f'{stem}_target.wav')
        sf.write(output / f'{stem}_recovered.wav', audio.reshape(-1).numpy(),
                 synth.sample_rate, subtype='FLOAT')
        examples.append({'id': stem, 'frequency': row['frequency'], 'prompt': row.get('prompt'),
                         'target': row.get('preset'), 'recovered': recovered})
    return examples
