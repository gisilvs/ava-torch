# Ava — PyTorch

The differentiable synthesizer and training pipeline for Ava, a synthesizer that
turns text descriptions into editable presets.

[Read the blog](https://gisilvs.github.io/blog/ava/) ·
[C++ plugin](https://github.com/gisilvs/ava-cpp)

Run the commands below from this repository's root. To reproduce the full
experiment, follow the sections in order; datasets and training outputs are
created locally, not downloaded with the code.

## Installation

```bash
conda create -n ava-torch python=3.11 pip
conda activate ava-torch
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

## Repository layout

```text
synth/          Differentiable synth and its components
networks/       Mapper architecture and ONNX latent generator
preparation/    Generate sounds, build prompts and encode datasets
datasets/       Dataset loaders used during training
training/       Parameter supervision, audio loss and shared audio previews
fitting/        Direct parameter-recovery experiment
export/         ONNX export, INT8 quantization and Ava metadata
configs/        Hydra settings
```

## Check parameter recovery

Before training a network, we check whether the synth can recover the settings
used to create a sound. We generate a target, start from random parameter values,
and adjust them to match the audio using waveform MSE.

Run the commands from the repository root. Start by recovering the gain of a
sine wave:

```bash
python -m fitting.fit
```

This runs on CPU and creates a six-second note, held for four seconds. It tries
four random starting points for 200 optimization steps. The terminal prints the
results and the output folder. Listen to `target.wav` and `recovered.wav`, then
check `report.json` to compare the original and recovered values.

To use the same spectral loss as the later audio-training stage, add `loss=stft`:

```bash
python -m fitting.fit loss=stft
```

This works with any fitting command below, including the full-synth test. Use
`loss=mse` for waveform MSE, which remains the default. The spectral settings are
shared in `configs/stft.yaml`.

Next, try recovering gain, attack and release together:

```bash
python -m fitting.fit preset.gain=0.6 preset.attack=0.1 preset.release=0.8 \
  'fit.parameters=[gain,attack,release]'
```

The `preset` values describe the target sound. `fit.parameters` lists what the
optimizer needs to recover; the remaining settings stay fixed. You can change
these values or choose other parameters in `configs/fitting.yaml`.

Finally, fit the whole synth: both oscillator choices and all 13 continuous
parameters, with the filter, LFO, noise and distortion enabled:

```bash
python -m fitting.fit \
  preset.oscillator=2 preset.oscillator_b=1 \
  preset.gain=0.5 preset.mix=0.35 preset.detune_cents=12 \
  preset.noise_mix=0.05 preset.drive=1.5 \
  preset.cutoff_octaves=2 preset.q=1.2 \
  preset.lfo_rate=2 preset.lfo_depth=0.4 \
  preset.attack=0.1 preset.decay=0.4 preset.sustain=0.6 preset.release=0.8 \
  'fit.parameters=[gain,mix,detune_cents,noise_mix,cutoff_octaves,q,lfo_rate,lfo_depth,drive,attack,decay,sustain,release,oscillator,oscillator_b]' \
  fit.steps=500 fit.restarts=4 fit.selector_training=soft \
  synth.lfo_control_rate=24000
```

Here the target mixes a sawtooth (`2`) and a square (`1`); the other choices are
sine (`0`) and triangle (`3`). Soft selection blends oscillators during fitting,
then chooses one per oscillator for the final preset. Use
`fit.selector_training=hard` to try straight-through selection instead.
The LFO updates every sample, matching dataset generation.

Each run saves its results under `outputs/fit/`. Alongside the audio and parameter
report, `loss.csv` records the fitting progress. You can append `fit.steps=1000`
for a longer run, `seed=42` for different initial values, or
`output_dir=outputs/my_fit` to choose the output folder.

The report includes the chosen `loss`, its final `audio_loss`, and waveform MSE
and SNR for comparison. With soft oscillator selection, `best_training_loss`
measures the blended sound, while `audio_loss` measures the final hard choices.


## Generate synthetic data

Next, create the sounds the network will learn from. Each sound gets a random
preset and a note between MIDI 21 and 78. We use six-second clips: four seconds
with the key held, followed by two seconds for the sound to fade out.

Start with a small dataset:

```bash
python -m preparation.generate_data device=cpu
```

Use `device=cpu` on a Mac or a machine without an NVIDIA GPU. On an NVIDIA GPU
machine, use `device=cuda`. MPS is not supported for this step because the filter
needs float64 operations.

CUDA generation requires TorchLPC to be built with the CUDA 12.8 toolkit available
inside a GPU allocation; installing it on a login node can produce a CPU-only build.

This saves 1,000 training, 100 validation and 100 test sounds under `data/synth`.
Each split contains mono WAVs at 24 kHz and a `labels.json` file with the preset
and note settings for each sound. The 16 oscillator pairs are balanced within
each split; the other parameters are sampled independently.

To change the dataset size, set the three counts. For example:

```bash
python -m preparation.generate_data device=cpu \
  dataset.counts.train=100000 dataset.counts.val=10000 dataset.counts.test=10000
```

That creates 120,000 sounds, using roughly 69 GB for audio before adding latents.
Use `data_dir=data/my_dataset` to choose another folder. Generation starts fresh
rather than resuming, so use a new folder to keep an earlier dataset.

The parameter ranges are in `configs/dataset.yaml`:

| Parameter | Range | Sampling |
|---|---|---|
| Gain | 0.25–0.8 | Linear |
| Oscillator mix | 0.2–0.8 | Linear |
| Attack | 0.01–0.5 s | Logarithmic |
| Decay | 0.05–1 s | Logarithmic |
| Sustain | 0.25–0.9 | Linear |
| Release | 0.1–1.8 s | Logarithmic |
| Cutoff above the note | 1–3 octaves | Linear |
| Resonance Q | 0.5–2.5 | Logarithmic |
| LFO rate | 0.5–8 Hz | Logarithmic |
| LFO depth | 0.1–0.75 octaves | Linear |
| Second-oscillator detune | 3–25 cents | Linear |
| Noise mix | 0–0.2 | Linear |
| Drive | 0.1–4 | Linear |

These ranges keep both oscillators audible and avoid nearly silent examples.
They describe the training data, not the limits of the synth. Cutoff leaves room
for the LFO sweep, which is evaluated at every sample.

## Encode and train on parameters

Encode the synthetic audio with SAME, then train the latent-to-parameter mapper:

```bash
python -m preparation.encode_data device=mps
python -m training.train_params device=mps
```

Use `mps` on Apple Silicon, `cuda` on NVIDIA, or `cpu`. Both commands read
`data/synth`; add `data_dir=...` to both if you generated another dataset.
Encoding saves unnormalized `[256, 66]` float32 tensors in each split's `latents/`
folder.

The default transformer predicts 13 continuous controls and two oscillator
choices. Training uses normalized parameter MSE plus `0.01` times the mean
cross-entropy of the two selectors. Defaults in `configs/training.yaml` are
200 epochs, batch size 32, AdamW at `1e-3`, seed 123 and EMA decay `0.999`.

Results go to `outputs/train_params/<date>/<time>/`, or your chosen `output_dir`.
`best.pt` selects the lowest combined validation loss; `best_parameter_mse.pt`
selects the lowest parameter MSE and is used for test evaluation and audio
examples. Both contain EMA weights; `ema.pt` contains the final EMA weights.
Learning curves are in `history.json`, test metrics in `report.json`, and up to
ten target/reconstructed WAV pairs in `audio/`. These checkpoints also work for
fine-tuning and ONNX export.

## Included weights

`weights/text_audio.pt` is the best text-audio model used in the blog: trained
from scratch at `1e-3`, selected at epoch 197 by validation audio loss. Its
442,005 parameters are EMA weights. `weights/parameter_supervised.pt` is the
parameter-supervised initialization used in the comparison, selected at epoch
183. Both use raw SAME-S `[256, 66]` latents and the same 13 synth controls.
Hashes, architecture and validation metrics are in `weights/metadata.json`.

You can export the final model without retraining, once the generator export
below is available:

```bash
python -m export.export_mapper checkpoint=weights/text_audio.pt
```

For fine-tuning, pass either checkpoint as `init_checkpoint`. The large Stable
Audio generator weights are not stored in Git: obtain access to
[Stable Audio 3 Small SFX](https://huggingface.co/stabilityai/stable-audio-3-small-sfx),
accept its model terms and log in with `hf auth login` before encoding or
exporting. The code downloads the upstream weights through Hugging Face.
See its [license](https://huggingface.co/stabilityai/stable-audio-3-small-sfx/blob/main/LICENSE.md)
and [Gemma terms](https://ai.google.dev/gemma/terms), which also apply to its
text encoder. Generation and export need several GB of free RAM and disk.

## Export the text-to-latent generator

We use Stable Audio's pretrained text encoder and denoiser to turn a prompt into
an audio latent. The denoiser is the latent generative model: conditioned on the
text encoder's output, it progressively transforms random noise into a
six-second audio latent over eight sampling steps.
Our mapper then converts that latent into synth parameters.

Here we export the two pretrained networks to ONNX, the format loaded by the
C++ plugin, and quantize their matrix weights to INT8 to reduce model size.
We quantize now because this is the format we will use at deployment. The next
stage generates training data with these INT8 networks, so the mapper learns
from the same generator it will receive latents from in the plugin. Quantization
can change the generated latents slightly; doing it before data generation lets
the mapper learn from those changes instead of introducing them only after
training. The reference audio is decoded from those same latents.

Run these commands in order, as separate processes to limit memory use:

```bash
# Export the text encoder, tokenizer and a PyTorch reference output.
python -m export.export_generator stage=text
# Check the exported text encoder against that reference.
python -m export.export_generator stage=verify-text
# Export the denoiser and save the eight-step sampling settings.
python -m export.export_generator stage=denoiser
# Check one denoising step against its PyTorch reference.
python -m export.export_generator stage=verify-denoiser
# Create an INT8 copy of both networks, keeping the FP32 originals.
python -m export.export_generator stage=quantize
```

The next step loads the quantized generator from `outputs/generator_int8`.
If you move it to another machine, copy the whole folder. Export paths are set
in `configs/generator_export.yaml`.

The generator produces latents, not WAV files. During data preparation, we use
Stable Audio's original audio decoder to turn those latents into reference WAVs
for the audio loss. In the plugin, our mapper turns the latents into synth
parameters instead, so the audio decoder is not needed there.


## Prepare text-conditioned data

First create and inspect fixed prompt lists:

```bash
python -m preparation.generate_prompts counts.train=10000 counts.val=1000 counts.test=1000
```

`configs/prompts.yaml` contains instrument-specific descriptions such as a warm
lo-fi piano or a distorted electric guitar, optional everyday modifiers and pitch
ranges. A sample combines an instrument description, an optional modifier and a
random pitch in its range, wrapped as a single isolated synth note.

Every complete prompt is unique across the training, validation and test sets.
Similar descriptions, such as a warm piano at different pitches, can still
appear in different sets. When a user supplies
their own description, we insert it into the same single-note template without
adding random modifiers.

Each set's list is saved as `prompts.json` inside `data/text/train/`,
`data/text/val/` or `data/text/test/`.

Generate their latents and reference audio:

```bash
python -m preparation.generate_text_conditioned_data generator_device=cpu device=mps
```

This uses the saved prompts and INT8 generator to save six-second mono WAVs at
24 kHz and their corresponding latents. The command above is for a Mac; use
`generator_device=cuda device=cuda` on NVIDIA, or both `cpu` without acceleration.

Each sample must pass a pitch check; failed attempts are regenerated, up to 20
times per prompt. If all attempts fail, a new unique prompt is added at the end
of that split's queue, keeping the requested sample count. Training uses the detected pitch and assumes a four-second
key hold. Settings and paths are in `configs/text_generation.yaml`.
Rerunning keeps accepted samples and the updated prompt lists. Rejected prompts
are recorded in `rejected_prompts.json`; `failed_sample.wav` and
`failed_sample.json` contain the latest rejected sound and its pitch measurements.

## Train through the audio loss

Here we train the mapper by rendering its predicted presets and comparing the
audio with the reference recordings. Both synthetic and text-generated data use
the same loss: long multi-resolution STFT (FFTs 32–16384) plus log-mel L1 at weight 1.

To train from scratch on the synthetic data:

```bash
python -m training.train_audio device=mps
```

To fine-tune on the text-generated data, start from your parameter-trained
checkpoint. Replace `DATE/TIME` with your pretraining run's folder:

```bash
python -m training.train_audio device=mps data_dir=data/text \
  init_checkpoint=outputs/train_params/DATE/TIME/best_parameter_mse.pt
```

Either dataset can be used with or without pretraining: omit `init_checkpoint`
to start from scratch. Loading a checkpoint keeps its architecture and weights
but starts a fresh optimizer. Choose `cuda` for NVIDIA or `cpu` without a GPU;
with `mps`, the mapper runs on the GPU while synthesis runs on CPU.

Defaults are batch size 128, AdamW at `1e-4`, and 50 epochs for synthetic data or
200 for text-generated data. Append `optimizer.lr=1e-3` to try the higher learning
rate, or use `batch_size=...` and `epochs=...` to change the training budget.

During training, the model blends oscillator choices so gradients can flow
through them. Use `selector_training=hard` to try straight-through selection
instead. Validation and listening examples always use argmax choices.
Known synthetic parameters are measured for comparison, not added to the loss.

Validation uses EMA weights (`ema_decay=0.999`). Results go to
`outputs/train_audio/DATE/TIME/`: `best.pt` contains the EMA weights with the lowest
validation audio loss, while `ema.pt` keeps the final epoch's EMA weights.
The best checkpoint is used for the test report and up to ten reference/reconstructed
audio pairs. Other settings are in `configs/audio_training.yaml`.

## Export the mapper for Ava

Once you are happy with the listening examples, export the checkpoint you want
to use in the plugin. For an audio-trained model, use:

```bash
python -m export.export_mapper \
  checkpoint=outputs/train_audio/DATE/TIME/best.pt
```

Replace `DATE/TIME` with your run's folder, or give the path to a downloaded
checkpoint. Use `ema.pt` for the final epoch, or a parameter-trained
`best_parameter_mse.pt` if you prefer that model.

The command saves `mapper.onnx` and `mapper.json` in `outputs/mapper/` and checks
that ONNX reproduces the PyTorch outputs. Add
`latent=data/synth/test/latents/000000.pt` to check a saved dataset latent too.
This checks the conversion; use the saved audio examples to judge the sound.
The export reads `outputs/generator_int8/` to identify the generator's latent
space. If you used another bundle for the data, pass its path as `generator_dir`.

## Package for the C++ plugin

Ava needs two parts: the INT8 generator bundle in `outputs/generator_int8/` and
the exported mapper in `outputs/mapper/`. Keep the generator folder intact,
including its weights, tokenizer and sampling settings. The audio decoder and
pitch tracker are only used to prepare training data; they are not shipped.

With [ava-cpp](https://github.com/gisilvs/ava-cpp) cloned next to this repository,
build and package on Apple Silicon macOS:

```bash
cd ../ava-cpp
export AVA_MAPPER_DIR="$PWD/../ava-torch/outputs/mapper"
export AVA_GENERATOR_ASSETS="$PWD/../ava-torch/outputs/generator_int8"
bash scripts/build_plugin.sh
bash scripts/package_macos.sh
```

The build puts the mapper inside the plugin. Packaging collects the plugin,
generator and installer in `build/release/Ava-0.1.0-macos-arm64/`. That folder
contains the AU, VST3 and standalone versions; run its `install.sh` to install
them and the shared generator assets. Python is not needed to use the plugin.
If the package folder already exists, move it aside before packaging again.

## License

The Python source is [MIT-licensed](LICENSE), copyright 2024–2026 Gianluigi
Silvestri. The included mapper weights have [separate model terms](weights/LICENSE.md).

**Powered by Stability AI.** Stable Audio, SAME-S and the T5Gemma text encoder
retain their upstream licenses, including after ONNX export and quantization.
See [third-party notices](THIRD_PARTY_NOTICES.md). Generator exports include the
required model agreements and notices; keep them with the exported files.
