import torch

from .controls import decode_controls
from .effects.distortion import SoftDistortion
from .envelopes.adsr import ADSREnvelope
from .filters.lowpass import LowPassFilter
from .modulation.lfo import SineLFO
from .oscillators.bank import OscillatorBank
from .oscillators.white_noise import WhiteNoise
from .validation import check_lfo_controls, check_note


class Synth:
    """Render physical synth parameters; frequency and controls are [batch, 1]."""

    def __init__(self, sample_rate=24000, noise_seed=0, lfo_control_rate=200):
        if not 0 < lfo_control_rate <= sample_rate:
            raise ValueError('LFO control rate must be positive and no higher than sample rate')
        self.sample_rate = sample_rate
        # At 24 kHz / 200 Hz, hold each filter coefficient set for 120 samples.
        self.lfo_block_size = max(1, round(sample_rate / lfo_control_rate))
        self.oscillators = OscillatorBank(sample_rate)
        self.noise = WhiteNoise(noise_seed)
        self.lowpass = LowPassFilter(sample_rate)
        self.lfo = SineLFO()
        self.distortion = SoftDistortion()
        self.envelope = ADSREnvelope()

    def __call__(self, freq, duration, note_press_duration, gain, oscillator=0, oscillator_b=None,
                 mix=0.5, detune_cents=0.0, velocity=1.0, noise_mix=0.0,
                 cutoff_octaves=None, q=0.707, lfo_rate=None, lfo_depth=0.0,
                 drive=0.0, attack=0.01, decay=0.2, sustain=1.0, release=1.5):
        check_note(freq, duration, note_press_duration, self.sample_rate)
        # Preserve the original float32 time grid before matching the input dtype.
        time = torch.arange(round(duration * self.sample_rate),
                            device=freq.device, dtype=torch.float32) / self.sample_rate
        time = time.to(freq)
        # Both oscillator selections accept an integer ID or [batch, 4] tensor of weights.
        wave = self.oscillators(freq, time, oscillator)
        if oscillator_b is not None:
            freq_b = freq * 2 ** (detune_cents / 1200)
            wave_b = self.oscillators(freq_b, time, oscillator_b)
            wave = (1 - mix) * wave + mix * wave_b
        wave = (1 - noise_mix) * wave + noise_mix * self.noise(time)
        if lfo_rate is not None and cutoff_octaves is None:
            raise ValueError('Cutoff LFO requires an enabled filter')
        if cutoff_octaves is not None:
            # The base cutoff is an octave offset from the note, not a frequency in Hz.
            offset = torch.as_tensor(cutoff_octaves, device=freq.device, dtype=torch.float64)
            # A missing LFO rate selects the static filter path.
            modulated = lfo_rate is not None
            if modulated:
                rate = torch.as_tensor(lfo_rate, device=freq.device, dtype=freq.dtype)
                depth = torch.as_tensor(lfo_depth, device=freq.device)
                check_lfo_controls(rate, depth, offset, freq, self.sample_rate)
                # Evaluate the LFO only when the filter coefficients need updating.
                update_times = time[::self.lfo_block_size]
                # [batch, 1] center + [batch, updates] sine offsets, both in octaves.
                offset = offset + self.lfo(update_times, rate, depth)
            # Convert octaves to Hz: +1 octave doubles the cutoff frequency.
            cutoff_hz = freq.double() * 2 ** offset
            resonance = torch.as_tensor(q, device=freq.device, dtype=torch.float64)
            # The filter handles static or varying coefficients internally;
            # recursive state stays continuous across coefficient updates.
            wave = self.lowpass(wave, cutoff_hz, resonance,
                                modulated=modulated, block_size=self.lfo_block_size)
        drive = torch.as_tensor(drive, device=wave.device, dtype=wave.dtype)
        wave = self.distortion(wave, drive)
        # Render duration sets the clip length; note_press_duration sets when release starts.
        hold = torch.as_tensor(note_press_duration, device=freq.device, dtype=freq.dtype)
        attack = torch.as_tensor(attack, device=freq.device, dtype=freq.dtype)
        decay = torch.as_tensor(decay, device=freq.device, dtype=freq.dtype)
        sustain = torch.as_tensor(sustain, device=freq.device, dtype=freq.dtype)
        release = torch.as_tensor(release, device=freq.device, dtype=freq.dtype)
        envelope = self.envelope(time, hold, attack, decay, sustain, release)
        return (gain * velocity * wave * envelope).unsqueeze(1)

    def from_controls(self, values, names, freq, duration, note_press_duration, **preset):
        """Decode normalized network outputs, then render with physical parameters."""
        controls = decode_controls(values, names, freq, self.sample_rate, preset)
        # Predicted controls override fixed preset values with the same name.
        return self(freq, duration, note_press_duration, **(preset | controls))
