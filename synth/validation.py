import math

import torch

from .controls import depth_limit


def check_note(freq, duration, note_press_duration, sample_rate):
    """Check physical note values; freq is a floating tensor shaped [batch, 1]."""
    if torch.any(~torch.isfinite(freq) | (freq <= 0) | (freq >= sample_rate / 2)):
        raise ValueError('Frequency must be positive and below Nyquist')
    if not math.isfinite(duration) or duration <= 0 or round(duration * sample_rate) < 1:
        raise ValueError('Render duration must be finite and contain at least one sample')

    hold = torch.as_tensor(note_press_duration)
    if torch.any(~torch.isfinite(hold) | (hold < 0)):
        raise ValueError('Note hold must be finite and nonnegative')


def check_filter_controls(cutoff_hz, q, sample_rate):
    """Check physical cutoff and resonance, allowing tiny cutoff rounding errors."""
    maximum_hz = min(10000., 0.45 * sample_rate)
    if torch.any(~torch.isfinite(cutoff_hz) | (cutoff_hz < 20 - 1e-6)
                 | (cutoff_hz > maximum_hz + 1e-6)):
        raise ValueError('Cutoff must be between 20 Hz and the filter maximum')
    if torch.any(~torch.isfinite(q) | (q < 0.5) | (q > 10)):
        raise ValueError('Q must be between 0.5 and 10')


def check_lfo_controls(rate, depth, center, freq, sample_rate):
    """Check the full possible cutoff sweep, even if a clip never reaches its peaks."""
    if torch.any(~torch.isfinite(rate) | (rate < 0.5) | (rate > 10)):
        raise ValueError('LFO rate must be between 0.5 and 10 Hz')
    limit = depth_limit(center, freq, sample_rate)
    if torch.any(~torch.isfinite(depth) | (depth < 0) | (depth > limit + 1e-6)):
        raise ValueError('LFO depth exceeds the legal cutoff sweep')
