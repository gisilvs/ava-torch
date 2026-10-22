import math

import torch


def linear(value, minimum, maximum):
    """Map a tensor from [0, 1] to [minimum, maximum]."""
    return minimum + value * (maximum - minimum)


def logarithmic(value, minimum, maximum):
    """Map [0, 1] to positive bounds, with equal ratios for equal input steps."""
    lower = math.log(minimum)
    upper = math.log(maximum)
    return torch.exp(lower + value * (upper - lower))


def gain(value):
    """Map [0, 1] to nonnegative linear amplitude, without a fixed ceiling."""
    epsilon = torch.finfo(value.dtype).eps
    return value / (1 - value).clamp_min(epsilon)


def duration(value):
    """Map [0, 1] to positive seconds, without a fixed ceiling."""
    epsilon = torch.finfo(value.dtype).eps
    return 0.1 * value.clamp_min(epsilon) / (1 - value).clamp_min(epsilon)


def cutoff_bounds(freq, sample_rate=24000):
    """Return cutoff limits in octaves relative to the note frequency."""
    # Match the filter's precision, including at the cutoff endpoints.
    freq = freq.double()
    maximum_hz = min(10000., 0.45 * sample_rate)
    return torch.log2(20. / freq), torch.log2(maximum_hz / freq)


def depth_limit(center, freq, sample_rate=24000):
    """Limit LFO depth so both extremes stay within the cutoff bounds."""
    lower, upper = cutoff_bounds(freq, sample_rate)
    return torch.minimum(center - lower, upper - center).clamp(0, 2)


def decode_controls(values, names, freq, sample_rate=24000, preset=None):
    """Convert [batch, controls] normalized values in the given name order."""
    if values.ndim != 2 or values.shape[1] != len(names):
        raise ValueError('Expected [batch, controls] in the configured order')
    if torch.any(~torch.isfinite(values) | (values < 0) | (values > 1)):
        raise ValueError('Normalized synth controls must be finite and between 0 and 1')

    controls = dict(preset or {})
    lower, upper = cutoff_bounds(freq, sample_rate)
    for index, name in enumerate(names):
        value = values[:, index:index + 1]
        if name in ('mix', 'noise_mix', 'sustain'):
            controls[name] = value
        elif name == 'drive':
            controls[name] = linear(value, 0, 10)
        elif name == 'detune_cents':
            controls[name] = linear(value, 0, 30)
        elif name in ('q', 'lfo_rate'):
            controls[name] = logarithmic(value, 0.5, 10)
        elif name == 'gain':
            controls[name] = gain(value)
        elif name in ('attack', 'decay', 'release'):
            controls[name] = duration(value)
        elif name == 'cutoff_octaves':
            controls[name] = linear(value, lower, upper)
        elif name != 'lfo_depth':
            raise ValueError(f'Unknown synth control: {name}')

    # Decode depth last because its range depends on the decoded cutoff.
    if 'lfo_depth' in names:
        if 'cutoff_octaves' not in controls:
            raise ValueError('LFO depth needs a cutoff control or preset')
        index = list(names).index('lfo_depth')
        limit = depth_limit(controls['cutoff_octaves'], freq, sample_rate)
        controls['lfo_depth'] = values[:, index:index + 1] * limit
    return {name: controls[name] for name in names}


def encode_controls(controls, names, freq, sample_rate=24000):
    """Convert physical parameters to [batch, controls] training targets."""
    lower, upper = cutoff_bounds(freq, sample_rate)
    columns = []
    for name in names:
        value = controls[name]
        if name in ('mix', 'noise_mix', 'sustain'):
            normalized = value
        elif name == 'drive':
            normalized = value / 10
        elif name == 'detune_cents':
            normalized = value / 30
        elif name in ('q', 'lfo_rate'):
            normalized = (value.log() - math.log(0.5)) / (math.log(10) - math.log(0.5))
        elif name == 'gain':
            normalized = value / (value + 1)
        elif name in ('attack', 'decay', 'release'):
            normalized = value / (value + 0.1)
        elif name == 'cutoff_octaves':
            normalized = (value - lower) / (upper - lower)
        elif name == 'lfo_depth':
            limit = depth_limit(controls['cutoff_octaves'], freq, sample_rate)
            normalized = value / limit.clamp_min(torch.finfo(value.dtype).eps)
        else:
            raise ValueError(f'Unknown synth control: {name}')
        columns.append(normalized)

    values = torch.cat(columns, dim=1)
    # Allow small rounding errors at the boundaries, not out-of-range presets.
    if torch.any(~torch.isfinite(values) | (values < -1e-5) | (values > 1 + 1e-5)):
        raise ValueError('Preset is outside the supported synth control range')
    return values.clamp(0, 1).to(freq.dtype)
