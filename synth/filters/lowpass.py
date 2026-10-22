import torch
from torchaudio.functional import lfilter
from torchlpc import sample_wise_lpc

from ..validation import check_filter_controls


class LowPassFilter:
    """Filter [batch, samples] audio with cutoff Hz and Q shaped [batch, 1]."""

    def __init__(self, sample_rate=24000):
        self.sample_rate = sample_rate

    def coefficients(self, cutoff_hz, q):
        omega = 2 * torch.pi * cutoff_hz / self.sample_rate
        cosine = torch.cos(omega)
        alpha = torch.sin(omega) / (2 * q)
        a = torch.cat([1 + alpha, -2 * cosine, 1 - alpha], dim=-1)
        b = torch.cat([(1 - cosine) / 2, 1 - cosine, (1 - cosine) / 2], dim=-1)
        # Normalize so the first denominator coefficient is 1.
        return a / a[:, :1], b / a[:, :1]

    def __call__(self, audio, cutoff_hz, q, modulated=False, block_size=1):
        # Double precision protects low cutoffs and high resonance.
        cutoff_hz, q = torch.broadcast_tensors(cutoff_hz.double(), q.double())
        check_filter_controls(cutoff_hz, q, self.sample_rate)
        maximum_hz = min(10000., 0.45 * self.sample_rate)
        # Correct endpoint rounding only; genuinely invalid cutoffs were rejected above.
        cutoff_hz = cutoff_hz.clamp(20, maximum_hz)
        if modulated:
            return self.modulated(audio, cutoff_hz, q, block_size)
        batch = max(audio.shape[0], cutoff_hz.shape[0])
        a, b = self.coefficients(cutoff_hz.expand(batch, 1), q.expand(batch, 1))
        filtered = lfilter(audio.double().expand(batch, -1), a, b,
                           clamp=False, batching=True)
        return filtered.to(audio.dtype)

    def modulated(self, audio, cutoff_hz, q, block_size=1):
        """Cutoff is [batch, updates]; each update lasts block_size samples."""
        cutoff_hz, q = torch.broadcast_tensors(cutoff_hz.double(), q.double())
        batch = max(audio.shape[0], cutoff_hz.shape[0])
        updates = cutoff_hz.shape[1]
        x = audio.double().expand(batch, -1)
        if block_size < 1 or updates != (x.shape[-1] + block_size - 1) // block_size:
            raise ValueError('Expected one cutoff per coefficient-update interval')

        a, b = self.coefficients(cutoff_hz.expand(batch, updates).reshape(-1, 1),
                                 q.expand(batch, updates).reshape(-1, 1))
        a = a.reshape(batch, updates, 3).repeat_interleave(block_size, dim=1)[:, :x.shape[-1]]
        b = b.reshape(batch, updates, 3).repeat_interleave(block_size, dim=1)[:, :x.shape[-1]]

        # Current and previous input samples, with zero history before the clip.
        padded = torch.nn.functional.pad(x, (2, 0))
        numerator = b[..., 0] * x + b[..., 1] * padded[:, 1:-1] + b[..., 2] * padded[:, :-2]
        # One recursive pass preserves history across every coefficient update.
        filtered = sample_wise_lpc(numerator.contiguous(), a[..., 1:].contiguous())
        return filtered.to(audio.dtype)
