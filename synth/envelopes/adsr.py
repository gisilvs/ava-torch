import math

import torch


class ADSREnvelope:
    """Time is relative to note-on; hold and ADSR tensors have shape [batch, 1]."""

    # Fixed shape inherited from the original synth, not a learned parameter.
    # With curvature 4, the transition reaches about 88% at halfway progress.
    curvature = 4.0

    def curve(self, progress):
        progress = progress.clamp(0, 1)
        curve = -torch.expm1(-self.curvature * progress) / -math.expm1(-self.curvature)
        return torch.where(progress >= 1, torch.ones_like(curve), curve)

    def held_level(self, time, attack, decay, sustain):
        attack_level = self.curve(time / attack)
        decay_level = 1 - (1 - sustain) * self.curve((time - attack) / decay)
        return torch.where(time < attack, attack_level, decay_level)

    def __call__(self, time, note_press_duration, attack, decay, sustain, release):
        # The caller supplies tensors on the same device and with the same dtype as time.
        # Durations are positive seconds; sustain is an amplitude in [0, 1].
        held = self.held_level(time, attack, decay, sustain)
        # Release starts from the actual level, including during attack or decay.
        level_at_release = self.held_level(note_press_duration, attack, decay, sustain)
        tail = level_at_release * (1 - self.curve((time - note_press_duration) / release))
        return torch.where(time < note_press_duration, held, tail)
