import torch


class SoftDistortion:
    """Shape [batch, samples] audio with drive tensors shaped [batch, 1]."""

    def __call__(self, audio, drive):
        # tanh(drive * audio) / tanh(drive), with its Taylor limit near zero.
        # Both branches must stay finite, including the unselected one.
        safe_drive = drive.clamp_min(1e-4)
        shaped = torch.tanh(safe_drive * audio) / torch.tanh(safe_drive)
        near_zero = audio + drive.square() * (audio - audio.pow(3)) / 3
        return torch.where(drive < 1e-4, near_zero, shaped)
