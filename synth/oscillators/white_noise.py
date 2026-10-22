import torch


class WhiteNoise:
    """Return noise shaped [1, samples], shared across batch elements."""

    def __init__(self, seed=0):
        self.seed = seed

    def __call__(self, time):
        # Restart a local generator so fitting uses the same noise on each call.
        generator = torch.Generator(device='cpu').manual_seed(self.seed)
        samples = torch.rand(1, time.shape[-1], generator=generator,
                             device='cpu', dtype=torch.float32)
        return (2 * samples - 1).to(time)
