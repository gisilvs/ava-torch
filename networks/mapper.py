import torch
from torch import nn


class ParameterTransformer(nn.Module):
    """Map [batch, channels, frames] audio latents to synth controls and selectors."""

    def __init__(self, feature_shape, controls=13, width=128, heads=4, layers=3,
                 feedforward=256, dropout=0.1):
        super().__init__()
        channels, frames = feature_shape
        self.project = nn.Linear(channels, width)
        # A learned token gathers information from the whole sequence.
        self.summary = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.position = nn.Parameter(torch.randn(1, frames + 1, width) * 0.02)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(width, heads, feedforward, dropout,
                                       batch_first=True, norm_first=True)
            for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, controls)
        # Two independent four-class heads, stored in one linear layer.
        self.selector_output = nn.Linear(width, 8)

    def forward(self, features):
        # Each time frame becomes one token: [batch, frames, channels].
        tokens = self.project(features.transpose(1, 2))
        tokens = torch.cat([self.summary.expand(len(features), -1, -1), tokens], dim=1)
        # Learned positions tell the transformer where each frame occurs.
        tokens = tokens + self.position
        for layer in self.layers:
            tokens = layer(tokens)
        summary = self.norm(tokens[:, 0])
        controls = self.output(summary).sigmoid()
        # Keep raw logits for cross-entropy; apply softmax/argmax when rendering.
        selectors = self.selector_output(summary).reshape(len(features), 2, 4)
        return controls, selectors
