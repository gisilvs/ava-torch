# Ava mapper weights

Copyright (c) 2026 Gianluigi Silvestri.

The mapper weights are excluded from the repository's source-code license.
They are distributed subject to the Stability AI Community License, provided
in LICENSE_STABILITY.md. No broader permission over upstream materials is granted.

The parameter-supervised mapper was trained on SAME-S encoder outputs. The final
text-audio mapper was trained from scratch on Stable Audio 3 Small SFX latents,
using decoded audio as its reconstruction target. These are independently trained
Ava networks, not copies of the generator weights.

The upstream agreement defines derivative models broadly, including models based
on a model's outputs. This distribution conservatively treats the mapper weights
as covered by those terms; it does not claim that training on outputs removes
upstream obligations. The accompanying metadata records the training provenance.

Retain this file, NOTICE and LICENSE_STABILITY.md when redistributing the weights
or their ONNX exports. Commercial use is subject to the upstream license's
registration, revenue and other conditions. See the full agreement, including its
integrated end-user product exception, rather than relying on this summary.
