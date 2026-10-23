# Third-party notices

Ava's original Python source is MIT-licensed. Dependencies, model weights and
upstream assets retain their own terms; the MIT license does not relicense them.
Packages are installed separately through requirements.txt. Preserve their own
license and copyright notices if redistributing their code or binaries.

| Dependency | Upstream license information |
| --- | --- |
| PyTorch and torchaudio | [PyTorch](https://github.com/pytorch/pytorch/blob/main/LICENSE), [torchaudio](https://github.com/pytorch/audio/blob/main/LICENSE) |
| TorchLPC | [MIT](https://github.com/DiffAPF/torchlpc/blob/main/LICENSE) |
| auraloss | [Apache-2.0](https://github.com/csteinmetz1/auraloss/blob/main/LICENSE) |
| torchcrepe and its pitch models | [Upstream terms](https://github.com/maxrmorrison/torchcrepe/blob/master/LICENSE) |
| Stable Audio 3 Python library | [MIT](https://github.com/Stability-AI/stable-audio-3/blob/main/LICENSE) |
| ONNX and ONNX Runtime | [ONNX](https://github.com/onnx/onnx/blob/main/LICENSE), [ONNX Runtime](https://github.com/microsoft/onnxruntime/blob/main/LICENSE) |

Other installed packages, including NumPy, SoundFile, Hydra, Transformers and
tokenizers, retain the notices supplied by their distributions. This is a guide
to direct dependencies, not a replacement for their complete license files.

## Models and generated assets

**Powered by Stability AI.** Stable Audio 3 Small SFX and SAME-S use the
[Stability AI Community License](licenses/STABILITY_AI_COMMUNITY_LICENSE.md),
not the MIT license of the Python library. T5Gemma text-encoder weights and
associated assets also retain the [Gemma Terms](licenses/GEMMA_TERMS.md).

The full agreements govern use and distribution, including commercial-use
conditions, use restrictions and attribution. They are not unrestricted
open-source licenses. Model outputs and model derivatives are distinct categories
under these agreements; do not assume that training on outputs eliminates their
requirements. See [the mapper-weight notice](weights/LICENSE.md).

Generator exports include copies of these agreements and a NOTICE describing
Ava's ONNX conversion and optional INT8 quantization. Keep them with the assets.
Exporting or quantizing weights does not remove the upstream terms. Obtain
upstream models through their official Hugging Face repositories.
