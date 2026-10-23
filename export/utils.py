"""Small file helpers for the asset format read by the C++ plugin."""
import hashlib
import json
import shutil
from pathlib import Path


def checksum(path):
    # Ava verifies these hashes when loading model assets.
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def write_manifest(root, precision):
    copy_model_notices(root, generator=True)
    # Ava requires this exact six-file list and checks each hash on load.
    names = ('text_encoder.onnx', 'text_encoder.weights', 'denoiser.onnx',
             'denoiser.weights', 'tokenizer.bin', 'sampler.json')
    files = {name: checksum(root / name) for name in names}
    save_json(root / 'manifest.json', {
        'format': 'ava-native-generator', 'version': 1,
        'model': 'stable-audio-3-small-sfx', 'precision': precision, 'files': files,
    })
    # The existing Ava installer reads this list before copying assets.
    (root / 'checksums.txt').write_text(''.join(
        f'{digest}  {name}\n' for name, digest in files.items()))


def copy_model_notices(root, generator=False):
    """Keep model terms with exported weights, separately from the code license."""
    repo = Path(__file__).resolve().parents[1]
    shutil.copyfile(repo / 'licenses/STABILITY_AI_COMMUNITY_LICENSE.md',
                    root / 'LICENSE_STABILITY.md')
    if generator:
        shutil.copyfile(repo / 'licenses/GEMMA_TERMS.md', root / 'LICENSE_GEMMA.md')
        shutil.copyfile(repo / 'licenses/GENERATOR_NOTICE.txt', root / 'NOTICE')
    else:
        shutil.copyfile(repo / 'weights/LICENSE.md', root / 'MODEL_LICENSE.md')
        shutil.copyfile(repo / 'weights/NOTICE', root / 'NOTICE')
