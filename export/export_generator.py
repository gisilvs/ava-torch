"""Export the text encoder and denoiser separately to keep memory use manageable."""
import json
import shutil
import struct
from pathlib import Path

import hydra
import numpy as np
import onnx
import onnxruntime as ort
import torch
from safetensors import safe_open
from stable_audio_3.factory import create_multi_conditioner_from_conditioning_config
from stable_audio_3.inference.distribution_shift import DistributionShift
from stable_audio_3.inference.sampling import build_schedule
from stable_audio_3.model_configs import models
from stable_audio_3.models.diffusion import DiTWrapper
from torch import nn
from torch.onnx.symbolic_helper import parse_args

from export.utils import checksum, copy_model_notices, save_json, write_manifest
from export.quantize_generator import quantize_bundle
from preparation.text_prompts import build_prompt


def write_tokenizer(root, vocabulary, merges, added, pad):
    # Length-prefixed UTF-8 strings and fixed-width integer merge rules. Avoid
    # parsing hundreds of thousands of JSON arrays whenever Generate is pressed.
    assert all(not any(token[k] for k in ['single_word','lstrip','rstrip','normalized']) for token in added)
    with (root / 'tokenizer.bin').open('wb') as file:
        file.write(b'AVATOK1\0')
        def integer(value):
            file.write(struct.pack('<I',value))
        def token(text, index):
            data = text.encode('utf-8')
            integer(index)
            integer(len(data))
            file.write(data)
        integer(pad)
        integer(len(vocabulary))
        for text,index in vocabulary.items():
            token(text,index)
        integer(len(merges))
        for rule in merges:
            file.write(struct.pack('<III',*rule))
        integer(len(added))
        for value in added:
            token(value['content'],value['id'])


def external_export(model, inputs, names, outputs, path):
    @parse_args('v', 'v', 'v', 'f')
    def rms_norm(graph, value, shape, weight, epsilon):
        mean = graph.op('ReduceMean', graph.op('Mul', value, value),
                        axes_i=[-1], keepdims_i=1)
        eps = graph.op('Constant', value_t=torch.tensor(epsilon, dtype=torch.float32))
        normalized = graph.op('Div', value, graph.op('Sqrt', graph.op('Add', mean, eps)))
        return graph.op('Mul', normalized, weight)
    # Standard ONNX arithmetic, not a custom runtime operator.
    torch.onnx.register_custom_op_symbolic('aten::rms_norm', rms_norm, 17)
    torch.onnx.export(model, inputs, str(path), export_params=False,
                      input_names=names, output_names=outputs, opset_version=17,
                      do_constant_folding=False, dynamo=False)
    graph = onnx.load(str(path), load_external_data=False)
    state = model.state_dict()
    parameters = [value for value in graph.graph.input if value.name not in names]
    data = path.with_suffix('.weights')
    with data.open('wb') as stream:
        for value in parameters:
            tensor = state[value.name].detach().cpu().contiguous()
            offset = stream.tell()
            tensor.numpy().tofile(stream)
            initializer = onnx.TensorProto(name=value.name, dims=list(tensor.shape),
                data_type=onnx.helper.np_dtype_to_tensor_dtype(tensor.numpy().dtype),
                data_location=onnx.TensorProto.EXTERNAL)
            for key, item in [('location', data.name), ('offset', str(offset)),
                              ('length', str(stream.tell() - offset))]:
                initializer.external_data.add(key=key, value=item)
            graph.graph.initializer.append(initializer)
            graph.graph.input.remove(value)
    onnx.save_model(graph, str(path))
    onnx.checker.check_model(str(path))
    print('Exported', path, 'weights MiB', data.stat().st_size / 1024**2, flush=True)


class TextEncoder(nn.Module):
    def __init__(self, conditioner):
        super().__init__()
        prompt = conditioner.conditioners['prompt']
        self.encoder = prompt.model
        self.projection = prompt.proj_out
        self.padding = prompt.padding_embedding
        duration = conditioner.conditioners['seconds_total']([6.], 'cpu')[0]
        self.register_buffer('duration', duration)

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                              return_dict=False)[0]
        hidden = self.projection(hidden)
        hidden = torch.where(attention_mask.unsqueeze(-1).bool(), hidden, self.padding[None, None])
        return torch.cat([hidden, self.duration], 1), self.duration[:, 0]


@torch.inference_mode()
def text_stage(root, config, weights):
    copy_model_notices(root, generator=True)
    conditioner = create_multi_conditioner_from_conditioning_config(config['model']['conditioning'])
    with safe_open(str(weights), framework='pt') as saved:
        state = {key.removeprefix('conditioner.'): saved.get_tensor(key)
                 for key in saved.keys() if key.startswith('conditioner.')}
    conditioner.load_state_dict(state, strict=True)
    del state
    conditioner.eval().requires_grad_(False)
    prompt = conditioner.conditioners['prompt']
    prompt.model.set_attn_implementation('eager')
    tokenizer = prompt.tokenizer
    # Export FP32 arithmetic; the original text encoder weights are BF16.
    model = TextEncoder(conditioner).float().eval()
    example = build_prompt('Trumpet', 'dark and mellow', 60)
    tokens = tokenizer(example, max_length=256, padding='max_length',
                       truncation=True, return_tensors='pt')
    inputs = (tokens['input_ids'], tokens['attention_mask'])
    context, global_cond = model(*inputs)
    # One reference case supports a basic export check, without research reports.
    np.savez(root / 'text_reference.npz', input_ids=inputs[0].numpy(),
             attention_mask=inputs[1].numpy(), cross_attention=context.numpy(),
             global_cond=global_cond.numpy())
    external_export(model, inputs, ['input_ids', 'attention_mask'],
                    ['cross_attention', 'global_cond'], root / 'text_encoder.onnx')

    source = weights.parent / 't5gemma-b-b-ul2/tokenizer.json'
    shutil.copyfile(source, root / 'tokenizer_hf.json')
    data = json.loads(source.read_text())
    assert data['model']['type'] == 'BPE' and data['model']['byte_fallback']
    vocabulary = data['model']['vocab']
    merges = [[vocabulary[a], vocabulary[b], vocabulary[a + b]]
              for a, b in data['model']['merges']]
    write_tokenizer(root, vocabulary, merges, data['added_tokens'], tokenizer.pad_token_id)


class Denoiser(nn.Module):
    def __init__(self, dit):
        super().__init__()
        self.dit = dit
        self.register_buffer('local', torch.zeros(1, 257, 66))
        # Six seconds at44.1kHz is65 effective latent frames; storage has66.
        self.register_buffer('valid', torch.arange(66)[None] < 65)

    def forward(self, latent, timestep, cross_attention, global_cond):
        return self.dit(latent, timestep, cross_attn_cond=cross_attention,
                        global_cond=global_cond, local_add_cond=self.local,
                        padding_mask=self.valid, cfg_scale=1., rescale_cfg=True)


@torch.inference_mode()
def denoiser_stage(root, config, weights):
    options = config['model']['diffusion']
    # Build an empty model first, then attach weights without a second full copy.
    with torch.device('meta'):
        dit = DiTWrapper(diffusion_objective=options['diffusion_objective'], **options['config'])
    with safe_open(str(weights), framework='pt') as saved:
        state = {key.removeprefix('model.'): saved.get_tensor(key)
                 for key in saved.keys() if key.startswith('model.')}
    dit.load_state_dict(state, strict=True, assign=True)
    del state
    if any(value.is_meta for value in dit.buffers()):
        raise ValueError('Denoiser has an uninitialized meta buffer')
    model = Denoiser(dit.eval().requires_grad_(False)).eval()
    text = np.load(root / 'text_reference.npz')
    context = torch.from_numpy(text['cross_attention'])
    global_cond = torch.from_numpy(text['global_cond'])
    schedule = build_schedule(8, dist_shift=DistributionShift(min_length=256, max_length=4096),
                              effective_seq_len=torch.tensor([65]), device='cpu').flatten()
    torch.manual_seed(123)
    latent = torch.randn(1, 256, 66)
    inputs = (latent, schedule[:1], context, global_cond)
    velocity = model(*inputs)
    np.savez(root / 'denoiser_reference.npz', latent=latent.numpy(),
             timestep=schedule[:1].numpy(), cross_attention=context.numpy(),
             global_cond=global_cond.numpy(), velocity=velocity.numpy())
    external_export(model, inputs, ['latent', 'timestep', 'cross_attention', 'global_cond'],
                    ['velocity'], root / 'denoiser.onnx')
    # These settings and source hashes are read by Ava's existing model loader.
    save_json(root / 'sampler.json', {
        'duration': 6.0, 'sample_rate': 44100, 'channels': 256, 'frames': 66,
        'effective_frames': 65, 'text_length': 256, 'steps': 8, 'cfg_scale': 1.0,
        'schedule': schedule.tolist(), 'sampler': 'rf_denoiser_pingpong',
        'checkpoint_sha256': checksum(weights),
        'config_sha256': checksum(weights.parent / 'model_config.json'),
    })
    write_manifest(root, 'FP32')


def verify_stage(root, stage):
    name = 'text_encoder' if stage == 'text' else 'denoiser'
    reference = 'text_reference.npz' if stage == 'text' else 'denoiser_reference.npz'
    session = ort.InferenceSession(str(root / f'{name}.onnx'), providers=['CPUExecutionProvider'])
    values = np.load(root / reference)
    inputs = {node.name: values[node.name] for node in session.get_inputs()}
    outputs = session.run(None, inputs)
    for node, actual in zip(session.get_outputs(), outputs):
        np.testing.assert_allclose(actual, values[node.name], rtol=2e-3, atol=2e-4)
    print(f'{name}: ONNX output matches the saved PyTorch reference')


@hydra.main(version_base=None, config_path='../configs', config_name='generator_export')
def main(cfg):
    root = Path(cfg.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    if cfg.stage == 'quantize':
        quantize_bundle(root, Path(cfg.quantized_dir))
    elif cfg.stage in ('verify-text', 'verify-denoiser'):
        verify_stage(root, cfg.stage.removeprefix('verify-'))
    elif cfg.stage in ('text', 'denoiser'):
        config_path, weights = models['small-sfx'].resolve()
        config = json.loads(Path(config_path).read_text())
        stage = text_stage if cfg.stage == 'text' else denoiser_stage
        stage(root, config, Path(weights))
    else:
        raise ValueError('Choose text, denoiser, verify-text, verify-denoiser or quantize')


if __name__ == '__main__':
    main()
