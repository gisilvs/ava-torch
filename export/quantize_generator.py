"""Weight-only ONNX quantization, one matrix at a time to limit export RAM.

Uses ONNX Runtime's quantizer, not custom quantization math. Keep the FP32
installation intact until the quantized model has been timed and auditioned.
"""
import shutil
from pathlib import Path

import onnx
from onnx import helper, numpy_helper
from onnxruntime.quantization.matmul_nbits_quantizer import (
    DefaultWeightOnlyQuantConfig, DefaultWeightOnlyQuantizer,
)

from export.utils import write_manifest


def quantize(source, destination, name, bits):
    model = onnx.load(source / f'{name}.onnx', load_external_data=False)
    weights = {tensor.name: tensor for tensor in model.graph.initializer}
    producers = {output: node for node in model.graph.node for output in node.output}
    quantizer = DefaultWeightOnlyQuantizer(DefaultWeightOnlyQuantConfig(
        bits=bits, block_size=128, is_symmetric=True, accuracy_level=4))
    nodes, packed, count = [], [], 0
    data_name = f'{name}.weights'

    def write_tensor(stream, tensor):
        # Write immediately, then retain only the small external-data descriptor.
        offset = stream.tell()
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            info = {entry.key: entry.value for entry in tensor.external_data}
            with (source / info['location']).open('rb') as original:
                original.seek(int(info.get('offset', 0)))
                remaining = int(info['length'])
                while remaining:
                    chunk = original.read(min(remaining, 1024 * 1024))
                    if not chunk:
                        raise ValueError('Truncated source weights')
                    stream.write(chunk)
                    remaining -= len(chunk)
        else:
            stream.write(numpy_helper.to_array(tensor).tobytes())
        result = onnx.TensorProto(name=tensor.name, dims=tensor.dims,
                                 data_type=tensor.data_type,
                                 data_location=onnx.TensorProto.EXTERNAL)
        for key, value in [('location', data_name), ('offset', offset),
                           ('length', stream.tell() - offset)]:
            result.external_data.add(key=key, value=str(value))
        return result

    with (destination / data_name).open('wb') as stream:
        for node in model.graph.node:
            transpose = producers.get(node.input[1]) if node.op_type == 'MatMul' else None
            tensor = weights.get(transpose.input[0]) if transpose is not None and transpose.op_type == 'Transpose' else None
            if tensor is None or len(tensor.dims) != 2:
                nodes.append(node)
                continue
            permutation = next((list(a.ints) for a in transpose.attribute if a.name == 'perm'), [1, 0])
            if permutation != [1, 0]:
                nodes.append(node)
                continue
            # The exporter leaves constant weight transposes as graph nodes.
            # Fold just these transposes before asking ORT to quantize them.
            array = numpy_helper.to_array(tensor, base_dir=str(source)).T.copy()
            weight = numpy_helper.from_array(array, name=f'ava_quant_weight_{count}')
            matmul = helper.make_node('MatMul', [node.input[0], weight.name],
                                      list(node.output), name=node.name)
            graph = helper.make_graph([matmul], 'matrix', [], [], [weight])
            nodes.extend(quantizer.quantize_matmul(matmul, [graph]))
            packed.extend(write_tensor(stream, item) for item in graph.initializer
                          if item.name != weight.name)
            count += 1
            del array, weight, graph

        # Discard dead transposes and their original FP32 matrices.
        needed = {output.name for output in model.graph.output}
        live = []
        for node in reversed(nodes):
            if needed.intersection(node.output):
                live.append(node)
                needed.update(node.input)
        initializers = [write_tensor(stream, tensor) for tensor in model.graph.initializer
                        if tensor.name in needed]
        initializers.extend(tensor for tensor in packed if tensor.name in needed)
    del model.graph.node[:]
    model.graph.node.extend(reversed(live))
    del model.graph.initializer[:]
    model.graph.initializer.extend(initializers)
    if not any(op.domain == 'com.microsoft' for op in model.opset_import):
        model.opset_import.append(helper.make_opsetid('com.microsoft', 1))
    onnx.save_model(model, destination / f'{name}.onnx')
    onnx.checker.check_model(str(destination / f'{name}.onnx'))
    print(f'{name}: {count} matrices quantized to {bits} bits; '
          f'{(destination / data_name).stat().st_size / 1024**2:.1f} MiB', flush=True)
    if count == 0:
        raise ValueError('No matrices were quantized')


def quantize_bundle(source, destination):
    """Produce the plugin's INT8 bundle without replacing the FP32 source."""
    source, destination = source.resolve(), destination.resolve()
    if source == destination:
        raise ValueError('Use a separate directory; preserve the FP32 reference')
    destination.mkdir(parents=True, exist_ok=True)
    for name in ('text_encoder', 'denoiser'):
        quantize(source, destination, name, bits=8)
    for name in ('tokenizer.bin', 'sampler.json'):
        shutil.copyfile(source / name, destination / name)
    # Python's tokenizer JSON is extra; Ava uses the binary copied above.
    shutil.copyfile(source / 'tokenizer_hf.json', destination / 'tokenizer_hf.json')
    write_manifest(destination, 'INT8 linear weights; FP32 remaining operations')
