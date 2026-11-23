"""Cheap MT3 port equivalence audit; source checkouts are supplied explicitly.

No model inference, dependency installation, or downloads. Extracts selected
upstream definitions without importing either project's dependency stack.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import numpy as np
import torch


def definition(source, name):
    matches = [n for n in ast.walk(ast.parse(source))
               if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name]
    if len(matches) != 1:
        raise ValueError(f'Expected one definition: {name}')
    node = matches[0]
    node.decorator_list = []
    return 'from __future__ import annotations\n' + ast.unparse(node)


def audit(port, official):
    # Run the actual port class and the official sinusoidal initializer. The
    # official function uses NumPy until its final jnp.array conversion; aliasing
    # that conversion to NumPy avoids needing the JAX runtime for this comparison.
    namespace = {'np': np, 'jnp': np, 'torch': torch, 'nn': torch.nn,
                 'rearrange': lambda x, pattern: x[None]}
    exec(definition((port / 'models/t5.py').read_text(), 'FixedPositionalEmbedding'), namespace)
    exec(definition((official / 'mt3/layers.py').read_text(), 'sinusoidal'), namespace)
    checkpoint = port / 'pretrained/mt3.pth'
    weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
    positions = namespace['FixedPositionalEmbedding'](512)
    positions.load_state_dict({'inv_freq': weights['encoder.pos_emb.inv_freq']})
    actual = positions(256).detach().numpy()[0]
    expected = namespace['sinusoidal']()(None, (256, 512))

    notebook = json.loads((official / 'mt3/colab/music_transcription_with_transformers.ipynb').read_text())
    official_code = '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code' and 'class InferenceModel' in ''.join(c['source']))
    namespace.update(
        vocabularies=SimpleNamespace(DECODED_EOS_ID=-1),
        note_sequences=SimpleNamespace(NoteEncodingWithTiesSpec=None),
        metrics_utils=SimpleNamespace(event_predictions_to_ns=lambda predictions, **kw: {'est_ns': predictions}),
    )
    exec(definition(official_code, '_trim_eos'), namespace)
    exec(definition((port / 'inference.py').read_text(), '_to_event'), namespace)
    decoder = SimpleNamespace(codec=SimpleNamespace(steps_per_second=100))
    cases = []
    for tokens in ([10, 20, 30], [10, 20, -1]):
        result = namespace['_to_event'](decoder, [np.array([tokens])], [[[0.]]])
        cases.append(dict(input=tokens, port=result[0]['est_tokens'].tolist(),
                          official=namespace['_trim_eos'](tokens).tolist()))
    return dict(
        port_revision=subprocess.check_output(['git', '-C', str(port), 'rev-parse', 'HEAD'], text=True).strip(),
        official_revision=subprocess.check_output(['git', '-C', str(official), 'rev-parse', 'HEAD'], text=True).strip(),
        port_weights_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        positional_encoding_max_abs_error=float(np.max(np.abs(actual - expected))),
        positional_encoding_equal=bool(np.allclose(actual, expected, atol=1e-6, rtol=1e-6)),
        checkpoint_frequency_matches_port_formula=bool(torch.equal(
            weights['encoder.pos_emb.inv_freq'], namespace['FixedPositionalEmbedding'](512).inv_freq)),
        eos_cases=cases,
        full_model_inference_run=False,
        original_weight_tensor_parity_verified=False,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=Path, required=True)
    parser.add_argument('--official', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.port, args.official), indent=2))
