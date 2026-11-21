"""Official MT3 inference without installing the T5X training stack.

Uses pinned upstream network, spectrogram, event decoding and T5X beam search.
Only dataset/training imports are omitted from the generated minimal packages.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
ASSETS = ROOT / 'benchmarks/datasets/pretrained/mt3-port'
MT3_REV = '462db32a567e6bdcefcee7acb1ec36b011f89e9e'
T5X_REV = '2045b332cf19887885a74ef1bd6b2adb2a7ca634'
CHECKPOINT_URL = 'https://storage.googleapis.com/mt3/checkpoints/mt3/'


class MT3Direct:
    @staticmethod
    def setup():
        """A private overlay reads host packages but never modifies the host venv."""
        import venv
        host = ROOT / 'at-venv/bin/python'
        if not host.is_file():
            raise FileNotFoundError('Create the project at-venv first')
        python = ASSETS / 'venv/bin/python'
        if not python.exists():
            subprocess.run([str(host), '-m', 'venv', str(ASSETS / 'venv')], check=True)
        host_site = subprocess.check_output([str(host), '-c',
            'import sysconfig; print(sysconfig.get_paths()["purelib"])'], text=True).strip()
        site = next((ASSETS / 'venv/lib').glob('python*/site-packages'))
        (site / 'attune-host.pth').write_text(host_site + '\n')
        requirements = ['jax==0.4.35', 'jaxlib==0.4.35', 'flax==0.10.2',
                        'optax==0.2.5', 'orbax-checkpoint==0.11.5', 'chex==0.1.90',
                        'tensorstore==0.1.69', 'note-seq==0.0.5', 'gin-config==0.5.0']
        check = 'from importlib.metadata import version; import json; print(json.dumps({n: version(n) for n in ' + repr([r.split('==')[0] for r in requirements]) + '}))'
        result = subprocess.run([str(python), '-c', check], text=True, capture_output=True)
        installed = json.loads(result.stdout) if result.returncode == 0 else {}
        if any(installed.get(r.split('==')[0]) != r.split('==')[1] for r in requirements):
            subprocess.run([str(python), '-m', 'pip', 'install', *requirements], check=True)
        sources = ASSETS / 'sources'
        sources.mkdir(parents=True, exist_ok=True)
        for name, url, revision in (
            ('mt3', 'https://github.com/magenta/mt3.git', MT3_REV),
            ('t5x', 'https://github.com/google-research/t5x.git', T5X_REV)):
            path = sources / name
            if not path.exists():
                subprocess.run(['git', 'init', str(path)], check=True)
                subprocess.run(['git', '-C', str(path), 'remote', 'add', 'origin', url], check=True)
                subprocess.run(['git', '-C', str(path), 'fetch', '--depth', '1', 'origin', revision], check=True)
                subprocess.run(['git', '-C', str(path), 'checkout', '--detach', revision], check=True)
        MT3Direct.prepare(sources / 'mt3', sources / 't5x')
        return {'python': str(python), 'runtime': str(ASSETS / 'runtime')}

    @staticmethod
    def prepare(mt3_source, t5x_source):
        """Build inference-only packages; retain upstream function bodies."""
        for source, revision in ((mt3_source, MT3_REV), (t5x_source, T5X_REV)):
            actual = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
            if actual != revision:
                raise ValueError(f'Expected {revision}, got {actual}')
        dest = ASSETS / 'runtime'
        for package in ('mt3', 't5x'):
            (dest / package).mkdir(parents=True, exist_ok=True)
            (dest / package / '__init__.py').write_text('"""Minimal inference package; see MT3Direct.py."""\n')
        for name in ('layers', 'network', 'spectral_ops', 'spectrograms', 'event_codec', 'note_sequences', 'metrics_utils'):
            shutil.copyfile(mt3_source / f'mt3/{name}.py', dest / f'mt3/{name}.py')
        source = (mt3_source / 'mt3/vocabularies.py').read_text()
        # The vocabulary facade depends on SeqIO/T5. Note-event helpers do not.
        source = source.split('\ndef vocabulary_from_codec(')[0]
        source = source.replace('import seqio\n', '').replace('import t5.data\n', '')
        (dest / 'mt3/vocabularies.py').write_text(source)
        source = (mt3_source / 'mt3/run_length_encoding.py').read_text()
        tree = ast.parse(source)
        keep = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if node.name not in ('EventEncodingSpec', 'decode_events'):
                    continue
            if isinstance(node, ast.Import) and any(a.name == 'seqio' for a in node.names):
                continue
            keep.append(node)
        # Preserve exact source segments (including decorators), not rewritten bodies.
        pieces = []
        lines = source.splitlines(keepends=True)
        for node in keep:
            start = min([node.lineno] + [d.lineno for d in getattr(node, 'decorator_list', [])])
            pieces.append(''.join(lines[start-1:node.end_lineno]))
        (dest / 'mt3/run_length_encoding.py').write_text('\n'.join(pieces))
        for name in ('decoding', 'binary_search'):
            shutil.copyfile(t5x_source / f't5x/{name}.py', dest / f't5x/{name}.py')
        manifest = {str(p.relative_to(dest)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in dest.rglob('*.py')}
        (ASSETS / 'sources.json').write_text(json.dumps(dict(mt3=MT3_REV, t5x=T5X_REV, files=manifest), indent=2))
        return manifest

    @staticmethod
    def download_weights():
        """Fetch only inference tensors, never optimizer state or a training runtime."""
        import concurrent.futures
        import gzip
        import urllib.request
        import msgpack
        import numpy as np

        folder = ASSETS / 'checkpoint'
        folder.mkdir(parents=True, exist_ok=True)
        def fetch(relative):
            path = folder / relative
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                with urllib.request.urlopen(CHECKPOINT_URL + relative, timeout=60) as response:
                    data = response.read()
                tmp = path.with_name(path.name + '.tmp')
                tmp.write_bytes(data)
                tmp.replace(path)
            return path.read_bytes()
        def ext(code, data):
            if code != 1:
                raise ValueError(f'Unsupported checkpoint extension {code}')
            shape, dtype, raw = msgpack.unpackb(data, raw=False)
            return np.frombuffer(raw, dtype=np.dtype(dtype)).reshape(shape)
        raw = fetch('checkpoint')
        target = msgpack.unpackb(raw, raw=False, ext_hook=ext)['optimizer']['target']
        leaves = []
        def walk(value, prefix=''):
            if isinstance(value, dict) and 'driver' not in value:
                for key, item in value.items():
                    walk(item, prefix + '/' + key if prefix else key)
            else:
                leaves.append((prefix, value))
        walk(target)
        def load(item):
            key, value = item
            if isinstance(value, np.ndarray):
                return key, value
            relative = value['kvstore']['path']
            metadata = json.loads(fetch(relative + '.zarray'))
            if metadata['chunks'] != metadata['shape'] or metadata['compressor']['id'] != 'gzip' or metadata['order'] != 'C':
                raise ValueError(f'Unsupported checkpoint tensor layout: {relative}')
            chunk = fetch(relative + '.'.join('0' for _ in metadata['shape']))
            array = np.frombuffer(gzip.decompress(chunk), dtype=metadata['dtype']).reshape(metadata['shape'])
            return key, array
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            arrays = dict(pool.map(load, leaves))
        np.savez(folder / 'params.npz', **arrays)
        provenance = dict(url=CHECKPOINT_URL, manifest_sha256=hashlib.sha256(raw).hexdigest(),
                          tensors={k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in arrays.items()})
        (folder / 'provenance.json').write_text(json.dumps(provenance, indent=2))
        return dict(tensors=len(arrays), bytes=sum(a.nbytes for a in arrays.values()))

    def __init__(self):
        os.environ.setdefault('JAX_PLATFORMS', 'cpu')
        os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
        os.environ.setdefault('OMP_NUM_THREADS', '1')
        os.environ.setdefault('XLA_FLAGS', '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')
        os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '1')
        os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
        os.environ.setdefault('MPLCONFIGDIR', str(ASSETS / 'matplotlib'))
        sys.path.insert(0, str(ASSETS / 'runtime'))
        import jax
        import numpy as np
        from flax.traverse_util import unflatten_dict
        from mt3 import network
        manifest = json.loads((ASSETS / 'sources.json').read_text())
        for relative, digest in manifest['files'].items():
            if hashlib.sha256((ASSETS / 'runtime' / relative).read_bytes()).hexdigest() != digest:
                raise ValueError(f'MT3 runtime source changed: {relative}; rerun setup')
        expected = json.loads((ASSETS / 'checkpoint/provenance.json').read_text())['tensors']
        with np.load(ASSETS / 'checkpoint/params.npz') as archive:
            arrays = dict(archive)
        if set(arrays) != set(expected) or any(hashlib.sha256(v.tobytes()).hexdigest() != expected[k] for k, v in arrays.items()):
            raise ValueError('MT3 checkpoint tensors changed; restore the official download')
        self.params = unflatten_dict({tuple(k.split('/')): v for k, v in arrays.items()})
        self.model = network.Transformer(network.T5Config(
            vocab_size=1536, emb_dim=512, num_heads=6, num_encoder_layers=8,
            num_decoder_layers=8, head_dim=64, mlp_dim=1024,
            mlp_activations=('gelu', 'linear'), dropout_rate=.1, logits_via_embedding=False))
        self._decoders = {}
        self.encode = jax.jit(lambda params, x: self.model.apply({'params': params}, x, enable_dropout=False, method=self.model.encode))

    def tokens(self, inputs, max_length=1024):
        """Use official T5X beam search (one decode, default alpha), not HF greedy."""
        import jax
        import jax.numpy as jnp
        import numpy as np
        from t5x import decoding
        x = jnp.asarray(inputs[None], dtype=jnp.float32)
        encoded = self.encode(self.params, x)
        dummy = jnp.zeros((1, max_length), dtype=jnp.int32)
        if max_length not in self._decoders:
            def predict(params, enc, audio):
                _, variables = self.model.apply({'params': params}, enc, audio, dummy, dummy,
                    enable_dropout=False, decode=True, max_decode_length=max_length,
                    method=self.model.decode, mutable=['cache'])
                def step(state):
                    logits, changed = self.model.apply({'params': params, 'cache': state.cache}, enc, audio,
                        state.cur_token, state.cur_token, enable_dropout=False, decode=True,
                        max_decode_length=max_length, method=self.model.decode, mutable=['cache'])
                    return logits[:, 0, :], changed['cache']
                return decoding.beam_search(dummy, variables['cache'], step, eos_id=1, num_decodes=1)
            self._decoders[max_length] = jax.jit(predict)
        result, scores = self._decoders[max_length](self.params, encoded, x)
        return np.asarray(result[0, -1]), float(scores[0, -1])

    @staticmethod
    def frames(audio):
        import numpy as np
        from mt3 import spectrograms
        config = spectrograms.SpectrogramConfig()
        audio = np.pad(np.asarray(audio, np.float32), (0, 128 - len(audio) % 128))
        frames = np.asarray(spectrograms.split_audio(audio, config))
        for i in range(0, len(frames), 256):
            samples = frames[i:i+256].reshape(-1)
            spec = np.asarray(spectrograms.compute_spectrogram(samples, config))
            # Official preprocessing computes the short final spectrogram first;
            # SeqIO then pads feature frames to 256 with zeros.
            yield i / 125., np.pad(spec, ((0, 256-len(spec)), (0, 0)))

    @staticmethod
    def notes(predictions):
        import numpy as np
        from mt3 import vocabularies, note_sequences, metrics_utils
        codec = vocabularies.build_codec(vocabularies.VocabularyConfig(num_velocity_bins=1))
        events = []
        for start, encoded in predictions:
            encoded = np.asarray(encoded)
            eos = np.flatnonzero(encoded == 1)
            if len(eos):
                encoded = encoded[:eos[0]]
            tokens = np.where((encoded >= 3) & (encoded < codec.num_classes + 3), encoded - 3, -2)
            start -= start % (1 / codec.steps_per_second)
            events.append(dict(est_tokens=tokens, start_time=start, raw_inputs=[]))
        return metrics_utils.event_predictions_to_ns(events, codec=codec,
            encoding_spec=note_sequences.NoteEncodingWithTiesSpec)['est_ns']

    def transcribe(self, path):
        import librosa
        audio, _ = librosa.load(path, sr=16000, mono=True)
        predictions = [(start, self.tokens(spec)[0]) for start, spec in self.frames(audio)]
        sequence = self.notes(predictions)
        return dict(intervals=[[n.start_time, n.end_time] for n in sequence.notes],
                    frequencies_hz=[440. * 2 ** ((n.pitch-69)/12) for n in sequence.notes])

    def smoke(self):
        """Bounded network/cache and event checks, plus one short synthetic clip."""
        import jax.numpy as jnp
        import numpy as np
        from mt3 import event_codec, vocabularies
        x = np.random.default_rng(0).normal(size=(1, 8, 512)).astype(np.float32)
        ids = jnp.array([[0, 100, 200]], dtype=jnp.int32)
        encoded = self.encode(self.params, x)
        full = self.model.apply({'params': self.params}, encoded, x, ids, jnp.ones_like(ids),
            enable_dropout=False, method=self.model.decode)
        dummy = jnp.zeros((1, 4), dtype=jnp.int32)
        _, state = self.model.apply({'params': self.params}, encoded, x, dummy, dummy,
            decode=True, enable_dropout=False, max_decode_length=4, method=self.model.decode, mutable=['cache'])
        differences = []
        for i in range(3):
            token = ids[:, i:i+1]
            cached, state = self.model.apply({'params': self.params, 'cache': state['cache']}, encoded, x, token, token,
                decode=True, enable_dropout=False, max_decode_length=4, method=self.model.decode, mutable=['cache'])
            differences.append(float(np.max(np.abs(np.asarray(cached[:,0]) - np.asarray(full[:,i])))))
        assert max(differences) < 1e-3, differences
        codec = vocabularies.build_codec(vocabularies.VocabularyConfig(num_velocity_bins=1))
        raw = [codec.encode_event(event_codec.Event(kind, value)) + 3 for kind, value in
               [('tie',0), ('program',0), ('velocity',1), ('pitch',69), ('shift',20), ('velocity',0), ('pitch',69)]]
        with_eos, without = self.notes([(0., raw + [1])]), self.notes([(0., raw)])
        assert len(with_eos.notes) == len(without.notes) == 1
        assert abs(with_eos.notes[0].end_time - .2) < 1e-6
        audio = .2 * np.sin(2*np.pi*440*np.arange(4000)/16000)
        start, spec = next(self.frames(audio))
        tokens, score = self.tokens(spec, max_length=32)
        return dict(cache_logit_max_abs_errors=differences, event_decode_passed=True,
                    synthetic_frames=list(spec.shape), bounded_generated_tokens=tokens.tolist(),
                    bounded_score=score, full_t5x_wrapper_parity_verified=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['setup', 'prepare', 'download', 'smoke', 'predict'])
    parser.add_argument('--mt3-source', type=Path)
    parser.add_argument('--t5x-source', type=Path)
    parser.add_argument('--audio')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    started = time.process_time()
    if args.action == 'setup':
        result = MT3Direct.setup()
    elif args.action == 'prepare':
        result = MT3Direct.prepare(args.mt3_source, args.t5x_source)
    elif args.action == 'download':
        result = MT3Direct.download_weights()
    else:
        runner = MT3Direct()
        result = runner.smoke() if args.action == 'smoke' else runner.transcribe(args.audio)
    result['cpu_seconds'] = time.process_time() - started
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)
