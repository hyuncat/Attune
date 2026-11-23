"""Pinned authors' YourMT3+ YPTF.MoE+Multi (noPS), CPU inference only."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
ASSETS = ROOT / 'experiments/modules/note/competitors/data/yourmt3'
SOURCE = ASSETS / 'source'
REVISION = '5e66c1ea173a8186e0d20432b841d3180cc015b5'
MODEL = 'mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops'
CHECKPOINT_PATH = f'amt/logs/2024/{MODEL}/checkpoints/last.ckpt'
CHECKPOINT_SHA256 = 'ae38e415c79efd5592dcb9b658cdb99ddb11d4c4e1eaa364cab04a052473fc25'
REQUIREMENTS = ['transformers==4.45.1', 'pytorch-lightning==2.4.0',
                'einops==0.8.0', 'wandb==0.18.3', 'deprecated==1.2.14', 'setuptools==75.1.0']
MODEL_ARGS = [MODEL + '@last.ckpt', '-p', '2024', '-tk', 'mc13_full_plus_256',
              '-dec', 'multi-t5', '-nl', '26', '-enc', 'perceiver-tf', '-sqr', '1',
              '-ff', 'moe', '-wf', '4', '-nmoe', '8', '-kmoe', '2', '-act', 'silu',
              '-epe', 'rope', '-rp', '1', '-ac', 'spec', '-hop', '300', '-atc', '1',
              '-pr', '32', '-w', '0']


def sha256(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def setup():
    """Install an isolated overlay; download only the chosen official checkpoint."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    if not SOURCE.exists():
        subprocess.run(['git', 'clone', 'https://huggingface.co/spaces/mimbres/YourMT3', str(SOURCE)],
                       env={**os.environ, 'GIT_LFS_SKIP_SMUDGE': '1'}, check=True)
        subprocess.run(['git', '-C', str(SOURCE), 'checkout', '--detach', REVISION],
                       env={**os.environ, 'GIT_LFS_SKIP_SMUDGE': '1'}, check=True)
    actual = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != REVISION:
        raise ValueError(f'Expected upstream revision {REVISION}, got {actual}')
    python = ASSETS / 'venv/bin/python'
    if not python.exists():
        subprocess.run([str(ROOT / 'at-venv/bin/python'), '-m', 'venv', str(ASSETS / 'venv')], check=True)
    host_site = subprocess.check_output([str(ROOT / 'at-venv/bin/python'), '-c',
        'import sysconfig; print(sysconfig.get_paths()["purelib"])'], text=True).strip()
    site = next((ASSETS / 'venv/lib').glob('python*/site-packages'))
    (site / 'attune-host.pth').write_text(host_site + '\n')
    subprocess.run([str(python), '-m', 'pip', 'install', *REQUIREMENTS], check=True)
    checkpoint = ASSETS / 'model.ckpt'
    if not checkpoint.exists():
        temporary = checkpoint.with_suffix('.download')
        subprocess.run(['curl', '-L', '--fail', '--output', str(temporary),
            f'https://huggingface.co/spaces/mimbres/YourMT3/resolve/{REVISION}/{CHECKPOINT_PATH}'], check=True)
        if sha256(temporary) != CHECKPOINT_SHA256:
            raise ValueError('Checkpoint SHA-256 mismatch')
        temporary.replace(checkpoint)
    if sha256(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError('Checkpoint SHA-256 mismatch')


def load_model():
    os.environ["USE_TF"] = "0"
    os.environ["HF_HOME"] = str(ASSETS / "hf-cache")
    os.environ.setdefault("MPLCONFIGDIR", str(ASSETS / "mpl-cache"))
    from copy import deepcopy
    from unittest.mock import patch
    import torch
    sys.path[:0] = [str(SOURCE), str(SOURCE / 'amt/src')]
    import model_helper
    from config.config import shared_cfg
    # The demo's initializer creates a training framework Trainer. Inference
    # only needs its config and paths; avoid GPU selection and logging side effects.
    def inference_context(args, stage):
        args.exp_id = MODEL
        return None, None, {'lightning_dir': str(ASSETS),
                           'last_ckpt_path': str(ASSETS / 'model.ckpt')}, deepcopy(shared_cfg)
    original = torch.nn.Module.load_state_dict
    def strict_load(module, state_dict, strict=True, **kwargs):
        # Upstream removes training-only pitchshift tensors before this call.
        return original(module, state_dict, strict=True, **kwargs)
    with patch.object(model_helper, 'initialize_trainer', inference_context), patch.object(
            torch.nn.Module, 'load_state_dict', strict_load):
        model = model_helper.load_model_checkpoint(MODEL_ARGS, device='cpu')
    return model


def predict(audio, output):
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    started = time.process_time()
    wall = time.monotonic()
    torch.set_num_threads(1)
    model = load_model()
    from utils.audio import slice_padded_array
    from utils.note2event import mix_notes
    from utils.event2note import merge_zipped_note_events_and_ties_to_notes
    # SoundFile replaces torchaudio's file loader only; model preprocessing uses
    # the same mono averaging, torchaudio resampling, padding and spectrogram.
    samples, sr = sf.read(audio, dtype='float32', always_2d=True)
    waveform = torch.from_numpy(samples.T).mean(dim=0, keepdim=True)
    waveform = torchaudio.functional.resample(waveform, sr, model.audio_cfg['sample_rate'])
    size = model.audio_cfg['input_frames']
    segments = slice_padded_array(waveform, size, size)
    segments = torch.from_numpy(segments.astype('float32')).unsqueeze(1)
    with torch.inference_mode():
        tokens, _ = model.inference_file(bsz=1, audio_segments=segments)
    starts = [size * i / model.audio_cfg['sample_rate'] for i in range(len(segments))]
    channels = []
    errors = {}
    for ch in range(model.task_manager.num_decoding_channels):
        batches = [arr[:, ch, :] for arr in tokens]
        zipped, _, token_errors = model.task_manager.detokenize_list_batches(batches, starts, return_events=True)
        notes, note_errors = merge_zipped_note_events_and_ties_to_notes(zipped)
        channels.append(notes)
        for counts in (token_errors, note_errors):
            for key, value in counts.items():
                errors[key] = errors.get(key, 0) + int(value)
    notes = mix_notes(channels)
    # Score all predicted pitches, including wrong instruments, without references.
    valid = [n for n in notes if np.isfinite(n.onset) and np.isfinite(n.offset) and n.offset > n.onset]
    result = dict(intervals=[[float(n.onset), float(n.offset)] for n in valid],
                  frequencies_hz=[float(440 * 2 ** ((n.pitch - 69) / 12)) for n in valid],
                  programs=[int(n.program) for n in valid],
                  cpu_seconds=time.process_time()-started, wall_seconds=time.monotonic()-wall,
                  segments=len(segments), decoding_channels=model.task_manager.num_decoding_channels,
                  token_shapes=[list(arr.shape) for arr in tokens], decoding_errors=errors,
                  parameters=sum(p.numel() for p in model.parameters()), strict_checkpoint_load=True)
    Path(output).write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['setup', 'predict'])
    parser.add_argument('--audio')
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.command == 'setup':
        setup()
    else:
        predict(args.audio, args.output)
