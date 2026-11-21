# Note benchmarks

The module follows the pitch benchmark layout:

- `NoteBenchmarker.py`: method registry, historical component benchmarks, dataset orchestration, CPU-timed evaluation, parallel workers, scoring, significance and CLI.
- `NoteDetectorBase.py`: shared detector contract, method dispatch, recording preparation and note conversion helpers.
- `NoteNotebook.py`: notebook configuration exports and presentation facade over the evaluation runner.
- `NoteCache.py`: note serialization, pitch-cache adapters, result compatibility and interrupted-run recovery for all methods.
- `competitors/`: one class-owned implementation per competitor: Attune, BasicPitch, CrepeNotes, Tony, Ruptures and SlopeWindow. Ruptures owns the PELT, KernelCPD, bottom-up, window and dynamic-programming variants.
- `sweeps/`: note-detection, injected-note and string-edit parameter experiments.
- `tests/`: focused validation without requiring neural models or benchmark corpora.

## Entry points

Run commands from the repository root:

```sh
python -m benchmarks.modules.note.NoteBenchmarker --list-methods
python -m benchmarks.modules.note.NoteBenchmarker --help
python -m unittest discover -s benchmarks/modules/note/tests
```

The CLI retains the historical cached-pitch component experiment. The current end-to-end evaluation is launched from `benchmarks/notebooks/note.ipynb`:

```python
from benchmarks.modules.note.NoteNotebook import NoteNotebook, NotebookConfig

benchmark = NoteNotebook(NotebookConfig(force_methods=()))
rows = benchmark.run_preliminary_coco()
```

`NoteNotebook` also exposes `run_preliminary_urmp()` and `run_preliminary_bach10()`.
Use `NoteEvaluation` from `NoteBenchmarker` for the same evaluation without notebook presentation helpers. Sweeps import from `benchmarks.modules.note.sweeps`.

## Evaluation and cache behavior

The current notebook evaluates score-conditioned Attune after alignment, robust timing refitting and repeat recovery. The optional `attune-audio-only` diagnostic stops after segmentation. External competitors use audio-only transcription. CPU totals include frontend, segmentation and, for score-conditioned Attune, refinement. Cached frontends retain their original CPU cost.

Summaries pool note-event TP/FP/FN for micro precision, recall and F1, with onset-only and offset-aware metrics reported separately. Paired significance tests compare the common successful track intersection and swap complete source groups.

`force_methods=("attune",)` refreshes Attune predictions while retaining compatible frontend caches and competitor results. Use `force_methods=()` to resume completed jobs. Result keys track the relevant competitor implementation, shared detector/cache code, scoring, inputs and configuration. The structural refactor gets new result identities; historical artifacts remain untouched and are not automatically certified as equivalent.

Historical results measure older configurations and are not current production or held-out validation. The full [September 23 audit, with October 4 protocol updates](../../../docs/archive/note-benchmark-audit-2026-09-23.md) is archived for provenance.

## Librosa onset baselines

`librosa-flux` and `librosa-superflux` are audio-only onset-to-note baselines,
registered in both the competitor registry and `note.ipynb`. They are not complete
transcription systems from the cited papers. No new result has been generated as
part of adding them.

- **Spectral flux:** librosa's default log-Mel positive spectral difference,
  2048-sample FFT, lag 1, no frequency maximum filter, mean aggregation.
- **SuperFlux (librosa approximation):** 1024-sample FFT, 138 Mel bands,
  27.5–16000 Hz, lag 2 and a 3-bin frequency maximum filter. Both use mono
  44.1 kHz audio, a 220-sample hop (approximately 200 fps), and librosa's default
  normalized peak picker without backtracking. The FFT size is explicitly passed
  to onset_strength so centering compensation matches the actual spectrogram.
  The librosa tutorial substitutes Mel bands for the paper's log-frequency bank;
  this is not a claim of bit-for-bit reproduction of the original SuperFlux system.

Both use exactly the same note adapter: librosa pYIN at 22050 Hz, FFT/frame length
4096, hop 220, fixed 32.7032–2093.005 Hz range, default pYIN voicing. Contiguous
voiced runs are split at acoustic onsets. Run starts provide fallback attacks;
notes end at the next onset or voicing loss. Segments shorter than 30 ms or with
no pitch frames are dropped. Pitch is the median log-frequency within the segment.
No pitch-change cuts, score, annotated pitch range, alignment, repeat recovery,
or reference-based timing correction are used. Thus the existing pitch-and-onset
and offset-aware F1 measure the **combined adapter**, not pure onset detection.
The frontend timing includes loading/resampling and pYIN; segmentation includes
spectrogram, onset extraction, and note decoding. Each method pays its own pYIN
cost. Completed results are cached and resumable.

Two additions suffice here: SuperFlux directly addresses vibrato in sustained
instruments, while plain flux supplies the conventional spectral-difference
comparison. Other onset_strength feature/aggregation options are variants, not
additional independently motivated competitors.

References and suggested related-work wording:

> We compare a conventional spectral-flux onset baseline, from the family reviewed
> by Bello et al. (2005), with librosa's Mel-spectrogram approximation of SuperFlux
> (Böck and Widmer, 2013). Both use a common pYIN-based pitch and duration adapter
> for note-level evaluation.

- Bello, J. P., Daudet, L., Abdallah, S., Duxbury, C., Davies, M., and Sandler,
  M. B. (2005). *A Tutorial on Onset Detection in Music Signals*. IEEE Transactions
  on Speech and Audio Processing, 13(5), 1035–1047.
  [DOI: 10.1109/TSA.2005.851998](https://doi.org/10.1109/TSA.2005.851998).
  This is a survey of onset methods, not a named librosa “Bello” implementation.
- Böck, S., and Widmer, G. (2013). *Maximum Filter Vibrato Suppression for Onset
  Detection*. DAFx-13.
  [Paper](https://phenicx.upf.edu/system/files/publications/Boeck_DAFx-13.pdf).
- [Librosa SuperFlux recipe](https://librosa.org/doc/0.11.0/auto_examples/plot_superflux.html)
  and [onset_strength API](https://librosa.org/doc/0.11.0/generated/librosa.onset.onset_strength.html).
- Mauch, M., and Dixon, S. (2014). *pYIN: A Fundamental Frequency Estimator Using
  Probabilistic Threshold Distributions*. ICASSP. Credit this pitch frontend when
  describing the combined note adapter.

### Incremental notebook execution

Restart the kernel and Run All in `benchmarks/notebooks/note.ipynb`.
The setup cell previews cached and pending jobs for each dataset. All eight methods
are selected, with `force_methods=()`. Saved results are discovered automatically;
there are no pinned run directories or method-specific cache-only settings.
Compatible results are reused, and only missing or incompatible jobs run. Once
MT3 finishes, its results are reused on subsequent runs too.

Compatibility checks cover inputs, relevant implementation sources, method dispatch,
scoring settings, and dependencies. MT3 also fingerprints its runtime and checkpoint.
Shared modules use inference-scoped fingerprints so adding a registry entry or
changing scheduling does not invalidate existing predictions. Actual inference
changes still invalidate affected methods, including Attune. A verified legacy
fingerprint receipt migrates existing results; see `provenance/README.md`.
Original provenance and CPU timings remain attached to reused predictions.
All eight methods appear in summaries and the seven-comparison Holm-corrected family.

## Official MT3 on macOS

Run `benchmarks/notebooks/mt3-port.ipynb` for setup and bounded validation. Despite
its historical filename, this notebook now uses **official MT3 directly**, not the
unofficial PyTorch port. The network, checkpoint, spectrogram, event conversion,
and T5X beam-search code are pinned upstream components. A minimal wrapper avoids
installing T5X/SeqIO/T5 and their `tensorflow-cpu` dependency; official spectral
computation still uses the project's Mac-compatible TensorFlow.

MT3 is included in the default `NotebookConfig.methods`. Complete setup before
running the note notebook on a new machine. MT3 uses one worker and a separate CPU subprocess; cache keys include
runtime sources, checkpoint and resolved packages. All predicted instrument
notes are scored, with no score conditioning or reference-based filtering. CPU
cost includes child-process inference and compilation, not just parent overhead.

The notebook does not run the corpus. Local checks have loaded all 189 official
inference tensors, compared three cached decoder steps to full-prefix logits
(maximum difference approximately 3.8e-5), checked event boundaries with/without
EOS, and generated a bounded sequence from a 0.25-second synthetic signal. These
validate plumbing; no independent full Colab/T5X-wrapper parity run or dataset
accuracy evaluation has been performed. Audit training-set overlap before making
held-out claims. See `experiments/MT3Direct.py` for the small inference wrapper.


## YourMT3+ on macOS

The default note notebook also includes `yourmt3`: **YPTF.MoE+Multi (noPS)**,
the authors' demo default. It is a distinct trained model, not an MT3 port.
The source is pinned to Hugging Face Space `mimbres/YourMT3` revision
`5e66c1ea173a8186e0d20432b841d3180cc015b5`. The selected checkpoint is
`mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops/last.ckpt`,
SHA-256 `ae38e415c79efd5592dcb9b658cdb99ddb11d4c4e1eaa364cab04a052473fc25`.
NoPS means no pitch-shift augmentation during training.

Setup on a new machine (already completed on this Mac):

```sh
at-venv/bin/python benchmarks/modules/note/experiments/YourMT3Direct.py setup
```

This installs a private dependency overlay under the ignored pretrained assets
folder and downloads only the chosen 562 MB checkpoint. The host environment is
not modified. PyTorch/TorchAudio come from the project environment; the other
runtime packages are pinned by setup and all resolved packages enter cache keys.

Inference uses float32 CPU with two persistent model processes. Each process
serves two recordings concurrently on threads sharing one set of weights: four
recordings in flight, two model instances total. Each request remains single-threaded
inside PyTorch with segment batch size one. The first forward per model is serialized
to initialize upstream positional buffers before concurrent reads. The wrapper calls the authors' model, preprocessing, token generation, and
note merging. Only training-Trainer construction is bypassed; SoundFile supplies
waveforms instead of TorchAudio's file-loader backend. All inference weights must
load strictly after upstream's exclusion of training-only pitchshift tensors.
No network architecture or positional encoding changes are made.

All predicted instruments are scored without score conditioning or reference-based
filtering. New shared-runtime CPU timings use per-request thread CPU plus an equal share
of model startup across the recordings assigned to that process. Existing cached
serial timings retain their original values and clock labels; do not interpret a
mixed resumed run as a controlled throughput comparison. Source files, checkpoint bytes,
runtime packages and wrapper code participate in automatic cache compatibility.
Adding this method does not invalidate compatible old-method results.

The October 14 Mac smoke test used only the first 2.048 seconds of one Coco
recording (two padded model windows), full default decoding, and all 13 channels.
It loaded 47,078,022 parameters and produced one finite pitched note, taking
about 7.4 seconds after imports (6.6 CPU seconds). Upstream reported one segment-tie
merge diagnostic at the cropped boundary; the wrapper preserves this in the smoke
receipt. This validates execution, not transcription quality or GPU parity.
See `provenance/yourmt3-smoke-2026-10-14.json`. No full benchmark was launched.

URMP is included among the paper's training datasets. Audit exact split overlap
before describing our URMP comparison as held-out evaluation.

Sources: [authors' release instructions](https://github.com/mimbres/YourMT3/issues/2),
[released demo/code/checkpoints](https://huggingface.co/spaces/mimbres/YourMT3/tree/main),
[YourMT3+ paper](https://arxiv.org/abs/2407.04822).


### Shared YourMT3+ execution and resume

Restart the note notebook kernel after changing execution code. Run All discovers
completed recording checkpoints before starting any models. The shared scheduler
loads only two model instances and reuses them for the pending recordings in each
dataset. Every finished recording is scored and checkpointed immediately; stopping
the notebook terminates both model processes. In-flight recordings without a
completed checkpoint will run again on restart. An individual recording failure
is recorded as an error and remains eligible for retry.

`YourMT3Shared.py` changes execution only: it calls the unchanged
`YourMT3Direct.predict` with an already-loaded model. Execution settings are recorded
separately from inference cache identity. The model/checkpoint, numerical prediction
body, upstream sources, and runtime packages still determine compatibility.

A four-request Mac smoke test (two trombone clips and two silence clips) verified
exact serial-output equality for note intervals, frequencies, programs, token
shapes and decoding diagnostics. It also verified two distinct process IDs and
one shared model identity per overlapping pair. See
`provenance/yourmt3-shared-smoke-2026-10-14.json`. This was a bounded functional test,
not a full-corpus speed or peak-memory benchmark.
