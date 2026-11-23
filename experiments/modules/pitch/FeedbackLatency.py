"""Exact steady-state display-age metric, independent of inference identities."""
import hashlib
import json
from zipfile import BadZipFile
from pathlib import Path

import numpy as np

CLOCK = "dedicated-worker process CPU simulation"
VERSION = "feedback_age_v1_first_completion_to_audio_end"
CAPTION = (
    "P95 feedback latency is the time-weighted 95th percentile of the age of the "
    "latest available pitch estimate during playback, measured from its audio timestamp. "
    "It includes analysis look-ahead, simulated computation and queueing, and time "
    "between updates. Initial startup and screen-rendering delay are excluded."
)


def intervals(times, latency_ms, cpu_ms, duration):
    arrays = [np.asarray(a, dtype=float) for a in (times, latency_ms, cpu_ms)]
    times, latency_ms, cpu_ms = arrays
    if (not np.isfinite(duration) or duration <= 0 or
            any(a.ndim != 1 or not np.isfinite(a).all() for a in arrays) or
            len({len(a) for a in arrays}) != 1 or not len(times)):
        raise ValueError("Invalid feedback timing arrays/duration")
    ready = times + latency_ms / 1000
    if (not np.isfinite(ready).all() or np.any(latency_ms < 0) or np.any(cpu_ms < 0) or
            np.any(np.diff(times) < 0) or np.any(np.diff(ready) < 0) or ready[0] < 0):
        raise ValueError("Negative delays or unordered feedback timeline")
    # Earlier commits at identical availability have zero display duration.
    ends = np.minimum(np.r_[ready[1:], duration], duration)
    visible = (ready < duration) & (ends > ready)
    ages = ready[visible] - times[visible]
    lengths = ends[visible] - ready[visible]
    return ages, lengths, float(ready[0])


def percentile(ages, lengths, q=0.95):
    ages, lengths = np.asarray(ages), np.asarray(lengths)
    if not len(lengths) or lengths.sum() <= 0:
        return float("nan")
    lo, hi = float(ages.min()), float((ages + lengths).max())
    target = q * lengths.sum()
    while hi - lo > 1e-10:
        mid = (lo + hi) / 2
        if np.clip(mid - ages, 0, lengths).sum() < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2 * 1000


def read(row, cache_dir=None):
    """Validate source evidence; optionally cache only the derived intervals."""
    if row.get("latency_clock") != CLOCK:
        raise ValueError("Incompatible feedback timing clock")
    source = Path(row["latency_samples_path"])
    duration = float(row["audio_seconds"])
    identity = dict(version=VERSION, clock=CLOCK, duration=duration,
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    with np.load(source, allow_pickle=False) as data:
        if len(data["output_latency_ms"]) != int(row["latency_sample_count"]):
            raise ValueError("Feedback timing count mismatch")
        values = intervals(data["estimate_times_seconds"], data["output_latency_ms"],
                           data["call_cpu_ms"], duration)
    if cache_dir is not None:
        path = Path(cache_dir) / (hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest() + ".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            try:
                with np.load(path, allow_pickle=False) as data:
                    cached = data["ages"], data["lengths"], float(data["startup"])
                    if all(np.array_equal(a, b) for a, b in zip(cached, values)):
                        return cached
            except (OSError, ValueError, KeyError, EOFError, BadZipFile):
                pass  # Derived corruption never invalidates inference evidence.
        np.savez_compressed(path, ages=values[0], lengths=values[1], startup=values[2])
    return values
