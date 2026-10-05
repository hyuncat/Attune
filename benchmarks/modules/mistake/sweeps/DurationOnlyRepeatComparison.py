"""Compatibility runner for the now-production duration-only repeat recovery."""

import hashlib
import json
from pathlib import Path

from benchmarks.modules.mistake.sweeps.CollapsedRepeatComparison import (
    run as run_comparison,
)


def duration_cost(pairs, detector):
    """Keep gap costs; matched pairs pay only absolute duration error in seconds."""
    if hasattr(pairs, "pairs"):
        pairs = pairs.pairs
    return sum(
        (
            detector.get_deletion_cost(score)
            if user is None
            else (
                detector.get_insertion_cost(user)
                if score is None
                else abs(user.duration() - score.duration())
            )
        )
        for user, score in pairs
    )


def run(source, output):
    summary = run_comparison(source, output)
    metadata_path = Path(output) / "run.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["experiment"] = {
        "name": "duration-only local repeat recovery",
        "matched_pair_cost": "abs(user duration - score duration), in seconds",
        "recovery_fee": "exact deletion cost of each recovered score note",
        "unchanged": "initial alignment, robust fit, 0.5-semitone eligibility, "
        "original matches, 30 ms minimum pieces, fewer-cuts tie break",
        "scope": "production default; duration-only scoring is local to recovery",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", default="benchmarks/results/repeat_refinement_urmp_production"
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(run(args.source, args.output).to_string(index=False))
