"""Pinned author-hosted CocoChorales-E test data; no synthesis.

Only selected test pieces are downloaded. The Coco mirror is explicitly the
LadderSym subset, not a claim of byte-equivalence to PolyTune's original Globus
release. A manifest records the exact selection, paths and content hashes.
"""

import hashlib
import json
from pathlib import Path
import re
import urllib.parse
import urllib.request

DATASETS = {
    "coco": (
        "ben2002chou/CocoChorales-E",
        "cd54562f0722efe80d115183f260bf94aa892a7e",
        "split.json",
    )
}
SPLIT_SHA256 = {
    "coco": "06d9b9a950d78fc21383dcf643fc167bbe6b87c0dd6491203cb2ec742bf537a8"
}
LABELS = {"extra": "extra_notes", "missed": "removed_notes", "correct": "correct_notes"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def download(repo, revision, relative, target, *, dataset=True, sha256=None):
    target = Path(target)
    if target.exists():
        if sha256 and digest(target) != sha256:
            raise ValueError(f"Checksum mismatch: {target}")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    prefix = "datasets/" if dataset else ""
    url = f"https://huggingface.co/{prefix}{repo}/resolve/{revision}/{urllib.parse.quote(relative)}"
    temporary = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as source, temporary.open(
        "wb"
    ) as out:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            out.write(chunk)
    if sha256 and digest(temporary) != sha256:
        raise ValueError(f"Checksum mismatch: {temporary}")
    temporary.replace(target)
    return target


def tree(repo, revision, relative, cache):
    """Paginate HF's tree API (the repository-info siblings list is truncated)."""
    cache = Path(cache)
    if cache.exists():
        return json.loads(cache.read_text())
    url = f"https://huggingface.co/api/datasets/{repo}/tree/{revision}/{urllib.parse.quote(relative)}?limit=1000"
    entries = []
    while url:
        with urllib.request.urlopen(url, timeout=120) as response:
            entries.extend(json.load(response))
            link = response.headers.get("Link", "")
        match = re.search('<([^>]+)>; rel="next"', link)
        url = match.group(1) if match else None
    save_json(cache, entries)
    return entries


def test_ids(split):
    return {
        Path(path).stem
        for key, path in split["midi_filename"].items()
        if split["split"][key] == "test"
    }


def stem_key(path):
    """Match by full stem identity, never positional zip or track ordinal alone."""
    return re.sub("[ _]+", "_", Path(path).stem.casefold())


def pair_piece(dataset, piece, entries):
    """Resolve authored inputs and labels, rejecting absent/ambiguous pairs."""
    if dataset != "coco":
        raise ValueError("Only native CocoChorales-E instrument stems are supported")

    def indexed(folder):
        result = {}
        for entry in entries[folder]:
            if entry["type"] != "file":
                continue
            key = stem_key(entry["path"])
            if key in result:
                raise ValueError(f"Ambiguous stem in {folder}: {key}")
            result[key] = entry
        return result

    score = indexed("score_midi")
    audio = indexed("score_audio")
    performance = indexed("performance")
    labels = {kind: indexed(kind) for kind in LABELS}
    rows = []
    for key, midi in sorted(score.items()):
        label_keys = {kind: key for kind in labels}
        required = [
            audio.get(key),
            performance.get(key),
            *(files.get(label_keys[kind]) for kind, files in labels.items()),
        ]
        if any((entry is None for entry in required)):
            raise ValueError(f"Incomplete official {dataset} piece/stem: {piece}/{key}")
        paths = {
            "score_midi": midi,
            "score_audio": audio[key],
            "performance": performance[key],
            **{kind: files[label_keys[kind]] for kind, files in labels.items()},
        }
        rows.append(
            {"case_id": f"{piece}/{key}", "piece": piece, "stem": key, "files": paths}
        )
    if not rows:
        raise ValueError(f"No official cases found for {piece}")
    return rows


def prepare(dataset, root, *, max_pieces=3, seed=0):
    """Select pieces before looking at labels/results; None selects full mirror test split."""
    if max_pieces is not None and (not isinstance(max_pieces, int) or max_pieces < 1):
        raise ValueError("max_pieces must be a positive integer or None")
    if dataset != "coco":
        raise ValueError("Only native CocoChorales-E instrument stems are supported")
    repo, revision, split_name = DATASETS[dataset]
    root = Path(root).resolve()
    cache = root / ".inventory" / revision
    split_path = download(
        repo, revision, split_name, root / split_name, sha256=SPLIT_SHA256[dataset]
    )
    wanted = test_ids(json.loads(split_path.read_text()))
    available = {
        Path(e["path"]).name
        for e in tree(repo, revision, "score", cache / "pieces.json")
        if e["type"] == "directory"
    }
    selected = sorted(
        wanted & available,
        key=lambda p: hashlib.sha256(f"{seed}:{p}".encode()).hexdigest(),
    )
    total_pieces = len(selected)
    if max_pieces is not None:
        selected = selected[:max_pieces]
    if not selected:
        raise ValueError("No official test pieces found")
    cases = []
    for index, piece in enumerate(selected):
        print(
            f"{dataset}: preparing piece {index + 1}/{len(selected)}: {piece}",
            flush=True,
        )
        folders = {
            "score_midi": f"score/{piece}/stems_midi",
            "score_audio": f"score/{piece}/stems_audio",
            "performance": f"mistake/{piece}/stems_audio",
            **{k: f"label/{v}/{piece}/stems_midi" for k, v in LABELS.items()},
        }
        entries = {}
        for key, folder in folders.items():
            suffix = ".wav" if key in ("score_audio", "performance") else ".mid"
            entries[key] = [
                e
                for e in tree(repo, revision, folder, cache / (folder + ".json"))
                if e["type"] == "file" and e["path"].endswith(suffix)
            ]
        for case in pair_piece(dataset, piece, entries):
            files, hashes = ({}, {})
            for key, entry in case.pop("files").items():
                path = download(
                    repo,
                    revision,
                    entry["path"],
                    root / entry["path"],
                    sha256=entry.get("lfs", {}).get("oid"),
                )
                files[key], hashes[key] = (str(path), digest(path))
            case.update(files=files, hashes=hashes)
            cases.append(case)
    manifest = dict(
        dataset=dataset,
        repo=repo,
        revision=revision,
        split="test",
        seed=seed,
        max_pieces=max_pieces,
        available_test_pieces=total_pieces,
        selected_pieces=selected,
        split_sha256=digest(split_path),
        cases=cases,
        scope="full author mirror test split" if max_pieces is None else "test subset",
        provenance_note="Coco mirror is the author-hosted LadderSym subset; equivalence to the original PolyTune Globus release is not established.",
    )
    path = root / f"manifest_seed{seed}_pieces{max_pieces}.json"
    save_json(path, manifest)
    return path


def prepare_models(dataset, *, methods=("PolyTune", "LadderSym")):
    from benchmarks.modules.mistake.competitors.PolyTune import PolyTune
    from benchmarks.modules.mistake.competitors.LadderSym import LadderSym

    if dataset != "coco":
        raise ValueError("Only native CocoChorales-E instrument stems are supported")
    specs = {
        "PolyTune": (
            "ben2002chou/Polytune",
            "8e3d59ec7b4cef371f26b2ee7770be814bdad41c",
            "CocoChorales-E/last.ckpt",
            PolyTune.ASSETS / "coco.ckpt",
            PolyTune.CHECKPOINT_SHA256,
        ),
        "LadderSym": (
            "ben2002chou/laddersym-checkpoints",
            "f0672031ee232318df4afb921156048bdb09c07c",
            "checkpoints/cocochorales/prompted/model.ckpt",
            LadderSym.ASSETS / "coco_prompted.ckpt",
            LadderSym.CHECKPOINT_SHA256,
        ),
    }
    for method in methods:
        if method not in specs:
            continue
        repo, revision, relative, target, sha = specs[method]
        print(
            f"{dataset}: verifying/downloading official {method} weights: {target}",
            flush=True,
        )
        download(repo, revision, relative, target, dataset=False, sha256=sha)
