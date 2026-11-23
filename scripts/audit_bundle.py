"""Print app.py's local import closure and the exact repository data manifest.

Run with Python 3.12; no third-party dependencies are needed. This static audit
includes function-local imports, but PyInstaller hooks handle third-party lazy
imports and native dependencies at build time.
"""
import ast
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packaging"))
from bundle_config import resource_files


def import_closure():
    pending = [ROOT / "app.py"]
    seen = set()
    external = set()

    def enqueue(module):
        parts = module.split(".")
        local = (ROOT / parts[0]).is_dir()
        for i in range(1, len(parts) + 1):
            stem = ROOT.joinpath(*parts[:i])
            for path in (stem.with_suffix(".py"), stem / "__init__.py"):
                if path.is_file():
                    pending.append(path)
                    local = True
        if not local and parts[0] not in sys.stdlib_module_names:
            external.add(parts[0])

    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    enqueue(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.level:
                    raise ValueError(f"Relative import needs audit support: {path}")
                enqueue(node.module)
                for alias in node.names:
                    enqueue(f"{node.module}.{alias.name}")
    return sorted(seen), sorted(external)


if __name__ == "__main__":
    modules, external = import_closure()
    resources = resource_files()
    forbidden = {"experiments", "benchmarks", "data"}
    leaked = [p for p in [*modules, *resources] if p.relative_to(ROOT).parts[0] in forbidden]
    if leaked:
        raise RuntimeError(f"Research files entered the app bundle: {leaked}")
    print(json.dumps({
        "entrypoint": "app.py",
        "local_modules": [p.relative_to(ROOT).as_posix() for p in modules],
        "direct_external_imports": external,
        "resources": [p.relative_to(ROOT).as_posix() for p in resources],
        "resource_bytes": sum(p.stat().st_size for p in resources),
    }, indent=2))
