# Inference source scope migration

`inference-source-scopes.json` maps verified historical whole-file SHA-256 hashes
to inference-scoped AST hashes for three shared note benchmark modules. This lets
existing checkpoints adopt scoped keys without rerunning unchanged algorithms.
It contains no run paths, track IDs, or scheduling decisions.

For the October 14, 2026 migration, the pre-MT3 source snapshots were reconstructed
by reversing the registry/runtime integration edits. Their complete file hashes
were checked against saved run metadata before computing the scoped hashes.
The intermediate integration snapshots were hashed directly. The retained
inference ASTs matched across these snapshots; only registry and orchestration
code differed. Method dispatch is separately included in new cache identities.

New provenance records both the complete source hash and scoped hash. A scope is
used only if its recorded complete hash matches. Known historical hashes use this
receipt; unknown historical hashes remain raw and cannot silently acquire the
current scope. Future inference changes produce new scoped hashes and invalidate
affected jobs normally. Do not extend this receipt without verifying the exact
historical source and its inference scope.
