"""Registry edits preserve caches; actual inference changes invalidate them."""
import copy
import json
from pathlib import Path
import unittest
from benchmarks.modules.note.NoteCache import NoteCache
from benchmarks.modules.note.tests import test_note_result_cache as cache_tests


class InferenceScopeTest(unittest.TestCase):
    root = Path(__file__).resolve().parents[4]

    def fingerprint(self, name, source):
        return NoteCache.inference_source_fingerprint(f'{name}.py', source)

    def test_registry_changes_but_inference_changes_do_not_share_keys(self):
        source = (self.root / 'benchmarks/modules/note/NoteDetectorBase.py').read_text()
        original = self.fingerprint('NoteDetectorBase', source)
        registry = source.replace('"attune": Attune,', '"attune": Attune, "new-model": Attune,')
        self.assertEqual(original, self.fingerprint('NoteDetectorBase', registry))
        helper = source.replace('n_candidates: int = 3', 'n_candidates: int = 2')
        self.assertNotEqual(original, self.fingerprint('NoteDetectorBase', helper))
        source = (self.root / 'benchmarks/modules/note/NoteCache.py').read_text()
        original = self.fingerprint('NoteCache', source)
        self.assertNotEqual(original, self.fingerprint('NoteCache', source.replace(
            'def attune_frontend(task, config, recording, cfg, adapter, conditioning):',
            'def attune_frontend(task, config, recording, cfg, adapter, conditioning=False):')))

    def test_verified_migration_and_unknown_sources(self):
        aliases = json.loads((self.root / 'benchmarks/modules/note/provenance/inference-source-scopes.json').read_text())
        for path, known in aliases.items():
            for raw, semantic in known.items():
                self.assertEqual(NoteCache.scoped_sources({'sources': {path: raw}})[path], semantic)
            self.assertEqual(NoteCache.scoped_sources({'sources': {path: 'unknown-source'}})[path], 'unknown-source')

    def test_dispatch_and_raw_source_guards(self):
        tracks, p = cache_tests.NoteResultCacheTest().fixture()
        sources = p['sources']
        sources.pop('benchmarks/modules/note/NoteDetectionBaselines.py')
        for name in ('NoteDetectorBase', 'NoteCache', 'competitors/BasicPitch'):
            sources[f'benchmarks/modules/note/{name}.py'] = 'same'
        path = 'benchmarks/modules/note/NoteDetectorBase.py'
        p['inference_sources'] = {path: dict(raw_sha256='same', semantic_sha256='helper-scope')}
        p['method_dispatch'] = {'basic-pitch': 'original'}
        task = {**tracks[0], 'method': 'basic-pitch'}
        original = NoteCache.job_key(p, task)
        changed = copy.deepcopy(p)
        changed['method_dispatch']['basic-pitch'] = 'different-implementation'
        self.assertNotEqual(original, NoteCache.job_key(changed, task))
        changed = copy.deepcopy(p)
        changed['sources'][path] = 'changed-with-stale-scope'
        self.assertNotEqual(original, NoteCache.job_key(changed, task))
        changed = copy.deepcopy(p)
        changed['sources'][path] = 'new-registry'
        changed['inference_sources'][path]['raw_sha256'] = 'new-registry'
        self.assertEqual(original, NoteCache.job_key(changed, task))
