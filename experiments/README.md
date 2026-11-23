# Experiments

- **Benchmarks:** [`benchmarks/`](benchmarks/) contains the pitch, note, and mistake evaluation notebooks. Implementations are in [`modules/pitch/`](modules/pitch/), [`modules/note/`](modules/note/), and [`modules/mistake/`](modules/mistake/).
- **Parameter sweeps:** [`sweeps/`](sweeps/) contains the three sweep notebooks; shared runners and helpers are in [`modules/sweeps/`](modules/sweeps/).
- **Datasets:** [`datasets/bach10/`](datasets/bach10/), [`datasets/urmp/`](datasets/urmp/), [`datasets/coco/`](datasets/coco/), and [`datasets/coco-e2/`](datasets/coco-e2/). The Coco directory contains the COCO-E audit; COCO-E2 contains the generation notebook, frozen track manifests, and materialization script.
- **Competitors:** each `modules/{pitch,note,mistake,vibrato}/competitors/` contains its implementations; downloaded models and associated assets live in its gitignored `data/` directory.
- **Results for the paper:** [`results/`](results/) contains interpretable tables, per-track results, selections, and provenance. Notebook caches are local and gitignored under [`results/caches/`](results/caches/).
- **Historical work:** [`archive/`](archive/) preserves earlier experiments, notebooks, and unused datasets. Additional vibrato implementations remain in [`modules/vibrato/`](modules/vibrato/).

Run notebooks from a checkout with the research dependencies in `requirements.txt`. Downloaded datasets, model assets, and caches are not committed. The entire `experiments/` directory, including paper results and caches, is excluded from macOS and web releases.
