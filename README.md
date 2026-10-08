# Reproducing and extending InfiniteYou

Course project, Project Representation Learning (FAU, IDEA Lab, summer 2026).
Paper: Jiang et al., *InfiniteYou: Flexible Photo Recrafting While Preserving
Your Identity*, ICCV 2025. Upstream code: `github.com/bytedance/InfiniteYou`.

- **Part 1** reproduces the Table 1 comparison of InfU against PuLID-FLUX,
  under eight face recognisers and two PuLID settings.
- **Part 2** stress-tests identity preservation under out-of-distribution
  stylisation, and measures *face masking* directly.

The report is in `report/` (LaTeX sources on the IDEA Lab template; build with
`bash tools/build_report.sh`). Working notes, measured numbers and cluster facts
are in [CONTEXT.md](CONTEXT.md).

See `benchmark/GENERATION.md` for why the benchmark had to be rebuilt and how.

## Layout

| Path | Contents |
|---|---|
| `benchmark/` | Reconstructed prompts, identity metadata, frozen manifests |
| `src/` | Manifest builder, generation runners, metrics, analysis |
| `slurm/` | TinyGPU batch scripts |
| `tools/` | Benchmark construction and validation |
| `configs/` | Pre-registered run and metric settings |
| `results-backup/scores/` | Per-image score tables and analysis summaries |
| `env/lock/` | Exact package versions and model revisions per environment |
| `report/` | Report sources, generated tables, figures |
| `talk/` | Presentation script and Q&A notes |
| `tests/` | Local tests (cropping, sharding, resume, paired statistics) |

The 7,201 generated images (7 GB) are not in this repository: every one can be
regenerated from the manifests, because each seed is derived from the cell key.
The per-image score tables are, so every table in the report can be rebuilt
without a GPU:

```bash
python tools/make_report_tables.py --scores results-backup/scores --out report/tables
```

The identity photographs (Face Research Lab London Set, CC BY 4.0) are not
redistributed; see `benchmark/identities/LICENSES.md` for the source.

## Status

Complete. Both parts were generated and scored on the FAU TinyGPU cluster
between 22 August and 29 September 2026, and the report was submitted on
8 October 2026.

## Tests

```bash
python tools/run_local_e2e.py --clean   # 11-step pipeline proof, no GPU needed
python tests/test_robustness.py         # adversarial failure modes
```

The end-to-end run exercises the real production code path, substituting only
what a laptop cannot run: the diffusion model (`src/gen_stub.py`) and insightface
(`--id-backend stub`, which uses OpenCV YuNet + SFace instead).

## Submitting on the cluster

See [slurm/SUBMITTING.md](slurm/SUBMITTING.md). Short version:

```bash
bash tools/setup_cluster.sh                                  # once, on the frontend
sbatch.tinygpu slurm/smoke.sbatch quantize "--quantize-8bit" # measure s/image
python tools/plan_shards.py --manifest benchmark/manifest_repro.parquet     --sec-per-image <MEASURED> --target-hours 2              # size the array
sbatch.tinygpu slurm/gen_infu.sbatch                         # then the real run
```

## Rebuilding the benchmark

```bash
python tools/build_benchmark.py
python tools/build_style_grid.py --n-base 36
python src/manifest.py --prompts benchmark/prompts_200.json --out benchmark/manifest_repro.parquet
```
