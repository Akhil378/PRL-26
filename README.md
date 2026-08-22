# Reproducing and extending InfiniteYou

Course project, Project Representation Learning (FAU, summer 2026).
Paper: Jiang et al., *InfiniteYou: Flexible Photo Recrafting While Preserving
Your Identity*, ICCV 2025. Upstream code: `github.com/bytedance/InfiniteYou`.

- **Part 1** reproduces the Table 1 comparison of InfU against PuLID-FLUX.
- **Part 2** stress-tests identity preservation under out-of-distribution
  stylization, and measures *face masking* directly.

See `benchmark/GENERATION.md` for why the benchmark had to be rebuilt and how.

## Layout

| Path | Contents |
|---|---|
| `benchmark/` | Reconstructed prompts, identity metadata, frozen manifests |
| `src/` | Manifest builder, generation runners, metrics, analysis |
| `slurm/` | TinyGPU batch scripts |
| `tools/` | Benchmark construction and validation |
| `configs/` | Pre-registered run and metric settings |
| `report/` | Final report sources |

Generated images and score tables live on `$WORK`, never in this repository.

## Status

- [x] Benchmark rebuilt and frozen â€” 200 prompts, 1,500 projected cells
- [x] Part 2 style grid â€” 36 base prompts x 4 conditions, 576 cells/method
- [x] Manifest builder with derived, paired seeds
- [ ] Identity set curated (15 CC0 portraits, contamination-free)
- [x] Generation runners (InfU + PuLID), sharded and resumable
- [x] Metric harness — ID Loss x2 recognisers, CLIPScore, PickScore, FMI
- [x] SLURM scripts and pre-registered configs
- [x] Local test suite (crop geometry, sharding, resume) — 31 checks passing
- [ ] Identity set curated (15 CC0 portraits, contamination-free)
- [ ] Smoke test + timing calibration on A100
- [ ] Part 1 run
- [ ] Part 2 run

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
