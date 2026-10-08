# Reproducing and extending InfiniteYou

Code and data for a reproduction of InfiniteYou (Jiang et al., *InfiniteYou:
Flexible Photo Recrafting While Preserving Your Identity*, ICCV 2025) against
PuLID-FLUX, written for the course Project Representation Learning (FAU, IDEA
Lab, summer 2026). Upstream code: `github.com/bytedance/InfiniteYou`.

- **Part 1** reproduces the paper's Table 1 comparison of InfU against
  PuLID-FLUX, under eight face recognisers and two PuLID settings.
- **Part 2** stress-tests identity preservation under out-of-distribution
  stylisation, and measures *face masking* directly.

`benchmark/GENERATION.md` explains why the benchmark had to be rebuilt and how.

## Layout

| Path | Contents |
|---|---|
| `benchmark/` | Reconstructed prompts, identity metadata, frozen manifests |
| `src/` | Manifest builder, generation runners, metrics, analysis |
| `tools/` | Benchmark construction, cluster setup, scoring, analysis, tables and figures |
| `slurm/` | Batch scripts for a SLURM cluster |
| `configs/` | Pre-registered run and metric settings |
| `env/` | Conda environments; `env/lock/` pins exact package versions and model revisions |
| `results-backup/scores/` | Per-image score tables and analysis summaries |
| `tests/` | Local tests (cropping, sharding, resume, paired statistics) |

## Data and results

The 7,201 generated images (7 GB) are not in this repository: every one can be
regenerated from the manifests, because each seed is derived from the cell key.
The per-image score tables are, so every table in the report can be rebuilt
without a GPU:

```bash
python tools/make_report_tables.py --scores results-backup/scores --out tables
```

The identity photographs (Face Research Lab London Set, CC BY 4.0) are not
redistributed. `tools/fetch_identities.py` downloads them from their DOI-stable
source; `benchmark/identities/LICENSES.md` records exactly which records they are.

## Status

Complete. Both parts were generated and scored between 22 August and
29 September 2026.

## Tests

```bash
python tools/run_local_e2e.py --clean   # 11-step pipeline proof, no GPU needed
python tests/test_robustness.py         # adversarial failure modes
```

The end-to-end run exercises the real production code path, substituting only
what a laptop cannot run: the diffusion model (`src/gen_stub.py`) and insightface
(`--id-backend stub`, which uses OpenCV YuNet + SFace instead).

## Running on a cluster

The scripts in `slurm/` were written for FAU's TinyGPU (A100 40 GB and V100
partitions, `sbatch.tinygpu`); adapt the account, partitions and paths for
another SLURM cluster. Each method has its own conda environment (`env/`),
because their pinned versions of torch and diffusers conflict.

```bash
bash tools/setup_cluster.sh                                  # once, on the frontend
sbatch.tinygpu slurm/smoke.sbatch quantize "--quantize-8bit" # measure s/image
python tools/plan_shards.py --manifest benchmark/manifest_repro.parquet --sec-per-image <MEASURED> --target-hours 2
sbatch.tinygpu slurm/gen_infu.sbatch                         # Part 1, InfU
sbatch.tinygpu slurm/gen_pulid.sbatch                        # Part 1, PuLID
sbatch.tinygpu slurm/score.sbatch infu_aes2 manifest_repro   # score each image set
```

| Stage | Scripts |
|---|---|
| Timing and pipeline gate | `smoke`, `pilot`, `probe_pulid` |
| Part 1 generation | `gen_infu`, `gen_pulid` |
| Part 2 and controls | `gen_style`, `gen_sweep`, `gen_calib`, `gen_extras` |
| Scoring | `score`, `score_irse50`, `score_ceiling`, `score_extras`, `dump_embeddings`, `score_panel` |

## Rebuilding the benchmark

```bash
python tools/build_benchmark.py
python tools/build_style_grid.py --n-base 36
python src/manifest.py --prompts benchmark/prompts_200.json --out benchmark/manifest_repro.parquet
```
