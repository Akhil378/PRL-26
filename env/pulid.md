# PuLID-FLUX environment

PuLID runs on the Black Forest Labs `flux` codebase, not Diffusers, and pulls in
EVA-CLIP. It conflicts with the InfU pins; keep it in a separate conda prefix.

```bash
cd $WORK/prl26
git clone https://github.com/ToTheBeginning/PuLID.git
conda create -p $WORK/envs/pulid python=3.11 -y
conda activate $WORK/envs/pulid
pip install -r PuLID/requirements.txt
pip install pandas pyarrow
```

Weights: run `tools/stage_pulid.py` (step 6 of `tools/setup_cluster.sh`). Do not
try to point `FLUX_DEV` and `AE` at the files -- in the pinned checkout those env
vars are only read for `flux-schnell`. The `flux-dev` spec in `flux/util.py`
hardcodes `ckpt_path='models/flux1-dev.safetensors'` and
`ae_path='models/ae.safetensors'`, both relative to the CWD, so setting the env
vars is a silent no-op. `pipeline_flux.py` likewise hardcodes
`models/antelopev2/` and `models/pulid_flux_<version>.safetensors`.

The staging script therefore makes the repo root's `models/` a symlink to
`$WORK/prl26/models` and fills it. `flux1-dev.safetensors` and `ae.safetensors`
are **linked out of the existing HF cache**, not re-downloaded: `load_flow_model`
and `load_ae` check `os.path.exists()` before any hub call, so presence is all
that is required. That saves 24 GB of transfer.

Memory on A100 40 GB: bf16 + `--offload` stays under 30 GB and preserves facial
detail. Avoid `--fp8` for the headline runs; the PuLID docs note degradation in
exactly the facial detail this project measures.

**Verify the `FluxGenerator.generate_image` signature against the checkout
before the first batch run** -- `src/gen_pulid.py` follows upstream `app_flux.py`
and the argument names have moved between PuLID revisions. Run with
`--limit 2` first.
