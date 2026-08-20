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

Weights: `flux1-dev.safetensors` and `ae.safetensors` from the gated
`black-forest-labs/FLUX.1-dev` repo; the PuLID weights auto-download from
`guozinan/PuLID`. Point `FLUX_DEV` and `AE` at the files under `$WORK`.

Memory on A100 40 GB: bf16 + `--offload` stays under 30 GB and preserves facial
detail. Avoid `--fp8` for the headline runs; the PuLID docs note degradation in
exactly the facial detail this project measures.

**Verify the `FluxGenerator.generate_image` signature against the checkout
before the first batch run** -- `src/gen_pulid.py` follows upstream `app_flux.py`
and the argument names have moved between PuLID revisions. Run with
`--limit 2` first.
