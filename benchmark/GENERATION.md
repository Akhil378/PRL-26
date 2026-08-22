# Benchmark provenance

This file records how the evaluation benchmark was constructed. It exists because
the original one does not: the InfiniteYou release
(`github.com/bytedance/InfiniteYou`) ships the model pipeline and three example
photographs, and no part of the evaluation setup. The 200 GPT-4o portrait
prompts, the 15 identity samples and the metric implementations behind Table 1
are all internal to ByteDance.

Absolute agreement with the paper's numbers is therefore not attainable. What is
attainable, and what this benchmark is built for, is reproducing the **ranking of
the methods and the size of the gaps between them** on a benchmark that is fully
specified and public.

## Prompts

**Produced:** 20 August 2026, by Claude Opus 5 (Anthropic), then hand-checked.

The paper states only that its benchmark was GPT-4o-generated and covers
"different prompt lengths, face sizes, views, scenes, ages, races, complexities"
with gender information attached. The reconstruction follows those declared axes.

| Field | Levels |
|---|---|
| `gender` | female, male |
| `length` | short (<=8 words), medium (9-20), long (>=21) |
| `face_size` | closeup, waist, full |
| `complexity` | plain, single_object, busy |
| `age_hint` | none, young, senior |

Realised marginals are printed by `tools/build_benchmark.py` and must be quoted
from that output rather than from intent.

### Design decisions that deviate from the paper

1. **`length` is derived, not asserted.** `prompts_raw.jsonl` holds the authored
   text and the semantic tags; the word count and its bin are computed by the
   build script. A tag can therefore never drift away from the text it labels.

2. **Ethnicity is deliberately absent from the prompts.** The paper's own
   examples include prompts such as "Asian girl in garden". Pairing a prompt that
   specifies ethnicity with an identity photograph of a different ethnicity puts
   the text conditioning and the identity conditioning in direct conflict, and
   any resulting ID Loss is then uninterpretable -- it cannot be separated from
   the model resolving a contradiction it was handed. Ethnicity is carried by the
   identity image alone. Coverage across ethnicities is a property of the
   identity set, not the prompt set.

3. **Age hints are kept, and tagged.** 10% of prompts specify young or senior,
   balanced across gender. These create the same text/identity conflict as
   above, deliberately, so that the conflict can be *measured* as its own
   stratum rather than being spread invisibly through the benchmark.

4. **The male and female subsets are content-parallel.** Prompts are written in
   matched pairs (p068<->p168, p099<->p199, ...) that differ only in the subject.
   Realised `gender x length` and `gender x face_size` cross-tabs are exactly
   equal, so any gender difference in the results cannot be attributed to the
   two subsets describing different scenes.

5. Seven prompts printed in the paper's Figures 1 and 5 are included verbatim
   (for example p001 "Blonde woman in office", p101 "Old man with beard") so that
   part of the qualitative panel is directly comparable to theirs.

### Recovering the paper's pairing scheme

The paper reports 200 prompts, 15 identities and **1,497** outputs. The full
cross product is 3,000, so pairing is not exhaustive. Prompts carry gender and
are paired with "appropriate" identities, which halves it.

This benchmark with a 8F/7M identity split projects **1,500** cells -- within
three of the reported figure. That agreement is the evidence that the
gender-matched reading is correct; it was a hypothesis before the benchmark was
built and a measurement afterwards.

## Identities

**Status: not yet curated.** `identities.json` holds the schema and the intended
balance; every row is `"status": "pending"` and `src/manifest.py` refuses to
build a real manifest until they are marked ready.

Sourcing constraints, in priority order:

1. **No dataset used in InfU stage-1 pretraining.** That rules out FFHQ, CelebA,
   CelebV-HQ, CelebV-Text, VGGFace2, MillionCelebs, VFHQ, EasyPortrait and
   CosmicManHQ. Drawing test identities from InfU's training data would evaluate
   it on data it has seen -- and because PuLID-FLUX was trained on a different
   corpus, the contamination would be **asymmetric** and would silently favour
   InfU. The paper does not state where its 15 identities came from.

2. **Licence-clean and documented.** Photographs of real people appear in the
   report. Every row records `source` and `licence`; CC0 portrait photography is
   the intended source.

3. **Balanced by construction:** 8 female / 7 male, spread across three age bands
   and three skin-tone bands. The paper claims coverage of diverse identities,
   races and age groups and never demonstrates it; a per-subgroup breakdown of
   the results is a cheap addition that it lacks.

4. **Technically valid:** one face, frontal to three-quarter, unobstructed,
   >=512 px on the face crop, neutral or mild expression, no heavy filtering.
   Each candidate must be detected by `antelopev2` before adoption.

The `part2` flag marks the 8-identity subset (4F/4M) used for the stylization
experiment.

## Seeds

Seeds are derived, never drawn:

```
seed = int(sha256(f"{iid}|{base_pid}").hexdigest()[:8], 16) % 2**31
```

Two consequences, both deliberate:

- **Across methods** -- InfU and PuLID read the same manifest, so a given cell
  starts from the same initial latent in both. The comparison is paired at the
  level of the noise, which removes a large variance component.
- **Across styles** -- the key uses the *base* prompt, so the photoreal control
  and its stylised siblings share a latent. The within-cell difference
  Delta_style therefore isolates the style effect instead of mixing it with noise.

`derive_seed` in `src/manifest.py` must never be modified once generation has
started; doing so silently invalidates every image already produced.

## Reproducing this benchmark

```bash
python tools/build_benchmark.py --check-clip-tokens
python tools/build_style_grid.py --n-base 36
python src/manifest.py --prompts benchmark/prompts_200.json  --out benchmark/manifest_repro.parquet
python src/manifest.py --prompts benchmark/prompts_style.json --out benchmark/manifest_style.parquet --part2-only
```

Outputs are deterministic. `prompts_raw.jsonl` is the only hand-edited file.
