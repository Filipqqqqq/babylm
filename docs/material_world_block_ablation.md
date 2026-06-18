# Material-World Block Ablation Notes

## Current Result

Initial GPT-2 8l512 BabyLM 10M block-replacement sweep, evaluated on `ewok_fast`
final checkpoints only:

| condition | EWoK average | material-properties | material-dynamics |
| --- | ---: | ---: | ---: |
| baseline | 51.06 | 49.75 | 49.75 |
| dyn0p5 | 49.73 | 49.00 | 50.75 |
| dynall0p85 | 50.94 | 50.25 | 52.50 |

Interpretation: the overall EWoK score did not move up, and material-properties
is basically flat. Material-dynamics has a small increasing pattern from baseline
to 0.5% to all available dynamics data, but the two-seed estimate is noisy. Treat
this as a promising follow-up signal, not as a confirmed gain.

The current dynamics corpus had full material-dynamics concept coverage but only
partial lexical-anchor coverage. Missing anchor terms included:

```text
banana, coke, feather, helmet, jello, lemonade, methane, nitrogen,
pepper, play-doh, propane, sesame, slipper, twill, vapor, velvet
```

## Next Step

Generate a dynamics top-up that hard-assigns lexical anchors, then mix it with
the previous dynamics set rather than replacing the previous set. The old set
appears to provide useful broad dynamics coverage; the new set should fill the
rare-anchor gap without spending credits on a much larger prompt-guided run.

Suggested target:

- old dynamics set plus new hard-anchor top-up
- 500 new examples first
- 1.0% replacement first
- two seeds first, then expand if the material-dynamics trend persists

## Generate Hard-Anchor Top-Up

The v2 config currently generates 500 examples. Every row receives a
round-robin `target_anchor`, and about 70% of rows also receive a
`target_dynamic_concept`.

```bash
cd /home/jorge/babylm-baby-steps

conda run --no-capture-output -n babylm \
  python -m babysteps.cli.generate_material_world_fireworks \
  --config configs/material_world/fireworks_probe_ewok_dyn_anchor_v2.json \
  --output-dir data/material_world/fireworks_probe_ewok_dyn_anchor_v2
```

If the provider fails after writing complete intermediate stages, resume with:

```bash
conda run --no-capture-output -n babylm \
  python -m babysteps.cli.generate_material_world_fireworks \
  --config configs/material_world/fireworks_probe_ewok_dyn_anchor_v2.json \
  --resume-existing \
  --output-dir data/material_world/fireworks_probe_ewok_dyn_anchor_v2
```

## Audit Anchor Coverage

```bash
conda run --no-capture-output -n babylm \
  python scripts/material_world/audit_ewok_material_coverage.py \
  --generated data/material_world/fireworks_probe_ewok_dyn_anchor_v2/examples_reaudited.jsonl \
  --report-json data/material_world/fireworks_probe_ewok_dyn_anchor_v2/ewok_anchor_coverage_report.json
```

The important checks are `material-dynamics` anchor coverage and the generated
`coverage_report.json` fields:

- `target_anchors`: the assigned anchor counts
- `target_anchor_text_hits`: assigned anchors found in final text
- `target_anchor_text_misses`: assigned anchors not found in final text

If anchors are still missing, generate a smaller top-up specifically for those
remaining terms before launching another training sweep.

## Anchor-v2 Generation Note

The first 500-row hard-anchor run was mechanically successful: anchors were
round-robin assigned and nearly all assigned anchor strings appeared in final
text. The main weakness was semantic, not mechanical. Some anchors appeared as
side props, absences, or loose comparisons instead of participating in the
material event itself. Examples to avoid:

- "methane is not present in this scenario"
- "a banana lay nearby"
- "the sand sits near a helmet"

Future anchor runs should require `target_anchor` to be a central participant in
the event. The anchor can be the object, material, contents, substance, fuel,
liquid, fabric, food, gas, or surface involved in the visible behavior. Good
patterns are closer to "twill wrinkles as it is folded", "lemonade splashes from
the cup", "propane burns under the pan", or "vapor rises from the warm bottle".

## Build Combined Mixes

The block mixer can take multiple synthetic files, so the combined v2 mix can use
both the old dynamics corpus and the new hard-anchor top-up directly:

```bash
cd /home/jorge/babylm-baby-steps

BASE_CHUNKS=/SSD2/babylm-baby-steps-cache/token_chunks/babylm10m_byte_bpe_16384_block512_eos/train.pkl
TOKENIZER=/SSD2/babylm-baby-steps-cache/tokenizers/babylm10m_byte_bpe_16384
OLD_DYN=data/material_world/fireworks_probe_ewok_softmix_1000_dyn/train_audited.txt
ANCHOR_DYN=data/material_world/fireworks_probe_ewok_dyn_anchor_v2/train_audited.txt
MIX_ROOT=/SSD2/babylm-baby-steps-cache/token_chunks

# 1.0%: floor(16,246,784 * 0.01) requested tokens -> 317 full 512-token blocks.
conda run --no-capture-output -n babylm \
  python scripts/data/mix_token_blocks.py \
  --config configs/data/material_world_token_mix_10m.yaml \
  --source-train-chunks "$BASE_CHUNKS" \
  --tokenizer "$TOKENIZER" \
  --requested-tokens 162467 \
  --seed 1002 \
  --synthetic-file "$OLD_DYN" \
  --synthetic-file "$ANCHOR_DYN" \
  --output-dir "$MIX_ROOT/mw_dyn_anchor_v2_k162467_mixseed1002"

# 1.5% may require a larger or second top-up, depending on final token count.
```

The generated `manifest.json` records requested tokens, effective full-block
tokens, source files, and replacement indices.
