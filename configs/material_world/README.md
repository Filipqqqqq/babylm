# Material World Catalogues

These files are the source catalogues for BabyLM material-world synthetic text generation.
Generated outputs under `data/material_world/` are snapshots; edit these config files first, then rerun the generator when you want new prose.

## Files

- `material_classes.json`
  Defines behavior-focused material classes. Each class has example materials, visible properties, dynamics, caveats, and scenario hooks.
- `material_objects.json`
  Defines everyday objects that can be paired with multiple materials. Candidate materials include common choices plus useful counterfactuals.
- `coverage_sets.json`
  Defines optional named coverage targets. These are not used by default. They let us ask for a focused probe or a soft upweighting mix, such as EWoK material-properties or material-dynamics coverage, while keeping ordinary material-world generation broad.
- `fireworks_probe.json`
  Default Fireworks/Qwen generation config. It points to the catalogues but does not activate any coverage set unless `coverage_set_ids` is added.

## Default Behavior

The default probe is intentionally general-purpose text generation. It uses the first few material classes and objects according to:

```json
{
  "material_class_limit": 4,
  "object_limit": 3,
  "style_limit": 4
}
```

This keeps quick smoke runs focused on ordinary prose about common material behavior. Adding EWoK coverage metadata does not change default generation unless a config explicitly requests it.

## EWoK Coverage Sets

`coverage_sets.json` currently has two EWoK-inspired sets.

`ewok_material_properties` tracks these property contrasts:

- bouncy / not bouncy
- cold / warm
- fragile / sturdy
- heavy / light
- smooth / rough
- soft / hard
- transparent / opaque

It also records the EWoK-like property objects:

- `balloon`
- `bin`
- `football`
- `volleyball`
- `wheel`

`ewok_material_dynamics` tracks action and behavior concepts from EWoK fast, including:

- squeeze, stir, hang, wrinkle, tap, pour, drape
- break, drip, rip, ripple, pile, fold, splash, flap

The dynamics set exists so concepts like `stir`, `tap`, `flap`, and `ripple` can be explicit row-level targets instead of only implicit prompt guidance.

## Focused Coverage Runs

To run a targeted probe, use a config override like:

```json
{
  "coverage_set_ids": ["ewok_material_properties"],
  "scenario_limit": 40,
  "output_dir": "data/material_world/ewok_material_properties_probe"
}
```

Or pass it at the CLI:

```bash
python -m babysteps.cli.generate_material_world_fireworks \
  --coverage-set ewok_material_properties \
  --scenario-limit 40 \
  --output-dir data/material_world/ewok_material_properties_probe
```

When `coverage_set_ids` is present by itself, the pipeline expands the coverage set into the relevant material classes and object cards. This is the focused mode: later EWoK-only object cards such as `balloon`, `football`, and `wheel` become the selected object pool for that run. The prompt also receives coverage guidance, but generated training text should still be natural prose. It should not mention EWoK, benchmarks, coverage sets, concept labels, or test names.

## Quota-Backed Soft EWoK Upweighting

For ordinary text generation with more EWoK-inspired examples, use `coverage_mix` instead of relying only on `coverage_set_ids`.

```json
{
  "coverage_mix": {
    "ewok_material_properties": {
      "object_fraction": 0.25,
      "material_fraction": 0.25
    },
    "ewok_material_dynamics": {
      "dynamic_fraction": 0.10
    }
  },
  "selection_seed": "mw-softmix-001",
  "scenario_limit": 500,
  "output_dir": "data/material_world/fireworks_probe_ewok_softmix"
}
```

In soft-mix mode, the selected catalogue expands to the full material-class and object catalogues unless you explicitly pass `material_class_ids` or `object_ids`. The EWoK set is used as an upweighting target, not as the whole pool. The pipeline converts these settings into deterministic row-level assignments:

- `object_fraction` assigns that share of rows to use an object from the coverage set.
- `material_fraction` assigns that share of rows to use a material from the coverage set.
- `dynamic_fraction` assigns that share of rows to include a dynamic concept from the coverage set.
- `row_fraction` assigns that share of rows to the coverage set's concepts more generally.

These fractions are independent quota slices. For example, `object_fraction: 0.25` and `material_fraction: 0.25` over 500 scenarios assigns about 125 object-target rows and about 125 material-target rows. If you want about 35% of rows to have one general EWoK target, use `row_fraction: 0.35` instead of stacking multiple role-specific fractions.

The generator writes the assignment into each scenario and example metadata:

- `coverage_set_id`
- `coverage_domain`
- `coverage_role`
- `target_property_concept`
- `target_dynamic_concept`
- `target_object_id`
- `target_material`
- `target_concept_pair`

The remaining unassigned examples stay broad and ordinary, spread across diverse non-EWoK material classes and objects. After generation, `coverage_report.json` records the requested mix, assigned quotas, and observed coverage in checked and final rows.

Set `selection_seed` when you want a semi-deterministic run. The seed deterministically shuffles the selected material classes, objects, and discourse styles before prompting, and it is written to `manifest.json`. It makes the prompt inputs reproducible, but the model can still vary unless generation temperatures are low and the provider behaves deterministically.

## Balanced Object Exposure

If object identity should be controlled as an experimental factor, turn on balanced object sampling:

```json
{
  "object_sampling": {
    "mode": "balanced_round_robin",
    "scope": "selected_objects"
  },
  "selection_seed": "mw-softmix-001"
}
```

This assigns a `target_object_id` to each scenario in a seeded round-robin over the selected object catalogue. Existing coverage object targets are preserved and counted first, then the remaining rows are filled to make the total object counts as even as possible. This reduces object-frequency confounds from model preferences for common objects like cups, bottles, or bowls.

Rows assigned by this layer include:

- `object_sampling_mode`
- `object_sampling_scope`
- `object_sampling_role`

Each run writes `object_sampling_report.json` with assigned object counts and observed checked/final object counts. Counts can still drift after audit drops, so use the final report when deciding whether to generate a top-up batch.

For one coverage set at a time, the same mode is available from the CLI:

```bash
python scripts/synthetic_data_generation/material_world/generate_fireworks.py \
  --coverage-set ewok_material_properties \
  --coverage-object-fraction 0.25 \
  --coverage-material-fraction 0.25 \
  --object-sampling-mode balanced_round_robin \
  --selection-seed mw-softmix-001 \
  --scenario-limit 500 \
  --scenario-batch-size 10 \
  --render-batch-size 10 \
  --check-batch-size 10 \
  --output-dir data/material_world/fireworks_probe_ewok_softmix
```

For different fractions on multiple coverage sets, put the `coverage_mix` object in a JSON config so each set can have its own role-specific quota.

## Adding Coverage Without Narrowing Defaults

When adding a benchmark-inspired material or object:

1. Add the material to the most natural class in `material_classes.json`.
2. Add caveats when the material is unusual, rare, or context-sensitive.
3. Add object cards to `material_objects.json` only when the object is useful outside a single benchmark template.
4. Add `coverage_tags` to object cards only when they are part of a named coverage set.
5. Update `coverage_sets.json` with the exact materials, objects, and concepts needed for targeted runs.
6. Leave `fireworks_probe.json` broad unless the whole default probe should change.

Prefer visible everyday behavior over microscopic explanations. If an object-material pair is intentionally odd, the generated scenario should frame it as a pretend or hypothetical example.
