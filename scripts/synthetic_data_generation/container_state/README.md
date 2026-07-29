# Container-State Pretraining Experiment

This experiment tests whether a small amount of naturalized synthetic text can
help a BabyLM-scale model maintain and retrieve entity state. The initial dose
comparison replaces either 1% or 2% of the 10-million-word BabyLM strict-small
corpus with deterministic container-state documents. It does not add words
beyond the 10-million-word budget.

The first experiment is deliberately small:

- each world has three to five containers;
- each container initially has one to three objects;
- a training document has zero or one operation;
- an operation adds, removes, or moves one object; and
- the final sentence recaps the contents of one target container.

The generated prose uses ordinary storage narratives, inventories, cleanup
summaries, and conversations. Training does not use the public Entity Tracking
surface form, its `Box N` identifiers, or its published object inventory.

## Research Question

The primary comparison is:

| Run | BabyLM text | Container-state text | Total words |
| --- | ---: | ---: | ---: |
| Baseline | 10,000,000 | 0 | 10,000,000 |
| 1% treatment | 9,900,000 | 100,000 | 10,000,000 |
| 2% treatment | 9,800,000 | 200,000 | 10,000,000 |

The model, tokenizer, objective schedule, optimizer, number of epochs, and seed
stay fixed. This makes the pretraining text the intended experimental variable.

The treatment asks whether the synthetic examples improve:

1. retrieval when no state change occurs;
2. retrieval after one operation;
3. transfer to unseen container and object vocabulary;
4. transfer to held-out prose templates; and
5. extrapolation from zero or one operation to two operations.

## Example Training Documents

The model receives plain text, one document per line. It does not receive the
symbolic state, operation labels, validation options, or other JSON metadata.

Zero-operation example:

```text
Two friends checked a group of labeled storage places. Mina read the storage list aloud: the Willow tray had a cork and a bead, the Alder tray had a pencil and a spool, and the Cedar tray had a shoelace and an eraser. Theo's final report said that the Willow tray held a bead and a cork.
```

One-operation example with an untouched target:

```text
Before cleanup, the Alder satchel started with a button and a fork, the Oak drawer started with a bead and a washer, the Elm bin started with a clip, and the Cedar cabinet started with a marker, a ribbon, and a pin. Next, the clip was taken out of the Elm bin. At the final check, the items in the Cedar cabinet were a ribbon, a pin, and a marker.
```

## Design and Leakage Controls

The generator first samples a symbolic world, applies checked state
transitions, and only then renders the result as text. The audit replays every
operation and verifies that the final recap agrees with the symbolic final
state.

The v1 lexicons are split into disjoint training, validation, and test lists.
The generator also checks that:

- generated text does not use `Box N` benchmark identifiers;
- the synthetic objects do not overlap the published benchmark inventory;
- container names and object names do not overlap;
- training and held-out object lists do not overlap;
- training documents are single-line documents; and
- validation initial-world signatures do not occur in training.

The relevant files are:

```text
configs/container_state/container_state_v1_seed42.yaml
configs/container_state/lexicons_v1.yaml
src/babysteps/container_state/world.py
src/babysteps/container_state/generation.py
src/babysteps/container_state/validation.py
```

## Reproduce the Data

Run all commands from the repository root. The examples assume the project
Python environment is active.

### 1. Generate the source pool and validation data

```bash
python scripts/synthetic_data_generation/container_state/generate_container_state.py \
  --config configs/container_state/container_state_v1_seed42.yaml
```

The seed-42 reference build produces:

```text
2,122 unique training documents
130,053 source-pool words
4,000 validation items
0 audit failures
```

Generated files appear under:

```text
data/container_state/container_state_v1_pool_seed42/
  train.txt
  documents.jsonl
  audit_report.json
  manifest.json
  lexicons.yaml
  validation/entity_tracking/
    iid.jsonl
    lexical_ood.jsonl
    template_ood.jsonl
    two_operations.jsonl
```

`train.txt` is the only file used as pretraining text. `documents.jsonl`
retains the symbolic state and provenance for auditing. Files below
`validation/` are held out and are never included in the text mix.

The 2% treatment uses a separate 260k-word source pool so its 200k-word patch
also contains no repeated documents:

```bash
python scripts/synthetic_data_generation/container_state/generate_container_state.py \
  --config configs/container_state/container_state_v1_pool260k_seed42.yaml
```

The expanded pool contains 4,252 unique documents and 260,015 words. Its first
130,053 words exactly match the 1% source pool, and its four validation JSONL
files are byte-for-byte identical to the 1% validation files.

For a quick local generation check, override the pool and validation sizes:

```bash
python scripts/synthetic_data_generation/container_state/generate_container_state.py \
  --config configs/container_state/container_state_v1_seed42.yaml \
  --output-dir /tmp/container_state_smoke \
  --pool-min-words 4000 \
  --validation-items-per-slice 12
```

### 2. Build the exact 1% text mix

```bash
python scripts/data/build_text_mix.py \
  --config configs/data_mixes/babylm2026_strict_small_replace1pct_container_state_v1_seed42.yaml
```

The output is:

```text
data/text_mixes/babylm2026_strict_small_replace1pct_container_state_v1_seed42/
```

Inspect `counts.json` after building. The reference mix contains exactly
9,900,000 BabyLM words and 100,000 container-state words. It samples 1,630
unique synthetic documents without repetition and trims only the BabyLM source
when final budget adjustment is necessary.

Build the 2% dose with:

```bash
python scripts/data/build_text_mix.py \
  --config configs/data_mixes/babylm2026_strict_small_replace2pct_container_state_v1_seed42.yaml
```

This mix contains exactly 9,800,000 BabyLM words and 200,000 container-state
words. It samples 3,276 unique synthetic documents without repetition.

## Tokenizer and Token Shards

The primary experiment deliberately reuses the existing GPT-BERT baseline-mixed
tokenizer:

```text
${BABYSTEPS_GPT_BERT_BASELINE_MIXED_TOKENIZER_DIR:-${BABYSTEPS_CACHE_DIR:-~/babylm-baby-steps-cache}/tokenizers/gpt_bert_baseline_mixed}
```

Some synthetic words are represented by multiple known subword tokens. That is
expected and does not modify the source text. Keeping the tokenizer fixed
prevents tokenizer changes from being confounded with the 1% text
intervention. Training a tokenizer on the mixed corpus can be tested later as a
separate ablation.

Build the treatment token shards with:

```bash
python scripts/data/build_token_shards.py \
  --config configs/tokenization/babylm2026_strict_small_replace1pct_container_state_v1_gpt_bert_mixed_shards.yaml
```

For the 2% treatment, use:

```bash
python scripts/data/build_token_shards.py \
  --config configs/tokenization/babylm2026_strict_small_replace2pct_container_state_v1_gpt_bert_mixed_shards.yaml
```

The shards are written under:

```text
${BABYSTEPS_CACHE_DIR:-~/babylm-baby-steps-cache}/token_shards/babylm2026_strict_small_replace1pct_container_state_v1_gpt_bert_mixed/
```

Token shards are generated cache artifacts and should not be committed.

## GPT-BERT Mixed Training

Here, **mixed** means GPT-BERT alternates causal and masked steps with a 1:1
schedule. It is distinct from the repository's causal-focus configuration,
which uses a 15:1 schedule.

Train the 1% treatment:

```bash
python scripts/train/train_hybrid_lm.py \
  --config configs/train/gpt_bert_mixed_babylm2026_strict_small_replace1pct_container_state_v1_muon.yaml
```

Train the matched 2% treatment:

```bash
python scripts/train/train_hybrid_lm.py \
  --config configs/train/gpt_bert_mixed_babylm2026_strict_small_replace2pct_container_state_v1_muon.yaml
```

The matched baseline config is:

```text
configs/train/gpt_bert_baseline_mixed_babylm2026_strict_small_muon.yaml
```

The baseline, 1%, and 2% configs use the same GPT-BERT mixed model, 1:1
objective schedule, Muon optimizer, peak Muon learning rate 0.01, ten epochs,
sequence length 512, batch size 32, bf16 precision, and seed 42. Tests assert
that their model, hybrid, training, logging, and checkpoint settings remain
equal.

Training runs are written below:

```text
${BABYSTEPS_RUNS_DIR:-~/babylm-baby-steps-runs}/training/
```

The training configuration runs the repository's standard BabyLM likelihood
diagnostics. The custom validation data described below is evaluated separately
so that it remains an explicit experiment-specific diagnostic.

## Custom Validation

Each validation item contains one prefix and five possible continuations. The
model does not see all five choices together. Evaluation scores each
prefix-continuation pair independently and checks whether the correct
continuation receives the highest likelihood.

The JSONL stores the correct continuation at `options[0]`; that position is
used by the evaluator and is not exposed as an answer cue to the model. Random
five-way accuracy is 20%.

| Slice | Items | Operations | What is held out |
| --- | ---: | ---: | --- |
| `iid` | 1,000 | 0 or 1 | Initial world only |
| `lexical_ood` | 1,000 | 0 or 1 | Containers, labels, and objects |
| `template_ood` | 1,000 | 0 or 1 | Rendering templates |
| `two_operations` | 1,000 | 2 | Longer state-transition sequence |

The first three slices contain 500 zero-operation and 500 one-operation items.
The two-operation slice contains 1,000 two-operation items.

After training, evaluate a final GPT-BERT checkpoint with:

```bash
python scripts/eval/eval_mntp_likelihood_hf.py \
  --model /path/to/training-run/final \
  --tasks entity_tracking \
  --eval-root data/container_state/container_state_v1_pool_seed42/validation \
  --score-reduction both \
  --trust-remote-code
```

Because the distractors can have different numbers of tokens, treat the mean
token score as the primary custom-validation result and the summed score as a
secondary diagnostic. Report accuracy and mean correct-versus-best-distractor
margin for every slice rather than only pooling all 4,000 items.

Run the same command for the matched baseline, 1%, and 2% checkpoints. Also
compare the standard BabyLM evaluation suite to check whether a targeted gain
comes with broader regressions.

## Interpreting the First Run

The most informative initial outcome is the pattern across slices:

- an `iid` gain alone may indicate template or vocabulary memorization;
- `iid` plus `lexical_ood` gains suggest more general state tracking;
- a `template_ood` gain suggests the behavior survives surface-form changes;
- a `two_operations` gain tests limited length extrapolation; and
- unchanged standard benchmarks suggest the 1% replacement is targeted rather
  than broadly disruptive.

The 1%-versus-2% comparison adds a simple dose-response check: a real
curriculum effect should ideally strengthen or remain stable at 2%, while a
2%-only regression may indicate that the replacement is displacing too much
natural text. One seed is suitable for validating the pipeline and estimating
the size of the effect. A promising effect should subsequently be replicated
across multiple training seeds.
