# GPT-BERT linguistic-repair dose response v1

This experiment compares nested 0%, 1%, and 1.5% linguistic-repair token
interventions while holding the canonical BabyLM validation shards, untreated
training tokens, token count, loader block count, and training configuration
fixed.

The 1% condition replaces 10 of the canonical 1,038 GPT-BERT loader blocks.
The 1.5% condition replaces those same 10 blocks plus 6 more. Replacement
blocks are ranked once with seed `20260714`. Synthetic components retain the
relative allocation of the original 1.5%-word repair recipe.

Pretraining uses seeds 42, 43, and 44. Each final checkpoint is fine-tuned on
the seven repository SuperGLUE tasks with learning rates `3e-5` and `8e-5` and
fine-tuning seed 42.

Large datasets, checkpoints, logs, fine-tuning results, summaries, and figures
are stored under:

```text
~/babylm-baby-steps-runs/experiments/
└── gpt_bert_linguistic_repair_dose_response_v1/
```

Build or validate the paired datasets with
`scripts/experiments/build_linguistic_repair_dose_response.py`. The tmux
launcher is `scripts/sweeps/launch_gptbert_linguistic_repair_dose_response_tmux.sh`.
After the 18 fine-tuning arms complete, run
`scripts/analysis/summarize_linguistic_repair_dose_response.py`.
