# Scripts

Preferred entrypoints live in small topical folders:

```text
scripts/data/                                      dataset preparation and tokenization
scripts/train/                                     training entrypoints
scripts/eval/                                      evaluation launchers and wrappers
scripts/sweeps/                                    training sweep launchers
scripts/analysis/                                  plotting and result summarization
scripts/hf/                                        Hugging Face download/upload helpers
scripts/synthetic_data_generation/material_world/  material-world synthetic generation, audit, and review utilities
scripts/material_world/                            older material-world compatibility entrypoints
```

The root-level Python scripts are compatibility shims for older commands. New
docs and new automation should prefer the organized paths above.
