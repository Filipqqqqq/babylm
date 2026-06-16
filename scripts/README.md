# Scripts

Preferred entrypoints live in small topical folders:

```text
scripts/data/            dataset preparation and tokenization
scripts/train/           training entrypoints
scripts/material_world/  material-world generation/audit utilities
```

The root-level Python scripts are compatibility shims for older commands. New
docs and new automation should prefer the organized paths above.

The root-level shell sweep/evaluation scripts are still in place for now. Move
them in a separate pass after checking their relative path assumptions.
