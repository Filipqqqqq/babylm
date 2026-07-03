#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

CONFIG="${CONFIG:-configs/concept_curriculum/archive/stir_pilot_20.json}"
OUTPUT_DIR="${OUTPUT_DIR:-data/concept_curriculum/archive/stir_pilot_80}"
API_KEY_FILE="${API_KEY_FILE:-$ROOT/fireworks_api.txt}"
DOC_COUNT="${DOC_COUNT:-80}"
CONDA_ENV="${CONDA_ENV:-babylm}"

PYTHON_CMD=(python -u)
if command -v conda >/dev/null 2>&1; then
  PYTHON_CMD=(conda run --no-capture-output -n "$CONDA_ENV" python -u)
fi

"${PYTHON_CMD[@]}" scripts/synthetic_data_generation/concept_curriculum/generate_fireworks.py \
  --config "$CONFIG" \
  --document-count "$DOC_COUNT" \
  --output-dir "$OUTPUT_DIR" \
  --api-key-file "$API_KEY_FILE" \
  "$@"

"${PYTHON_CMD[@]}" - "$OUTPUT_DIR" <<'PY'
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

output_dir = Path(sys.argv[1])
if not output_dir.is_absolute():
    output_dir = Path.cwd() / output_dir

docs_path = output_dir / "documents_audited.jsonl"
report_path = output_dir / "audit_report.json"

rows = [json.loads(line) for line in docs_path.read_text(encoding="utf-8").splitlines() if line.strip()]
report = json.loads(report_path.read_text(encoding="utf-8"))

print()
print("=== stir pilot audit summary ===")
print(f"output_dir: {output_dir}")
print(f"counts: {report.get('counts', {})}")
print(f"flags: {report.get('flag_counts', {})}")
print("document_type:", dict(Counter(row.get("metadata", {}).get("document_type") for row in rows)))
print("style_id:", dict(Counter(row.get("metadata", {}).get("style_id") for row in rows)))

interesting = [
    row for row in rows
    if row.get("audit_status") != "pass"
    or any(flag.get("code") in {"missing_neighbor", "weak_concept_map"} for flag in row.get("audit_flags", []))
]
if interesting:
    print()
    print("=== rows to inspect ===")
    for row in interesting:
        metadata = row.get("metadata", {})
        flags = ",".join(flag.get("code", "") for flag in row.get("audit_flags", []))
        text = " ".join(str(row.get("text", "")).split())
        print(
            f"{row.get('id')} "
            f"status={row.get('audit_status')} "
            f"type={metadata.get('document_type')} "
            f"style={metadata.get('style_id')} "
            f"neighbor={metadata.get('neighbor')} "
            f"flags={flags} "
            f"text={text[:180]}"
        )
PY
