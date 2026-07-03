#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

CONFIG="${CONFIG:-configs/concept_curriculum/physical_dynamics/physical_dynamics_ewok_micro_500.json}"
OUTPUT_DIR="${OUTPUT_DIR:-data/concept_curriculum/physical_dynamics_ewok_micro_1000_slip_bridge_qwen}"
API_KEY_FILE="${API_KEY_FILE:-$ROOT/fireworks_api.txt}"
DOC_COUNT="${DOC_COUNT:-1000}"
CONDA_ENV="${CONDA_ENV:-babylm}"

PYTHON_CMD=(python -u)
if command -v conda >/dev/null 2>&1; then
  PYTHON_CMD=(conda run --no-capture-output -n "$CONDA_ENV" python -u)
fi

echo "=== physical dynamics EWoK micro curriculum ==="
echo "config:     $CONFIG"
echo "output_dir: $OUTPUT_DIR"
echo "conda_env:  $CONDA_ENV"
echo

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
plans_path = output_dir / "scenario_plans.jsonl"
report_path = output_dir / "audit_report.json"
train_path = output_dir / "train.txt"

missing = [path for path in (docs_path, plans_path, report_path) if not path.exists()]
if missing:
    raise SystemExit("Missing expected output file(s): " + ", ".join(str(path) for path in missing))

docs = [
    json.loads(line)
    for line in docs_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
plans = [
    json.loads(line)
    for line in plans_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
report = json.loads(report_path.read_text(encoding="utf-8"))

def metadata(row: dict) -> dict:
    value = row.get("metadata", {})
    return value if isinstance(value, dict) else {}

print()
print("=== audit summary ===")
print(f"output_dir: {output_dir}")
print(f"documents_audited: {len(docs)}")
print(f"train_txt: {train_path}")
print(f"counts: {report.get('counts', {})}")
print(f"flag_counts: {report.get('flag_counts', {})}")

print()
print("=== planned balance ===")
print("canonical_concept_id:", dict(Counter(plan.get("canonical_concept_id") for plan in plans)))
print("family:", dict(Counter(plan.get("family") for plan in plans)))
print("style_id:", dict(Counter(plan.get("style_id") for plan in plans)))
print("document_type:", dict(Counter(plan.get("document_type") for plan in plans)))

print()
print("=== audited rows ===")
print("status:", dict(Counter(row.get("audit_status") for row in docs)))
print("canonical_concept_id:", dict(Counter(metadata(row).get("canonical_concept_id") for row in docs)))
print("family:", dict(Counter(metadata(row).get("family") for row in docs)))
print("style_id:", dict(Counter(metadata(row).get("style_id") for row in docs)))

bridge_flags = {
    "missing_same_sentence_bridge_evidence",
    "bridge_contrast_evidence_violation",
    "missing_lexical_target",
    "metadata_only_contrast_mentioned",
    "banned_term",
}
rows_to_inspect = []
for row in docs:
    flag_codes = {str(flag.get("code")) for flag in row.get("audit_flags", [])}
    if row.get("audit_status") != "pass" or flag_codes & bridge_flags:
        rows_to_inspect.append(row)

if rows_to_inspect:
    print()
    print("=== rows to inspect ===")
    for row in rows_to_inspect[:25]:
        meta = metadata(row)
        flags = ",".join(str(flag.get("code", "")) for flag in row.get("audit_flags", []))
        text = " ".join(str(row.get("text", "")).split())
        print(
            f"{row.get('id')} "
            f"status={row.get('audit_status')} "
            f"concept={meta.get('canonical_concept_id')} "
            f"family={meta.get('family')} "
            f"style={meta.get('style_id')} "
            f"lexical_target={meta.get('lexical_target')} "
            f"flags={flags} "
            f"text={text[:220]}"
        )
    if len(rows_to_inspect) > 25:
        print(f"... {len(rows_to_inspect) - 25} more rows omitted")
else:
    print()
    print("No failing/warning bridge rows to inspect.")
PY
