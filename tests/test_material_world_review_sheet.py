import csv

from babysteps.cli.export_material_world_review_sheet import review_rows, write_csv


def test_review_rows_flattens_audited_material_world_rows(tmp_path):
    rows = [
        {
            "id": "mw-001",
            "audit_status": "needs_review",
            "metadata": {
                "material": "rubber",
                "material_class": "elastic_materials",
                "object": "ball",
                "dynamics": ["bounce", "squeeze"],
                "style_id": "child_science_explanation",
            },
            "audit_flags": [
                {"code": "object_material_mismatch"},
                {"code": "too_benchmarky"},
                {"note": "ignored because it has no code"},
            ],
            "text": "A rubber ball can bounce.\nIt can also squish a little.",
        }
    ]

    review = review_rows(rows)

    assert review == [
        {
            "id": "mw-001",
            "audit_status": "needs_review",
            "material": "rubber",
            "material_class": "elastic_materials",
            "object": "ball",
            "dynamics": "bounce; squeeze",
            "style_id": "child_science_explanation",
            "flag_codes": "object_material_mismatch; too_benchmarky",
            "text": "A rubber ball can bounce. It can also squish a little.",
        }
    ]

    output_csv = tmp_path / "review.csv"
    write_csv(output_csv, review)

    with output_csv.open(encoding="utf-8", newline="") as f:
        csv_rows = list(csv.DictReader(f))

    assert csv_rows == review
