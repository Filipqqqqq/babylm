#!/usr/bin/env python3
"""Evaluate a Hugging Face MNTP/MaskedLM model on BabyLM likelihood tasks."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from babysteps.evaluation import blimp_likelihood  # noqa: E402
from babysteps.evaluation import comps_likelihood  # noqa: E402
from babysteps.evaluation import entity_tracking_likelihood  # noqa: E402
from babysteps.evaluation import mntp_likelihood  # noqa: E402
from babysteps.paths import resolve_path  # noqa: E402


TASKS = ("blimp", "blimp_supplement", "entity_tracking", "comps")


def default_eval_root() -> Path:
    return ROOT / "external" / "babylm-eval" / "strict" / "evaluation_data" / "full_eval"


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def parse_reductions(raw_value: str) -> tuple[str, ...]:
    value = raw_value.strip().lower()
    if value == "both":
        return ("sum", "mean")
    if value in {"sum", "mean"}:
        return (value,)
    raise ValueError("--score-reduction must be sum, mean, or both.")


def load_dtype(value: str) -> Any:
    if value == "auto":
        return "auto"
    if value == "float32":
        return torch.float32
    if value == "float16":
        return torch.float16
    if value == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {value}")


def load_tokenizer(args: argparse.Namespace) -> Any:
    kwargs: dict[str, Any] = {"trust_remote_code": args.trust_remote_code}
    if args.revision:
        kwargs["revision"] = args.revision
    if args.local_files_only:
        kwargs["local_files_only"] = True
    return AutoTokenizer.from_pretrained(args.model, **kwargs)


def set_padding_token(tokenizer: Any) -> None:
    if getattr(tokenizer, "pad_token_id", None) is not None:
        return
    for attr in ("cls_token", "eos_token", "bos_token", "mask_token"):
        value = getattr(tokenizer, attr, None)
        if value is not None:
            tokenizer.pad_token = value
            return


def task_data_dir(eval_root: Path, task: str) -> Path:
    if task == "blimp":
        return eval_root / "blimp_filtered"
    if task == "blimp_supplement":
        return eval_root / "supplement_filtered"
    if task == "entity_tracking":
        return eval_root / "entity_tracking"
    if task == "comps":
        return eval_root / "comps"
    raise ValueError(f"Unsupported task: {task}")


def load_task_items(eval_root: Path, task: str, max_items: int | None) -> list[Any]:
    data_dir = task_data_dir(eval_root, task)
    if task in {"blimp", "blimp_supplement"}:
        items = blimp_likelihood.read_blimp_items(data_dir)
    elif task == "entity_tracking":
        items = entity_tracking_likelihood.read_entity_tracking_items(data_dir)
    elif task == "comps":
        items = comps_likelihood.read_comps_items(data_dir)
    else:
        raise ValueError(f"Unsupported task: {task}")
    return items[:max_items] if max_items is not None else items


def evaluate_task(
    *,
    task: str,
    model: torch.nn.Module,
    tokenizer: Any,
    items: list[Any],
    reductions: tuple[str, ...],
    batch_size: int,
    device: torch.device,
) -> list[tuple[str, mntp_likelihood.MntpBenchmarkResult]]:
    results: list[tuple[str, mntp_likelihood.MntpBenchmarkResult]] = []
    if task in {"blimp", "blimp_supplement"}:
        good_scores = mntp_likelihood.score_mntp_sentences(
            model=model,
            tokenizer=tokenizer,
            sentences=[item.sentence_good for item in items],
            batch_size=batch_size,
            device=device,
        )
        bad_scores = mntp_likelihood.score_mntp_sentences(
            model=model,
            tokenizer=tokenizer,
            sentences=[item.sentence_bad for item in items],
            batch_size=batch_size,
            device=device,
        )
    elif task == "entity_tracking":
        entity_score_map = mntp_likelihood.score_mntp_entity_tracking_continuations(
            model=model,
            tokenizer=tokenizer,
            requests=entity_tracking_likelihood.make_continuation_requests(items),
            batch_size=batch_size,
            device=device,
        )
    elif task == "comps":
        comps_scores = mntp_likelihood.score_mntp_comps_continuations(
            model=model,
            tokenizer=tokenizer,
            requests=comps_likelihood.make_continuation_requests(items),
            batch_size=batch_size,
            device=device,
        )
    else:
        raise ValueError(f"Unsupported task: {task}")

    for reduction in reductions:
        if task in {"blimp", "blimp_supplement"}:
            records = blimp_likelihood.build_item_records(
                items=items,
                good_scores=good_scores,  # type: ignore[possibly-undefined,arg-type]
                bad_scores=bad_scores,  # type: ignore[possibly-undefined,arg-type]
                score_reduction=reduction,  # type: ignore[arg-type]
            )
            result = mntp_likelihood.MntpBenchmarkResult(
                item_records=records,
                summary_rows=blimp_likelihood.aggregate_item_records(records),
            )
        elif task == "entity_tracking":
            records = entity_tracking_likelihood.build_item_records(
                items=items,
                score_map=entity_score_map,  # type: ignore[possibly-undefined,arg-type]
                score_reduction=reduction,  # type: ignore[arg-type]
            )
            result = mntp_likelihood.MntpBenchmarkResult(
                item_records=records,
                summary_rows=entity_tracking_likelihood.aggregate_item_records(records),
            )
        elif task == "comps":
            records = comps_likelihood.build_item_records(
                items=items,
                scores=comps_scores,  # type: ignore[possibly-undefined,arg-type]
                score_reduction=reduction,  # type: ignore[arg-type]
            )
            result = mntp_likelihood.MntpBenchmarkResult(
                item_records=records,
                summary_rows=comps_likelihood.aggregate_item_records(records),
            )
        else:
            raise ValueError(f"Unsupported task: {task}")
        results.append((reduction, result))
    return results


def overall_row(rows: list[dict[str, Any]], scope: str = "overall") -> dict[str, Any]:
    return next(row for row in rows if row["group_scope"] == scope and row["group_name"] == "all")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face model path or model id.")
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=TASKS,
        default=list(TASKS),
    )
    parser.add_argument(
        "--eval-root",
        default=str(default_eval_root()),
        help="BabyLM full_eval or fast_eval root directory.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-items", type=int, help="Optional quick-test item cap per task.")
    parser.add_argument("--score-reduction", choices=("sum", "mean", "both"), default="both")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device. With CUDA_VISIBLE_DEVICES=1, use cuda.",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--revision", help="Optional Hugging Face revision.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "runs" / "analysis" / "mntp_likelihood_hf"),
    )
    parser.add_argument("--no-items", action="store_true", help="Only write the summary CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eval_root = resolve_path(args.eval_root, base_dir=ROOT)
    output_dir = resolve_path(args.output_dir, base_dir=ROOT) / slugify(args.model)
    summary_path = output_dir / "likelihood_summary.csv"
    items_path = output_dir / "likelihood_items.jsonl"
    if summary_path.exists():
        summary_path.unlink()
    if items_path.exists():
        items_path.unlink()

    model_kwargs: dict[str, Any] = {
        "dtype": load_dtype(args.dtype),
        "trust_remote_code": args.trust_remote_code,
    }
    if args.revision:
        model_kwargs["revision"] = args.revision
    if args.local_files_only:
        model_kwargs["local_files_only"] = True

    tokenizer = load_tokenizer(args)
    set_padding_token(tokenizer)
    device = torch.device(args.device)
    model = AutoModelForMaskedLM.from_pretrained(args.model, **model_kwargs).to(device)
    model.eval()

    reductions = parse_reductions(args.score_reduction)
    metadata = {
        "run_name": args.model,
        "epoch": "",
        "step": "",
        "tokens_seen": "",
        "words_seen": "",
        "checkpoint": args.revision or "",
        "checkpoint_path": args.model,
    }

    print(f"Model: {args.model}")
    print(f"Eval root: {eval_root}")
    print(f"Tasks: {', '.join(args.tasks)}")
    print(f"Reductions: {', '.join(reductions)}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print()
    print("task             reduction  macro_score  micro_score  correct/items  ties")

    for task in args.tasks:
        items = load_task_items(eval_root, task, args.max_items)
        for reduction, result in evaluate_task(
            task=task,
            model=model,
            tokenizer=tokenizer,
            items=items,
            reductions=reductions,
            batch_size=args.batch_size,
            device=device,
        ):
            mntp_likelihood.append_summary_csv(
                result.summary_rows,
                summary_path,
                benchmark=task,
                metadata=metadata,
            )
            if not args.no_items:
                mntp_likelihood.append_item_records_jsonl(
                    result.item_records,
                    items_path,
                    benchmark=task,
                    metadata=metadata,
                )

            overall = overall_row(result.summary_rows, "overall")
            overall_micro = overall_row(result.summary_rows, "overall_micro")
            print(
                f"{task:<16} "
                f"{reduction:<9} "
                f"{overall['score']:.4f}       "
                f"{overall_micro['score']:.4f}       "
                f"{overall_micro['correct']}/{overall_micro['items']}       "
                f"{overall_micro['ties']}"
            )

    print(f"\nWrote summary: {summary_path}")
    if not args.no_items:
        print(f"Wrote items: {items_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
