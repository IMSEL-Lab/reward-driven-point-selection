from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from sam_rl.result_manifest import validate_result_shards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize method-level CSV results into mean metrics by (method, N)."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="CSV files with method,N,image_id,seed,iou,dice,boundary_iou,wallclock_ms.",
    )
    parser.add_argument("--output", required=True, help="Output summary CSV path.")
    parser.add_argument(
        "--expected-manifest", help="JSON with expected_episode_keys, methods, budgets, and seeds."
    )
    parser.add_argument(
        "--allow-legacy",
        action="store_true",
        help="Explicitly permit metadata-free historical results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.expected_manifest and not args.allow_legacy:
        raise ValueError(
            "Strict reduction requires --expected-manifest; archived results require --allow-legacy."
        )
    if args.expected_manifest:
        expected = json.loads(Path(args.expected_manifest).read_text(encoding="utf-8"))
        rows = validate_result_shards(
            args.inputs,
            expected_keys=expected["expected_episode_keys"],
            expected_methods=expected["methods"],
            budgets=expected["budgets"],
            seeds=expected.get("seeds", []),
            training_seeds=expected.get("training_seeds"),
            eval_seeds=expected.get("eval_seeds"),
            metadata_paths=expected.get("metadata_paths"),
        )
    else:
        rows = _load_legacy_rows(args.inputs)
    aggregates: dict[tuple[str, int], dict[str, float]] = defaultdict(
        lambda: {
            "num_rows": 0.0,
            "sum_iou": 0.0,
            "sum_dice": 0.0,
            "sum_boundary_iou": 0.0,
            "sum_wallclock_ms": 0.0,
        }
    )
    for row in rows:
        key = (str(row["method"]), int(row["N"]))
        bucket = aggregates[key]
        bucket["num_rows"] += 1.0
        bucket["sum_iou"] += float(row["iou"])
        bucket["sum_dice"] += float(row["dice"])
        bucket["sum_boundary_iou"] += float(row["boundary_iou"])
        bucket["sum_wallclock_ms"] += float(row["wallclock_ms"])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "N",
                "num_rows",
                "mean_iou",
                "mean_dice",
                "mean_boundary_iou",
                "mean_wallclock_ms",
            ],
        )
        writer.writeheader()
        for method, n_points in sorted(aggregates.keys(), key=lambda item: (item[0], item[1])):
            bucket = aggregates[(method, n_points)]
            denom = max(bucket["num_rows"], 1.0)
            writer.writerow(
                {
                    "method": method,
                    "N": int(n_points),
                    "num_rows": int(bucket["num_rows"]),
                    "mean_iou": bucket["sum_iou"] / denom,
                    "mean_dice": bucket["sum_dice"] / denom,
                    "mean_boundary_iou": bucket["sum_boundary_iou"] / denom,
                    "mean_wallclock_ms": bucket["sum_wallclock_ms"] / denom,
                }
            )
    print(output_path)


def _load_legacy_rows(paths):
    rows = []
    seen = set()
    for input_path in [Path(path) for path in paths]:
        with input_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                for metric in ("iou", "dice", "boundary_iou", "wallclock_ms"):
                    if not math.isfinite(float(row[metric])):
                        raise ValueError(f"Nonfinite {metric} in legacy result {input_path}.")
                identity = (row["method"], row["N"], row["image_id"], row["seed"])
                if identity in seen:
                    raise ValueError(f"Duplicate legacy result identity {identity}.")
                seen.add(identity)
                rows.append(row)
    return rows


if __name__ == "__main__":
    main()
