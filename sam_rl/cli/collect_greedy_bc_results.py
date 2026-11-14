from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

from sam_rl.cli.common import load_config
from sam_rl.result_manifest import validate_result_shards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect per-budget greedy-BC runs into shared CSVs, summaries, and checkpoint manifests."
    )
    parser.add_argument("--config", default="configs/greedy_bc/fss1000_parallel_v2.yaml")
    parser.add_argument("--budgets", type=int, nargs="*", default=None)
    parser.add_argument(
        "--output-dir", default=None, help="Optional collected-results directory override."
    )
    parser.add_argument("--expected-manifest", help="Expected result identity contract JSON.")
    parser.add_argument(
        "--allow-legacy",
        action="store_true",
        help="Explicitly collect archived runs without identity metadata.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    budgets = [int(value) for value in (args.budgets or cfg.get("launcher", {}).get("budgets", []))]
    if not budgets:
        raise ValueError("No budgets configured. Set launcher.budgets or pass --budgets.")
    if not args.expected_manifest and not args.allow_legacy:
        raise ValueError(
            "Strict collection requires --expected-manifest; archived runs require --allow-legacy."
        )

    run_root = Path(cfg["paths"].get("run_root", "artifacts/greedy_bc/runs"))
    run_name = str(cfg["train"].get("run_name", "fss1000_greedy_bc"))
    output_dir = Path(args.output_dir or f"artifacts/greedy_bc/collected/{run_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir = output_dir / "summaries"
    checkpoints_dir = output_dir / "checkpoints"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    manifests: dict[str, dict[str, str]] = {
        "checkpoints_by_budget": {},
    }

    for budget in budgets:
        run_dir = run_root / f"{run_name}_N{budget}"
        best_checkpoint = run_dir / "best.pt"
        if not best_checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint for N={budget}: {best_checkpoint}")
        copied_checkpoint = checkpoints_dir / f"N{budget}_best.pt"
        shutil.copy2(best_checkpoint, copied_checkpoint)
        manifests["checkpoints_by_budget"][str(budget)] = str(copied_checkpoint)

    for split_name in ("val", "test"):
        rows = []
        source_paths = []
        for budget in budgets:
            run_dir = run_root / f"{run_name}_N{budget}"
            csv_path = run_dir / f"{split_name}_results.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"Missing {split_name} CSV for N={budget}: {csv_path}")
            source_paths.append(csv_path)
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                rows.extend(reader)
        if args.expected_manifest:
            expected = json.loads(Path(args.expected_manifest).read_text(encoding="utf-8"))
            contract = expected.get(split_name, expected)
            validate_result_shards(
                source_paths,
                expected_keys=contract["expected_episode_keys"],
                expected_methods=contract["methods"],
                budgets=contract["budgets"],
                seeds=contract.get("seeds", []),
                training_seeds=contract.get("training_seeds"),
                eval_seeds=contract.get("eval_seeds"),
                metadata_paths=contract.get("metadata_paths"),
            )
        else:
            _validate_legacy_rows(rows)
        merged_csv = output_dir / f"{split_name}_results.csv"
        write_rows(merged_csv, rows)
        summary_rows = summarize_rows(rows)
        write_summary_csv(summaries_dir / f"{split_name}_method_summary_by_N.csv", summary_rows)
        metadata = {
            "split": split_name,
            "budgets": budgets,
            "num_rows": len(rows),
            "source_run_root": str(run_root),
            "run_name": run_name,
        }
        (output_dir / f"{split_name}_results.metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    manifest_path = output_dir / "checkpoint_manifest.json"
    manifest_path.write_text(
        json.dumps(manifests, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"output_dir": str(output_dir), "checkpoint_manifest": str(manifest_path)}, indent=2
        )
    )


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "method",
        "N",
        "image_id",
        "episode_key",
        "seed",
        "training_seed",
        "eval_seed",
        "iou",
        "dice",
        "boundary_iou",
        "wallclock_ms",
    ]
    extras = sorted({key for row in rows for key in row if key not in fieldnames})
    fieldnames.extend(extras)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _validate_legacy_rows(rows: list[dict[str, str]]) -> None:
    seen = set()
    for row in rows:
        identity = (row.get("method"), row.get("N"), row.get("image_id"), row.get("seed"))
        if identity in seen:
            raise ValueError(f"Duplicate legacy BC result identity {identity}.")
        seen.add(identity)
        for metric in ("iou", "dice", "boundary_iou", "wallclock_ms"):
            if not math.isfinite(float(row[metric])):
                raise ValueError(f"Nonfinite legacy BC metric {metric}.")


def summarize_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in rows:
        key = (str(row["method"]), int(row["N"]))
        grouped.setdefault(key, []).append(row)
    summary_rows: list[dict[str, object]] = []
    for (method, budget), items in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][1])
    ):
        count = len(items)
        summary_rows.append(
            {
                "method": method,
                "N": budget,
                "num_rows": count,
                "mean_iou": sum(float(item["iou"]) for item in items) / max(count, 1),
                "mean_dice": sum(float(item["dice"]) for item in items) / max(count, 1),
                "mean_boundary_iou": sum(float(item["boundary_iou"]) for item in items)
                / max(count, 1),
                "mean_wallclock_ms": sum(float(item["wallclock_ms"]) for item in items)
                / max(count, 1),
            }
        )
    return summary_rows


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "method",
        "N",
        "num_rows",
        "mean_iou",
        "mean_dice",
        "mean_boundary_iou",
        "mean_wallclock_ms",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
