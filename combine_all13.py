
import json
from pathlib import Path

import pandas as pd

ROOT = Path("results")
BACKUP = Path("results_backup_before_forecast_runs")

ALGORITHMS = [
    "HHO-GWO", "ABC", "GWO", "HHO", "GA", "WOA", "ACO",
    "FCFS", "Round Robin", "BFD", "SA", "DE", "NSGA-II",
]

# These fields must match between the original and new runs.
PROTOCOL_KEYS = [
    "benchmark_type",
    "task_counts",
    "load_levels",
    "vm_count",
    "vms_per_host",
    "objective_profile",
    "host_count",
    "evaluation_budget",
    "weights",
    "normalization_references",
    "host_power_curve_exponent",
    "host_power_formula",
    "host_model",
    "vm_initial_state",
    "demand_model",
]

EXPERIMENTS = {
    "balanced": {
        "original": BACKUP / "final_convex_balanced",
        "additional": ROOT / "additional_balanced_merged_v2",
        "output": ROOT / "final_13algo_balanced",
    },
    "sla": {
        "original": BACKUP / "final_convex_sla",
        "additional": ROOT / "additional_sla_missing",
        "output": ROOT / "final_13algo_sla",
    },
}


def read_json(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def canonical(value):
    return json.dumps(value, sort_keys=True)


def combine(name, cfg):
    original_dir = cfg["original"]
    additional_dir = cfg["additional"]
    output_dir = cfg["output"]

    original_csv = original_dir / "runs.csv"
    additional_csv = additional_dir / "runs.csv"
    original_protocol_path = original_dir / "protocol.json"
    additional_protocol_path = additional_dir / "protocol.json"

    for path in (
        original_csv, additional_csv,
        original_protocol_path, additional_protocol_path,
    ):
        if not path.exists():
            raise FileNotFoundError(f"{name}: missing {path}")

    # Never overwrite an existing output.
    if output_dir.exists():
        raise FileExistsError(
            f"{output_dir} already exists. Inspect it first; "
            "this script will not overwrite it."
        )

    old = pd.read_csv(original_csv)
    new = pd.read_csv(additional_csv)
    old_protocol = read_json(original_protocol_path)
    new_protocol = read_json(additional_protocol_path)

    print(f"\n=== {name.upper()} ===")
    print("Original rows:", len(old))
    print("Additional rows:", len(new))

    # Check the essential protocol settings.
    for key in PROTOCOL_KEYS:
        if key not in old_protocol or key not in new_protocol:
            raise ValueError(f"{name}: protocol missing key {key!r}")
        if canonical(old_protocol[key]) != canonical(new_protocol[key]):
            raise ValueError(
                f"{name}: protocol mismatch for {key!r}\n"
                f"Original:   {old_protocol[key]!r}\n"
                f"Additional: {new_protocol[key]!r}"
            )

    if list(old.columns) != list(new.columns):
        raise ValueError(
            f"{name}: CSV columns differ.\n"
            f"Original:   {old.columns.tolist()}\n"
            f"Additional: {new.columns.tolist()}"
        )

    required = {"algorithm", "scenario", "seed"}
    if not required.issubset(old.columns):
        raise ValueError(f"{name}: required columns missing: {required}")

    old_algorithms = set(old["algorithm"].dropna().unique())
    new_algorithms = set(new["algorithm"].dropna().unique())

    overlap = old_algorithms & new_algorithms
    if overlap:
        raise ValueError(
            f"{name}: algorithms appear in both datasets: {sorted(overlap)}"
        )

    if old_algorithms | new_algorithms != set(ALGORITHMS):
        raise ValueError(
            f"{name}: unexpected algorithm panel.\n"
            f"Missing: {sorted(set(ALGORITHMS) - (old_algorithms | new_algorithms))}\n"
            f"Unexpected: {sorted((old_algorithms | new_algorithms) - set(ALGORITHMS))}"
        )

    # Confirm that the two datasets cover exactly the same
    # seed/scenario cells.
    cells = ["scenario", "seed"]
    old_cells = set(map(tuple, old[cells].drop_duplicates().to_numpy()))
    new_cells = set(map(tuple, new[cells].drop_duplicates().to_numpy()))

    if old_cells != new_cells:
        raise ValueError(
            f"{name}: seed/scenario coverage differs. "
            f"Original cells={len(old_cells)}, additional cells={len(new_cells)}"
        )

    combined = pd.concat([old, new], ignore_index=True)

    # Every algorithm must have one row per seed/scenario cell.
    keys = ["algorithm", "scenario", "seed"]
    duplicated = combined.duplicated(keys, keep=False)
    if duplicated.any():
        print(combined.loc[duplicated, keys].head(20).to_string(index=False))
        raise ValueError(f"{name}: duplicate algorithm/scenario/seed rows")

    counts = combined.groupby("algorithm").size().to_dict()
    expected_per_algorithm = len(old_cells)

    if set(counts) != set(ALGORITHMS):
        raise ValueError(f"{name}: final algorithm list is incomplete")

    if any(n != expected_per_algorithm for n in counts.values()):
        raise ValueError(
            f"{name}: unexpected rows per algorithm. "
            f"Expected {expected_per_algorithm}; got {counts}"
        )

    expected_total = len(ALGORITHMS) * expected_per_algorithm
    if len(combined) != expected_total:
        raise ValueError(
            f"{name}: expected {expected_total} rows, got {len(combined)}"
        )

    # Build new metadata from the original protocol.
    protocol = dict(old_protocol)
    protocol["algorithms"] = ALGORITHMS
    protocol["seeds"] = sorted(
        int(x) for x in combined["seed"].drop_duplicates().tolist()
    )
    protocol["seed_count"] = len(protocol["seeds"])
    protocol["algorithm_runs"] = len(combined)
    protocol["runs_per_algorithm"] = expected_per_algorithm
    protocol["combined_from"] = [
        str(original_dir),
        str(additional_dir),
    ]

    output_dir.mkdir(parents=True, exist_ok=False)
    combined.to_csv(output_dir / "runs.csv", index=False)
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2),
        encoding="utf-8",
    )

    print("Algorithms:", len(counts))
    print("Rows per algorithm:", expected_per_algorithm)
    print("Total rows:", len(combined))
    print("Algorithm counts:", counts)
    print("Validated and saved:", output_dir)


def main():
    for name, cfg in EXPERIMENTS.items():
        combine(name, cfg)

    print("\nBoth datasets passed the combination checks.")
    print("Original backup files were not modified.")


if __name__ == "__main__":
    main()
