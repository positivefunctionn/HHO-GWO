"""Evaluate supplied CPU prediction columns on a chronological tail segment.

IMPORTANT: This evaluates already-supplied predictions, not a trained model. Because
upstream model training/inference provenance is absent, this does NOT prove that the
predictions were generated without leakage or that the segment is a true unseen test set.
The output is a diagnostic and must not be reported as validated model performance.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    "timestamp_ms", "actual_cpu", "bilstm_predicted_cpu",
    "persistence_predicted_cpu", "bilstm_predicted_overload",
    "persistence_predicted_overload",
}


def regression_metrics(actual, predicted):
    n = len(actual)
    mae = sum(abs(a-p) for a, p in zip(actual, predicted)) / n
    rmse = math.sqrt(sum((a-p)**2 for a, p in zip(actual, predicted)) / n)
    mean = sum(actual) / n
    ss_tot = sum((a-mean)**2 for a in actual)
    r2 = 1 - sum((a-p)**2 for a, p in zip(actual, predicted)) / ss_tot if ss_tot else None
    return {"mae": mae, "rmse": rmse, "r2": r2}


def classification_metrics(actual_labels, predicted_labels):
    tp = sum(a == 1 and p == 1 for a, p in zip(actual_labels, predicted_labels))
    tn = sum(a == 0 and p == 0 for a, p in zip(actual_labels, predicted_labels))
    fp = sum(a == 0 and p == 1 for a, p in zip(actual_labels, predicted_labels))
    fn = sum(a == 1 and p == 0 for a, p in zip(actual_labels, predicted_labels))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "confusion_matrix_tn_fp_fn_tp": [[tn, fp], [fn, tp]]}


def evaluate(path, holdout_fraction=0.2, threshold=80.0):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        missing = sorted(REQUIRED - fields)
        if missing:
            raise ValueError("Missing required columns: " + ", ".join(missing))
        rows = list(reader)
    if len(rows) < 10:
        raise ValueError("At least 10 rows are required for a useful chronological diagnostic")
    for i, row in enumerate(rows, start=2):
        for key in REQUIRED:
            try:
                value = float(row[key])
                if not math.isfinite(value):
                    raise ValueError
            except (TypeError, ValueError):
                raise ValueError(f"Row {i}: invalid numeric value in {key}")
    rows.sort(key=lambda r: float(r["timestamp_ms"]))
    timestamps = [float(r["timestamp_ms"]) for r in rows]
    if len(set(timestamps)) != len(timestamps):
        raise ValueError("Duplicate timestamps found; resolve ordering before evaluation")
    split = max(1, min(len(rows)-1, int(len(rows) * (1-holdout_fraction))))
    tail = rows[split:]
    actual = [float(r["actual_cpu"]) for r in tail]
    labels = [int(v >= threshold) for v in actual]
    results = {}
    for name, cpu_col, label_col in (
        ("bilstm", "bilstm_predicted_cpu", "bilstm_predicted_overload"),
        ("persistence", "persistence_predicted_cpu", "persistence_predicted_overload"),
    ):
        preds = [float(r[cpu_col]) for r in tail]
        pred_labels = [int(float(r[label_col])) for r in tail]
        if any(p not in (0, 1) for p in pred_labels):
            raise ValueError(f"{label_col} must contain only 0/1 labels")
        results[name] = {
            "cpu_regression": regression_metrics(actual, preds),
            "overload_classification": classification_metrics(labels, pred_labels),
        }
    return {
        "evaluation_type": "chronological-tail diagnostic of precomputed predictions",
        "valid_model_test_claim": False,
        "warning": ("Upstream training, scaler, forecast horizon, and prediction-generation provenance are absent. "
                    "Predictions may have been generated using information from this segment; do not call this a validated unseen test."),
        "source_file": str(Path(path).name), "rows_total": len(rows), "rows_diagnostic_tail": len(tail),
        "holdout_fraction_requested": holdout_fraction, "timestamp_column": "timestamp_ms",
        "timestamp_note": "Column name retained from source; units are not inferred or converted.",
        "first_tail_timestamp_value": timestamps[split], "last_tail_timestamp_value": timestamps[-1],
        "overload_threshold_cpu": threshold,
        "label_definition": f"actual_cpu >= {threshold}",
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/person2_scheduler_input.csv")
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=80.0)
    parser.add_argument("--output", default="results/prediction_handoff_diagnostic.json")
    args = parser.parse_args()
    if not 0.05 <= args.holdout_fraction <= 0.5:
        parser.error("--holdout-fraction must be between 0.05 and 0.5")
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    result = evaluate(input_path, args.holdout_fraction, args.threshold)
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nSaved diagnostic to: {out_path}")

if __name__ == "__main__":
    main()
