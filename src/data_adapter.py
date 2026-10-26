
"""Load and validate Person 1's handoff without modifying source values.

Original CSV fields and values are preserved in memory.
Actual CPU is used only for post-hoc evaluation.
"""

import csv
import math


def _as_float(value, name, row_number, errors):
    try:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError
        return number
    except (TypeError, ValueError):
        errors.append(f"row {row_number}: invalid numeric {name}")
        return 0.0


def load_and_validate(
    path,
    expected_rows=None,
    overload_threshold=80.0,
    forecast_mode="bilstm",
):
    """Load a Person 1 or legacy scheduler CSV.

    forecast_mode:
        bilstm      -> BiLSTM CPU predictions and labels
        persistence -> persistence CPU predictions and labels
    """

    if forecast_mode not in ("bilstm", "persistence"):
        raise ValueError(
            "forecast_mode must be 'bilstm' or 'persistence'"
        )

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        source_rows = list(reader)
        columns = set(reader.fieldnames or [])

    errors = []

    if not source_rows:
        return [], {
            "row_count": 0,
            "errors": ["CSV contains no data rows"],
            "valid": False,
            "risk_score_source": None,
            "input_columns": sorted(columns),
        }

    person1_base_columns = {"timestamp_ms", "actual_cpu"}
    person1_columns = person1_base_columns | {
        "bilstm_predicted_cpu",
        "bilstm_predicted_overload",
    }
    legacy_columns = {
        "actual_cpu",
        "predicted_cpu",
        "p_overload",
        "actual_overload",
        "predicted_overload",
    }

    if person1_base_columns.issubset(columns):
        if forecast_mode == "bilstm":
            schema = "person1_bilstm_handoff"
            cpu_column = "bilstm_predicted_cpu"
            label_column = "bilstm_predicted_overload"
        else:
            schema = "person1_persistence_handoff"
            cpu_column = "persistence_predicted_cpu"
            label_column = "persistence_predicted_overload"

        required_columns = {
            "timestamp_ms",
            "actual_cpu",
            cpu_column,
            label_column,
        }
        missing = sorted(required_columns - columns)

        if missing:
            return [], {
                "row_count": len(source_rows),
                "errors": [
                    f"Missing required columns for {forecast_mode} mode: "
                    + ", ".join(missing)
                ],
                "valid": False,
                "schema": schema,
                "forecast_mode": forecast_mode,
                "risk_score_source": None,
                "input_columns": sorted(columns),
            }

        risk_source = (
            f"{label_column}: binary 0/1 label, "
            "not a calibrated probability"
        )

    elif legacy_columns.issubset(columns):
        if forecast_mode != "bilstm":
            return [], {
                "row_count": len(source_rows),
                "errors": [
                    "Persistence mode requires Person 1's CSV "
                    "with persistence prediction columns."
                ],
                "valid": False,
                "schema": "legacy_scheduler_handoff",
                "forecast_mode": forecast_mode,
                "risk_score_source": None,
                "input_columns": sorted(columns),
            }

        schema = "legacy_scheduler_handoff"
        cpu_column = "predicted_cpu"
        label_column = "predicted_overload"
        risk_source = "p_overload column"

    else:
        missing_person1 = sorted(person1_columns - columns)
        missing_legacy = sorted(legacy_columns - columns)

        return [], {
            "row_count": len(source_rows),
            "errors": [
                "Unrecognized handoff schema. Missing Person 1 "
                "columns: "
                + ", ".join(missing_person1)
                + "; missing legacy columns: "
                + ", ".join(missing_legacy)
            ],
            "valid": False,
            "risk_score_source": None,
            "input_columns": sorted(columns),
        }

    if expected_rows is not None and len(source_rows) != expected_rows:
        errors.append(
            f"row count {len(source_rows)} != expected {expected_rows}"
        )

    rows = []

    for i, raw in enumerate(source_rows, start=2):
        actual = _as_float(
            raw.get("actual_cpu"), "actual_cpu", i, errors
        )
        predicted = _as_float(
            raw.get(cpu_column), cpu_column, i, errors
        )
        pred_label_value = _as_float(
            raw.get(label_column), label_column, i, errors
        )

        # This entire block must stay inside the for loop.
        if schema in (
            "person1_bilstm_handoff",
            "person1_persistence_handoff",
        ):
            timestamp = _as_float(
                raw.get("timestamp_ms"), "timestamp_ms", i, errors
            )
            # The supplied binary label is a 0/1 risk score,
            # not a calibrated probability.
            p_overload = pred_label_value
        else:
            timestamp = _as_float(
                raw.get("timestamp", i - 2), "timestamp", i, errors
            )
            p_overload = _as_float(
                raw.get("p_overload"), "p_overload", i, errors
            )

        if pred_label_value not in (0.0, 1.0):
            errors.append(
                f"row {i}: {label_column} must be 0 or 1"
            )

        if not 0.0 <= p_overload <= 1.0:
            errors.append(
                f"row {i}: p_overload/risk score outside [0,1]"
            )

        pred_label = int(pred_label_value)
        actual_label = int(actual >= overload_threshold)

        # Preserve original CSV fields and add normalized fields.
        # The scheduler should use predicted values, not actual CPU.
        normalized = dict(raw)
        normalized.update({
            "timestamp": timestamp,
            "actual_cpu": actual,
            "predicted_cpu": predicted,
            "p_overload": p_overload,
            "actual_overload": actual_label,
            "predicted_overload": pred_label,
            "source_schema": schema,
            "forecast_mode": forecast_mode,
        })

        rows.append(normalized)

    return rows, {
        "row_count": len(rows),
        "errors": errors,
        "valid": not errors,
        "schema": schema,
        "forecast_mode": forecast_mode,
        "prediction_cpu_column": cpu_column,
        "prediction_label_column": label_column,
        "risk_score_source": risk_source,
        "input_columns": sorted(columns),
        "overload_threshold": overload_threshold,
        "note": (
            "Original CSV fields are preserved. Actual CPU is for "
            "post-hoc evaluation only. Verify the timestamp unit "
            "before interpreting timestamps."
        ),
    }


def horizons(rows, length):
    if length < 1:
        raise ValueError("horizon length must be positive")

    return [
        rows[i:i + length]
        for i in range(0, len(rows), length)
    ]
