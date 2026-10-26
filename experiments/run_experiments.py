
"""Full configured comparison runner for BFD, GWO, HHO, HHO-GWO, PSO and GA."""

import csv
import statistics
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_adapter import horizons, load_and_validate
from src.demand_adapter import adapt, write_repair_log
from src.gwo import gwo
from src.hho import hho
from src.hybrid import hho_gwo
from src.woa import whale_optimization
from src.aco import ant_colony
from src.abc import artificial_bee_colony
from src.ga import ga
from src.models import VM
from src.fitness import reset_evaluation_count, get_evaluation_count


def write_csv(path, rows, fields):
    """Write experiment results to CSV."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def task_rows(tasks, assignment, vms, cfg, algorithm, horizon_id, trial, seed):
    """Generate scheduled-task results, including post-hoc actual-load metrics."""
    predicted_loads = [vm.current_cpu_load for vm in vms]
    actual_loads = [vm.current_cpu_load for vm in vms]
    output = []

    for task, index in zip(tasks, assignment):
        vm = vms[index]

        predicted_loads[index] += task.demand
        predicted_util = predicted_loads[index] / vm.cpu_capacity
        predicted_execution = task.demand / vm.cpu_capacity

        predicted_violation = (
            predicted_util > cfg["allowable_utilization"]
            or predicted_execution > cfg["execution_sla"]
        )

        # Post-hoc evaluation only. Actual CPU must not influence optimization.
        actual_demand = (
            max(0.0, task.actual_cpu) / 100.0 * vms[0].cpu_capacity
        )
        actual_loads[index] += actual_demand
        actual_util = actual_loads[index] / vm.cpu_capacity
        actual_execution = actual_demand / vm.cpu_capacity

        actual_violation = (
            actual_util > cfg["allowable_utilization"]
            or actual_execution > cfg["execution_sla"]
        )

        output.append({
            "task_id": task.task_id,
            "horizon_id": horizon_id,
            "timestamp_source_value": task.timestamp,
            "actual_cpu_evaluation_only": task.actual_cpu,
            "predicted_cpu": task.predicted_cpu,
            "p_overload_risk_score": task.p_overload,
            "predicted_overload": task.predicted_overload,
            "actual_overload_label": int(
                task.actual_cpu >= cfg["overload_threshold"]
            ),
            "selected_vm_id": vm.vm_id,
            "host_id": vm.host_id,
            "predicted_resulting_utilization_proxy": predicted_util,
            "predicted_sla_violation_proxy": int(predicted_violation),
            "actual_resulting_utilization_posthoc_proxy": actual_util,
            "actual_sla_violation_posthoc_proxy": int(actual_violation),
            "algorithm": algorithm,
            "trial": trial,
            "seed": seed,
        })

    return output


def main():
    # ---------------------------------------------------------
    # 1. Load and validate configuration
    # ---------------------------------------------------------
    config_path = ROOT / "config" / "config.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    forecast_mode = cfg.get("forecast_mode", "bilstm")

    if forecast_mode not in ("bilstm", "persistence"):
        raise ValueError(
            "forecast_mode must be 'bilstm' or 'persistence' "
            "in config/config.yaml"
        )

    trials_count = cfg["trials"]
    seeds = cfg["seeds"][:trials_count]

    if len(seeds) != trials_count:
        raise ValueError("Configuration has fewer seeds than trials")

    # ---------------------------------------------------------
    # 2. Load the selected forecast input
    # ---------------------------------------------------------
    source = ROOT / cfg.get(
        "input_file", "data/person2_scheduler_input.csv"
    )

    rows, report = load_and_validate(
        source,
        expected_rows=cfg.get("expected_rows"),
        overload_threshold=cfg.get("overload_threshold", 80.0),
        forecast_mode=forecast_mode,
    )

    if not report["valid"]:
        raise ValueError(
            "Handoff validation failed: "
            + "; ".join(report["errors"])
        )

    # Separate output folders prevent the two forecast runs
    # from overwriting one another.
    out = ROOT / "results" / f"forecast_{forecast_mode}_seven_algorithms"
    out.mkdir(parents=True, exist_ok=True)

    validation_text = (
        "Input validation report\n"
        + "\n".join(f"{key}: {value}" for key, value in report.items())
        + "\n"
    )

    (out / "input_validation.txt").write_text(
        validation_text,
        encoding="utf-8",
    )

    vms = [VM(**item) for item in cfg["vms"]]

    # ---------------------------------------------------------
    # 3. Initialize result collections
    # ---------------------------------------------------------
    repairs = []
    horizon_results = []
    schedules = []

    convergence = {
        "ABC": [],
        "GWO": [],
        "WOA": [],
        "HHO": [],
        "ACO": [],
        "GA": [],
        "HHO-GWO": [],
    }

    methods = {
        "ABC": lambda tasks, seed: artificial_bee_colony(
            tasks, vms, cfg, seed,
            max_evaluations=cfg.get("evaluation_budget", 300),
        )[:3],
        "GWO": lambda tasks, seed: gwo(
            tasks, vms, cfg, seed,
            max_evaluations=cfg.get("evaluation_budget", 300),
        )[:3],
        "WOA": lambda tasks, seed: whale_optimization(
            tasks, vms, cfg, seed,
            max_evaluations=cfg.get("evaluation_budget", 300),
        )[:3],
        "HHO": lambda tasks, seed: hho(
            tasks, vms, cfg, seed,
            max_evaluations=cfg.get("evaluation_budget", 300),
        )[:3],
        "ACO": lambda tasks, seed: ant_colony(
            tasks, vms, cfg, seed,
            max_evaluations=cfg.get("evaluation_budget", 300),
        )[:3],
        "GA": lambda tasks, seed: ga(
            tasks, vms, cfg, seed,
            max_evaluations=cfg.get("evaluation_budget", 300),
        )[:3],
        "HHO-GWO": lambda tasks, seed: hho_gwo(
            tasks, vms, cfg, seed,
            max_evaluations=cfg.get("evaluation_budget", 300),
        )[:3],
    }

    all_horizons = horizons(rows, cfg["horizon_length"])

    # ---------------------------------------------------------
    # 4. Run every algorithm on every horizon and seed
    # ---------------------------------------------------------
    for hid, batch in enumerate(all_horizons):
        if hid % 10 == 0:
            print(
                f"[{forecast_mode}] processing horizon "
                f"{hid + 1}/{len(all_horizons)}",
                flush=True,
            )

        before = len(repairs)

        tasks = adapt(
            batch,
            hid,
            vms[0].cpu_capacity,
            repairs,
        )

        for algorithm, run in methods.items():
            for trial, seed in enumerate(seeds):
                reset_evaluation_count(
                    limit=cfg.get("evaluation_budget")
                )

                started = time.perf_counter()
                assignment, metric, history = run(tasks, seed)
                runtime = time.perf_counter() - started
                evaluations = get_evaluation_count()

                if (
                    len(assignment) != len(tasks)
                    or any(not 0 <= i < len(vms) for i in assignment)
                ):
                    raise RuntimeError(
                        f"Invalid {algorithm} schedule in horizon "
                        f"{hid}, seed {seed}"
                    )

                horizon_results.append({
                    "horizon_id": hid,
                    "algorithm": algorithm,
                    "trial": trial,
                    "seed": seed,
                    **metric.__dict__,
                    "sla_violation_rate": (
                        metric.sla_violation_count / len(tasks)
                        if tasks else 0.0
                    ),
                    "overload_rate": metric.overload_count / len(vms),
                    "runtime": runtime,
                    "evaluations": evaluations,
                    "prediction_repair_count": len(repairs) - before,
                })

                schedules.extend(
                    task_rows(
                        tasks,
                        assignment,
                        vms,
                        cfg,
                        algorithm,
                        hid,
                        trial,
                        seed,
                    )
                )

                if algorithm in convergence:
                    convergence[algorithm].extend(
                        {
                            "horizon_id": hid,
                            "trial": trial,
                            "seed": seed,
                            "iteration": i + 1,
                            "fitness": value,
                        }
                        for i, value in enumerate(history)
                    )

    if not horizon_results:
        raise RuntimeError("No horizons were produced from the input data")

    # ---------------------------------------------------------
    # 5. Save detailed results
    # ---------------------------------------------------------
    write_csv(
        out / "horizon_results.csv",
        horizon_results,
        list(horizon_results[0]),
    )

    write_csv(
        out / "schedule_results.csv",
        schedules,
        list(schedules[0]),
    )

    write_repair_log(out / "repair_log.csv", repairs)

    for algorithm, records in convergence.items():
        write_csv(
            out / f"convergence_{algorithm.lower().replace('-', '_')}.csv",
            records,
            ["horizon_id", "trial", "seed", "iteration", "fitness"],
        )

    # ---------------------------------------------------------
    # 6. Aggregate results by algorithm and trial
    # ---------------------------------------------------------
    fields = (
        "fitness",
        "energy",
        "makespan",
        "sla_violation_count",
        "load_imbalance",
        "average_utilization",
        "overload_count",
        "runtime",
        "evaluations",
    )

    trials = []

    for algorithm in methods:
        for trial, seed in enumerate(seeds):
            group = [
                item for item in horizon_results
                if item["algorithm"] == algorithm
                and item["seed"] == seed
            ]

            if not group:
                continue

            trials.append({
                "algorithm": algorithm,
                "trial": trial,
                "seed": seed,
                **{
                    name: statistics.fmean(item[name] for item in group)
                    for name in fields
                },
            })

    comparison = []

    for algorithm in methods:
        group = [
            item for item in trials
            if item["algorithm"] == algorithm
        ]

        if not group:
            continue

        record = {
            "algorithm": algorithm,
            "seed_count": len(group),
        }

        for name in fields:
            values = [item[name] for item in group]
            record[f"{name}_fmean"] = statistics.fmean(values)
            record[f"{name}_stdev"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )

        comparison.append(record)

    write_csv(
        out / "trial_results.csv",
        trials,
        list(trials[0]),
    )

    write_csv(
        out / "algorithm_comparison.csv",
        comparison,
        list(comparison[0]),
    )

    # ---------------------------------------------------------
    # 7. Print experiment summary
    # ---------------------------------------------------------
    print("\nExperiment completed successfully.")
    print(f"Forecast mode: {forecast_mode}")
    print(f"Output directory: {out}")
    print(f"Input rows: {len(rows)}")
    print(f"Horizons: {len(all_horizons)}")
    print(f"Input schema: {report['schema']}")
    print(f"Risk input: {report['risk_score_source']}")
    print(f"Algorithms: {len(methods)}")
    print(f"Seeds/trials: {len(seeds)}")
    print(f"Schedules evaluated: {len(horizon_results)}")
    print(
        "Objective-evaluation cap per horizon/run: "
        f"{cfg.get('evaluation_budget', 'unlimited')}"
    )


if __name__ == "__main__":
    main()
