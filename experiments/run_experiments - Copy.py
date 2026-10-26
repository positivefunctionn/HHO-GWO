
"""Run the seven configured algorithms: HHO-GWO, ABC, GWO, WOA, HHO, ACO, and GA."""

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
from src.ga import ga
from src.abc import artificial_bee_colony
from src.aco import ant_colony
from src.woa import whale_optimization
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