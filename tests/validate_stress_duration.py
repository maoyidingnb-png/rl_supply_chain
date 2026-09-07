"""
validate_stress_duration.py

Guide-aligned controlled disruption experiment for SupplyChainEnv.

SOURCE-ALIGNED EXPERIMENT GRID
------------------------------
The experiment durations follow the supplied Stage-2 guide:

T1  U1: 1, 3, 5 days
T2  U2: 1, 3, 5 days
T3  U3: 1, 3, 5, 7 days

S1  A1: 3, 7, 14 days
S2  A2: 3, 7, 14 days
S3  A3: 1, 3, 5, 7 days

C1  A2 + U2: 3, 5 days
C2  A3 + U3: 1, 3 days

Plus one Normal baseline.

Total:
    24 forced-disruption combinations + 1 Normal = 25 experiment groups.

IMPORTANT
---------
1. This script does NOT train RL.
2. This script does NOT modify P_A or P_U.
3. During the forced disruption window, A/U are held at the specified state.
4. After the disruption window, the environment is forced to A0/U0 for recovery.
5. A fixed diagnostic policy is used by default:
       action 0 = (l=0, e=0, g=0)
   so differences are attributable to the disruption rather than policy changes.
6. Same replication seed schedule is used for every experiment group.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from env import SupplyChainEnv


# =============================================================================
# Guide-aligned experiment grid
# =============================================================================

EXPERIMENTS = [
    # Baseline
    {
        "group": "N0",
        "risk_family": "Normal",
        "scenario": "Normal",
        "A_state": 0,
        "U_state": 0,
        "duration": 0,
    },

    # T1: U1, 1/3/5 days
    *[
        {
            "group": "T1",
            "risk_family": "Transport",
            "scenario": "U1",
            "A_state": 0,
            "U_state": 1,
            "duration": d,
        }
        for d in (1, 3, 5)
    ],

    # T2: U2, 1/3/5 days
    *[
        {
            "group": "T2",
            "risk_family": "Transport",
            "scenario": "U2",
            "A_state": 0,
            "U_state": 2,
            "duration": d,
        }
        for d in (1, 3, 5)
    ],

    # T3: U3, 1/3/5/7 days
    *[
        {
            "group": "T3",
            "risk_family": "Transport",
            "scenario": "U3",
            "A_state": 0,
            "U_state": 3,
            "duration": d,
        }
        for d in (1, 3, 5, 7)
    ],

    # S1: A1, 3/7/14 days
    *[
        {
            "group": "S1",
            "risk_family": "Supplier",
            "scenario": "A1",
            "A_state": 1,
            "U_state": 0,
            "duration": d,
        }
        for d in (3, 7, 14)
    ],

    # S2: A2, 3/7/14 days
    *[
        {
            "group": "S2",
            "risk_family": "Supplier",
            "scenario": "A2",
            "A_state": 2,
            "U_state": 0,
            "duration": d,
        }
        for d in (3, 7, 14)
    ],

    # S3: A3, 1/3/5/7 days
    *[
        {
            "group": "S3",
            "risk_family": "Supplier",
            "scenario": "A3",
            "A_state": 3,
            "U_state": 0,
            "duration": d,
        }
        for d in (1, 3, 5, 7)
    ],

    # C1: A2 + U2, 3/5 days
    *[
        {
            "group": "C1",
            "risk_family": "Compound",
            "scenario": "A2_U2",
            "A_state": 2,
            "U_state": 2,
            "duration": d,
        }
        for d in (3, 5)
    ],

    # C2: A3 + U3, 1/3 days
    *[
        {
            "group": "C2",
            "risk_family": "Compound",
            "scenario": "A3_U3",
            "A_state": 3,
            "U_state": 3,
            "duration": d,
        }
        for d in (1, 3)
    ],
]


def experiment_id(exp):
    if exp["group"] == "N0":
        return "N0_Normal"
    return f'{exp["group"]}_{exp["scenario"]}_{exp["duration"]}d'


def ci95(values):
    x = np.asarray(values, dtype=np.float64)
    if x.size <= 1:
        return 0.0
    return 1.96 * float(np.std(x, ddof=1) / np.sqrt(x.size))


def run_one_replication(
    exp,
    seed,
    horizon,
    action,
):
    """
    Run one controlled replication.

    For duration d > 0:
        decision days 1..d       -> forced scenario state
        decision days d+1..H     -> forced A0/U0

    For Normal:
        all decision days -> A0/U0
    """

    duration = int(exp["duration"])
    A_state = int(exp["A_state"])
    U_state = int(exp["U_state"])

    env = SupplyChainEnv(
        seed=seed,
        max_steps=horizon,
    )

    decoded = env.decode_action(action)

    # Initial state before reset.
    env.set_disruption_override(
        A_t=A_state,
        U_t=U_state,
    )
    obs, _ = env.reset(seed=seed)

    totals = {
        "profit": 0.0,
        "revenue": 0.0,
        "lost_sales_penalty": 0.0,
        "holding_cost": 0.0,
        "expedite_cost": 0.0,
        "demand_units": 0.0,
        "sales_units": 0.0,
        "lost_units": 0.0,

        "stress_profit": 0.0,
        "stress_demand_units": 0.0,
        "stress_sales_units": 0.0,
        "stress_lost_units": 0.0,

        "recovery_profit": 0.0,
        "recovery_demand_units": 0.0,
        "recovery_sales_units": 0.0,
        "recovery_lost_units": 0.0,
    }

    supplier_backlog_history = []
    capacity_backlog_history = []
    transport_backlog_history = []
    pipeline_history = []
    cw_inventory_history = []
    fw_inventory_history = []

    stress_supplier_backlog = []
    stress_transport_backlog = []
    stress_cw_inventory = []
    stress_fw_inventory = []

    steps = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        # After the forced disruption window, switch to Normal.
        # Normal baseline has duration=0 and therefore switches immediately.
        if steps == duration:
            env.set_disruption_override(
                A_t=0,
                U_t=0,
            )

        mask = env.action_masks()

        if not bool(mask[action]):
            raise RuntimeError(
                f"Diagnostic action {action}={decoded} is invalid for "
                f"{experiment_id(exp)} at A={env.A_t}, U={env.U_t}. "
                "Use action 0 for a scenario-independent diagnostic policy."
            )

        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        # Full episode totals.
        totals["profit"] += float(info["profit"])
        totals["revenue"] += float(info["revenue"])
        totals["lost_sales_penalty"] += float(
            info["lost_sales_penalty"]
        )
        totals["holding_cost"] += float(
            info["holding_cost"]
        )
        totals["expedite_cost"] += float(
            info["expedite_cost"]
        )
        totals["demand_units"] += float(
            info["demand_units"]
        )
        totals["sales_units"] += float(
            info["sales_units"]
        )
        totals["lost_units"] += float(
            info["lost_units"]
        )

        in_stress = steps < duration

        prefix = "stress" if in_stress else "recovery"

        totals[f"{prefix}_profit"] += float(
            info["profit"]
        )
        totals[f"{prefix}_demand_units"] += float(
            info["demand_units"]
        )
        totals[f"{prefix}_sales_units"] += float(
            info["sales_units"]
        )
        totals[f"{prefix}_lost_units"] += float(
            info["lost_units"]
        )

        supplier_backlog = float(
            info["supplier_backlog_units"]
        )
        capacity_backlog = float(
            info["capacity_backlog_units"]
        )
        transport_backlog = float(
            info["transport_backlog_units"]
        )
        pipeline_units = float(
            info["pipeline_units"]
        )
        cw_inventory = float(
            info["cw_inventory_total"]
        )
        fw_inventory = float(
            info["fw_inventory_total"]
        )

        supplier_backlog_history.append(
            supplier_backlog
        )
        capacity_backlog_history.append(
            capacity_backlog
        )
        transport_backlog_history.append(
            transport_backlog
        )
        pipeline_history.append(
            pipeline_units
        )
        cw_inventory_history.append(
            cw_inventory
        )
        fw_inventory_history.append(
            fw_inventory
        )

        if in_stress:
            stress_supplier_backlog.append(
                supplier_backlog
            )
            stress_transport_backlog.append(
                transport_backlog
            )
            stress_cw_inventory.append(
                cw_inventory
            )
            stress_fw_inventory.append(
                fw_inventory
            )

        steps += 1

    fill_rate = (
        totals["sales_units"] / totals["demand_units"]
        if totals["demand_units"] > 0
        else 1.0
    )

    stress_fill_rate = (
        totals["stress_sales_units"]
        / totals["stress_demand_units"]
        if totals["stress_demand_units"] > 0
        else 1.0
    )

    recovery_fill_rate = (
        totals["recovery_sales_units"]
        / totals["recovery_demand_units"]
        if totals["recovery_demand_units"] > 0
        else 1.0
    )

    def mean_or_zero(x):
        return float(np.mean(x)) if len(x) else 0.0

    def max_or_zero(x):
        return float(np.max(x)) if len(x) else 0.0

    result = {
        "experiment_id": experiment_id(exp),
        "group": exp["group"],
        "risk_family": exp["risk_family"],
        "scenario": exp["scenario"],
        "A_state": A_state,
        "U_state": U_state,
        "duration": duration,
        "seed": int(seed),
        "horizon": int(horizon),
        "action": int(action),
        "l_t": int(decoded[0]),
        "e_t": int(decoded[1]),
        "g_t": int(decoded[2]),
        "steps": int(steps),

        **totals,

        "fill_rate": float(fill_rate),
        "stress_fill_rate": float(stress_fill_rate),
        "recovery_fill_rate": float(recovery_fill_rate),

        "mean_supplier_backlog": mean_or_zero(
            supplier_backlog_history
        ),
        "max_supplier_backlog": max_or_zero(
            supplier_backlog_history
        ),
        "end_supplier_backlog": float(
            supplier_backlog_history[-1]
        ),

        "mean_capacity_backlog": mean_or_zero(
            capacity_backlog_history
        ),
        "max_capacity_backlog": max_or_zero(
            capacity_backlog_history
        ),

        "mean_transport_backlog": mean_or_zero(
            transport_backlog_history
        ),
        "max_transport_backlog": max_or_zero(
            transport_backlog_history
        ),

        "mean_pipeline_units": mean_or_zero(
            pipeline_history
        ),
        "mean_cw_inventory": mean_or_zero(
            cw_inventory_history
        ),
        "mean_fw_inventory": mean_or_zero(
            fw_inventory_history
        ),

        # Stress-window-only state metrics.
        "stress_mean_supplier_backlog": mean_or_zero(
            stress_supplier_backlog
        ),
        "stress_max_supplier_backlog": max_or_zero(
            stress_supplier_backlog
        ),
        "stress_mean_transport_backlog": mean_or_zero(
            stress_transport_backlog
        ),
        "stress_max_transport_backlog": max_or_zero(
            stress_transport_backlog
        ),
        "stress_mean_cw_inventory": mean_or_zero(
            stress_cw_inventory
        ),
        "stress_mean_fw_inventory": mean_or_zero(
            stress_fw_inventory
        ),
    }

    return result


def build_summary(raw_df):
    metrics = [
        "profit",
        "revenue",
        "fill_rate",
        "lost_units",
        "lost_sales_penalty",
        "holding_cost",
        "expedite_cost",

        "stress_profit",
        "stress_fill_rate",
        "stress_lost_units",

        "recovery_profit",
        "recovery_fill_rate",
        "recovery_lost_units",

        "mean_supplier_backlog",
        "max_supplier_backlog",
        "end_supplier_backlog",
        "mean_capacity_backlog",
        "max_capacity_backlog",
        "mean_transport_backlog",
        "max_transport_backlog",
        "mean_pipeline_units",
        "mean_cw_inventory",
        "mean_fw_inventory",

        "stress_mean_supplier_backlog",
        "stress_max_supplier_backlog",
        "stress_mean_transport_backlog",
        "stress_max_transport_backlog",
        "stress_mean_cw_inventory",
        "stress_mean_fw_inventory",
    ]

    rows = []

    for exp in EXPERIMENTS:
        eid = experiment_id(exp)

        sub = raw_df[
            raw_df["experiment_id"] == eid
        ]

        row = {
            "experiment_id": eid,
            "group": exp["group"],
            "risk_family": exp["risk_family"],
            "scenario": exp["scenario"],
            "A_state": exp["A_state"],
            "U_state": exp["U_state"],
            "duration": exp["duration"],
            "n": len(sub),
        }

        for metric in metrics:
            vals = sub[metric].to_numpy(
                dtype=np.float64
            )

            row[f"{metric}_mean"] = float(
                np.mean(vals)
            )
            row[f"{metric}_std"] = float(
                np.std(vals, ddof=1)
                if len(vals) > 1
                else 0.0
            )
            row[f"{metric}_ci95"] = ci95(vals)

        rows.append(row)

    return pd.DataFrame(rows)


def add_paired_normal_deltas(
    raw_df,
    summary_df,
):
    """
    Pair by replication seed against Normal.

    Demand uses a separate RNG stream in env.py, so same seed gives directly
    comparable customer demand paths across experiment groups. Because actions
    and disruptions can alter lead-time RNG consumption, describe this as
    paired-by-seed / CRN-style evaluation rather than claiming every random
    draw is identical.
    """

    normal = (
        raw_df[
            raw_df["experiment_id"] == "N0_Normal"
        ][
            [
                "seed",
                "profit",
                "fill_rate",
                "lost_units",
                "holding_cost",
            ]
        ]
        .rename(
            columns={
                "profit": "normal_profit",
                "fill_rate": "normal_fill_rate",
                "lost_units": "normal_lost_units",
                "holding_cost": "normal_holding_cost",
            }
        )
    )

    delta_rows = []

    for exp in EXPERIMENTS:
        eid = experiment_id(exp)

        sub = raw_df[
            raw_df["experiment_id"] == eid
        ][
            [
                "seed",
                "profit",
                "fill_rate",
                "lost_units",
                "holding_cost",
            ]
        ]

        paired = sub.merge(
            normal,
            on="seed",
            how="inner",
        )

        d_profit = (
            paired["profit"]
            - paired["normal_profit"]
        ).to_numpy(dtype=np.float64)

        d_fill = (
            paired["fill_rate"]
            - paired["normal_fill_rate"]
        ).to_numpy(dtype=np.float64)

        d_lost = (
            paired["lost_units"]
            - paired["normal_lost_units"]
        ).to_numpy(dtype=np.float64)

        d_holding = (
            paired["holding_cost"]
            - paired["normal_holding_cost"]
        ).to_numpy(dtype=np.float64)

        delta_rows.append(
            {
                "experiment_id": eid,

                "delta_profit_vs_normal_mean":
                    float(np.mean(d_profit)),
                "delta_profit_vs_normal_ci95":
                    ci95(d_profit),

                "delta_fill_vs_normal_mean":
                    float(np.mean(d_fill)),
                "delta_fill_vs_normal_ci95":
                    ci95(d_fill),

                "delta_lost_vs_normal_mean":
                    float(np.mean(d_lost)),
                "delta_lost_vs_normal_ci95":
                    ci95(d_lost),

                "delta_holding_vs_normal_mean":
                    float(np.mean(d_holding)),
                "delta_holding_vs_normal_ci95":
                    ci95(d_holding),
            }
        )

    return summary_df.merge(
        pd.DataFrame(delta_rows),
        on="experiment_id",
        how="left",
    )


def print_compact_summary(summary_df):
    view = summary_df[
        [
            "experiment_id",
            "A_state",
            "U_state",
            "duration",
            "profit_mean",
            "fill_rate_mean",
            "lost_units_mean",
            "stress_fill_rate_mean",
            "stress_lost_units_mean",
            "stress_max_supplier_backlog_mean",
            "stress_max_transport_backlog_mean",
            "delta_profit_vs_normal_mean",
        ]
    ].copy()

    money_cols = [
        "profit_mean",
        "delta_profit_vs_normal_mean",
    ]

    unit_cols = [
        "lost_units_mean",
        "stress_lost_units_mean",
        "stress_max_supplier_backlog_mean",
        "stress_max_transport_backlog_mean",
    ]

    for c in money_cols:
        view[c] = view[c].map(
            lambda x: f"{x:,.0f}"
        )

    for c in unit_cols:
        view[c] = view[c].map(
            lambda x: f"{x:,.1f}"
        )

    for c in [
        "fill_rate_mean",
        "stress_fill_rate_mean",
    ]:
        view[c] = view[c].map(
            lambda x: f"{x:.4f}"
        )

    print("\n" + "=" * 150)
    print("GUIDE-ALIGNED FORCED-DISRUPTION SUMMARY")
    print("=" * 150)
    print(
        view.to_string(
            index=False
        )
    )


def monotone(values, direction):
    vals = [
        float(x)
        for x in values
    ]

    if direction == "up":
        return all(
            vals[i] <= vals[i + 1] + 1e-12
            for i in range(len(vals) - 1)
        )

    if direction == "down":
        return all(
            vals[i] >= vals[i + 1] - 1e-12
            for i in range(len(vals) - 1)
        )

    raise ValueError(direction)


def diagnostic_checks(summary_df):
    """
    Directional diagnostics.

    These are warnings rather than hard assertions. Mild shocks can be fully
    absorbed by safety stock, and small profit increases can occur when lower
    inventory reduces holding cost. The important checks are:
      - longer disruption should not systematically reduce backlog/lost sales;
      - stronger supplier restriction should create more backlog at comparable
        duration;
      - complete disruptions should eventually damage service/profit.
    """

    s = summary_df.set_index(
        "experiment_id"
    )

    checks = []

    def add_check(name, ok, values):
        checks.append(
            (
                name,
                bool(ok),
                [
                    float(v)
                    for v in values
                ],
            )
        )

    # ------------------------------------------------------------------
    # Duration effect within supplier states.
    # ------------------------------------------------------------------
    for group, scenario, durations in [
        ("S1", "A1", (3, 7, 14)),
        ("S2", "A2", (3, 7, 14)),
        ("S3", "A3", (1, 3, 5, 7)),
    ]:
        ids = [
            f"{group}_{scenario}_{d}d"
            for d in durations
        ]

        backlog = [
            s.loc[
                eid,
                "stress_max_supplier_backlog_mean",
            ]
            for eid in ids
        ]

        lost = [
            s.loc[
                eid,
                "lost_units_mean",
            ]
            for eid in ids
        ]

        add_check(
            f"{scenario}: supplier backlog nondecreasing with duration",
            monotone(backlog, "up"),
            backlog,
        )

        add_check(
            f"{scenario}: lost sales nondecreasing with duration",
            monotone(lost, "up"),
            lost,
        )

    # ------------------------------------------------------------------
    # Duration effect within transport states.
    # ------------------------------------------------------------------
    for group, scenario, durations in [
        ("T1", "U1", (1, 3, 5)),
        ("T2", "U2", (1, 3, 5)),
        ("T3", "U3", (1, 3, 5, 7)),
    ]:
        ids = [
            f"{group}_{scenario}_{d}d"
            for d in durations
        ]

        lost = [
            s.loc[
                eid,
                "lost_units_mean",
            ]
            for eid in ids
        ]

        add_check(
            f"{scenario}: lost sales nondecreasing with duration",
            monotone(lost, "up"),
            lost,
        )

    # ------------------------------------------------------------------
    # Supplier severity at common durations 3 and 7.
    # A1 -> A2 -> A3 should create nondecreasing supplier backlog.
    # ------------------------------------------------------------------
    for d in (3, 7):
        ids = [
            f"S1_A1_{d}d",
            f"S2_A2_{d}d",
            f"S3_A3_{d}d",
        ]

        backlog = [
            s.loc[
                eid,
                "stress_max_supplier_backlog_mean",
            ]
            for eid in ids
        ]

        lost = [
            s.loc[
                eid,
                "lost_units_mean",
            ]
            for eid in ids
        ]

        add_check(
            f"Supplier severity at {d}d: backlog A1<=A2<=A3",
            monotone(backlog, "up"),
            backlog,
        )

        add_check(
            f"Supplier severity at {d}d: lost sales A1<=A2<=A3",
            monotone(lost, "up"),
            lost,
        )

    # ------------------------------------------------------------------
    # Transport severity at common durations 1,3,5.
    # ------------------------------------------------------------------
    for d in (1, 3, 5):
        ids = [
            f"T1_U1_{d}d",
            f"T2_U2_{d}d",
            f"T3_U3_{d}d",
        ]

        lost = [
            s.loc[
                eid,
                "lost_units_mean",
            ]
            for eid in ids
        ]

        add_check(
            f"Transport severity at {d}d: lost U1<=U2<=U3",
            monotone(lost, "up"),
            lost,
        )

    # ------------------------------------------------------------------
    # Compound checks.
    # ------------------------------------------------------------------
    c1 = [
        s.loc[
            f"C1_A2_U2_{d}d",
            "lost_units_mean",
        ]
        for d in (3, 5)
    ]
    add_check(
        "C1 A2+U2: lost sales nondecreasing 3d->5d",
        monotone(c1, "up"),
        c1,
    )

    c2 = [
        s.loc[
            f"C2_A3_U3_{d}d",
            "lost_units_mean",
        ]
        for d in (1, 3)
    ]
    add_check(
        "C2 A3+U3: lost sales nondecreasing 1d->3d",
        monotone(c2, "up"),
        c2,
    )

    # Complete-shutdown / stop should have a visible effect at longest guide
    # duration. Do not impose arbitrary numeric thresholds.
    normal_profit = float(
        s.loc[
            "N0_Normal",
            "profit_mean",
        ]
    )

    normal_fill = float(
        s.loc[
            "N0_Normal",
            "fill_rate_mean",
        ]
    )

    for eid in [
        "S3_A3_7d",
        "T3_U3_7d",
    ]:
        p = float(
            s.loc[
                eid,
                "profit_mean",
            ]
        )
        f = float(
            s.loc[
                eid,
                "fill_rate_mean",
            ]
        )

        add_check(
            f"{eid}: profit below Normal",
            p < normal_profit,
            [normal_profit, p],
        )

        add_check(
            f"{eid}: fill rate below Normal",
            f < normal_fill,
            [normal_fill, f],
        )

    print("\n" + "=" * 150)
    print("DIRECTIONAL SANITY DIAGNOSTICS")
    print("=" * 150)

    pass_count = 0

    for name, ok, values in checks:
        mark = "PASS" if ok else "CHECK"

        print(
            f"[{mark:5s}] {name}"
            f" | values={values}"
        )

        if ok:
            pass_count += 1

    print(
        f"\nDiagnostics passed: "
        f"{pass_count}/{len(checks)}"
    )

    if pass_count == len(checks):
        print(
            "Overall diagnostic: PASS. "
            "The environment shows coherent severity/duration responses "
            "under the guide-aligned forced-event design."
        )
    else:
        print(
            "Overall diagnostic: CHECK. "
            "Do not tune parameters automatically. Inspect which response "
            "metric is weak/reversed and determine whether inventory buffering "
            "provides a legitimate explanation."
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Guide-aligned forced disruption and duration validation."
        )
    )

    parser.add_argument(
        "--reps",
        type=int,
        default=100,
        help="Replications per experiment group. Default: 100.",
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=90,
        help=(
            "Total simulated days per replication, including recovery. "
            "Default: 90."
        ),
    )

    parser.add_argument(
        "--base-seed",
        type=int,
        default=20260817,
        help="Base replication seed. Default: 20260817.",
    )

    parser.add_argument(
        "--action",
        type=int,
        default=0,
        help=(
            "Fixed diagnostic action. "
            "Default: 0 = (l=0,e=0,g=0)."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="validation_results",
        help="Directory for CSV outputs.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.reps <= 0:
        raise ValueError(
            "--reps must be positive."
        )

    if args.horizon <= 0:
        raise ValueError(
            "--horizon must be positive."
        )

    max_duration = max(
        int(exp["duration"])
        for exp in EXPERIMENTS
    )

    if args.horizon <= max_duration:
        raise ValueError(
            f"--horizon must exceed the longest forced disruption "
            f"({max_duration} days) so recovery can be observed."
        )

    probe_env = SupplyChainEnv(
        seed=args.base_seed,
        max_steps=args.horizon,
    )

    if not 0 <= args.action < probe_env.action_space.n:
        raise ValueError(
            f"--action must be in [0, {probe_env.action_space.n - 1}]."
        )

    decoded = probe_env.decode_action(
        args.action
    )

    output_dir = Path(
        args.output_dir
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 120)
    print("GUIDE-ALIGNED FORCED-DISRUPTION / DURATION VALIDATION")
    print("=" * 120)
    print("Experiment groups           :", len(EXPERIMENTS))
    print("  Normal                    : 1")
    print("  Forced-disruption groups  :", len(EXPERIMENTS) - 1)
    print("Replications per group      :", args.reps)
    print("Total horizon               :", args.horizon, "days")
    print("Fixed diagnostic action     :", args.action, "->", decoded)
    print("IMPORTANT                   : P_A and P_U are NOT modified.")
    print(
        "Recovery                    : after each forced event, "
        "A/U are set to A0/U0."
    )

    rows = []

    total_runs = len(EXPERIMENTS) * args.reps

    print(
        "Total replications          :",
        total_runs,
    )

    for idx, exp in enumerate(EXPERIMENTS, start=1):
        eid = experiment_id(exp)

        print(
            f"\n[{idx:02d}/{len(EXPERIMENTS)}] "
            f"{eid:18s} "
            f"A{exp['A_state']}/U{exp['U_state']} "
            f"duration={exp['duration']}d ..."
        )

        for rep in range(args.reps):
            seed = args.base_seed + rep

            result = run_one_replication(
                exp=exp,
                seed=seed,
                horizon=args.horizon,
                action=args.action,
            )

            rows.append(
                result
            )

        print("  complete")

    raw_df = pd.DataFrame(
        rows
    )

    summary_df = build_summary(
        raw_df
    )

    summary_df = add_paired_normal_deltas(
        raw_df,
        summary_df,
    )

    raw_path = (
        output_dir
        / "guide_forced_disruption_raw.csv"
    )

    summary_path = (
        output_dir
        / "guide_forced_disruption_summary.csv"
    )

    raw_df.to_csv(
        raw_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    print_compact_summary(
        summary_df
    )

    diagnostic_checks(
        summary_df
    )

    print("\n" + "=" * 120)
    print("FILES WRITTEN")
    print("=" * 120)
    print(
        "Raw results    :",
        os.path.abspath(raw_path),
    )
    print(
        "Summary results:",
        os.path.abspath(summary_path),
    )

    print(
        "\nNext interpretation:"
        "\n  - A1/A2 may first appear as supplier backlog rather than lost sales."
        "\n  - U1/U2 may be absorbed by safety stock for short events."
        "\n  - Duration effects should be inspected within the same state."
        "\n  - Complete events A3/U3 should eventually reduce service/profit."
        "\n  - Parameter changes must be justified by model realism, not by a "
        "desire to produce a preferred RL ranking."
    )


if __name__ == "__main__":
    main()
