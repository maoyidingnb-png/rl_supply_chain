"""
validate_stress_severity.py

Purpose
-------
Controlled stress-severity sanity check for the guide-aligned supply-chain MDP.

This script DOES NOT train an RL model and DOES NOT modify P_A / P_U.

Default experiment:
    - 90 simulated days per replication
    - Days 1-30: force one disruption scenario
    - Days 31-90: force Normal (A0/U0)
    - 100 replications per scenario
    - Same replication seed schedule across scenarios
    - Fixed diagnostic policy: action 0 = (l=0, e=0, g=0)

Why use a fixed policy?
-----------------------
This is an environment sanity test, not an algorithm comparison.  We want to
check whether stronger exogenous disruptions actually cause worse operational
outcomes under the same decision rule.

Important interpretation
------------------------
The fixed scenario override is used only for controlled stress testing.
It does not change the environment's Markov transition matrices.  Natural
Markov evaluation should be performed separately later.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from env import SupplyChainEnv


SCENARIOS = [
    ("Normal",             0, 0),
    ("Supplier_Mild",      1, 0),
    ("Supplier_Severe",    2, 0),
    ("Supplier_Shutdown",  3, 0),
    ("Transport_Mild",     0, 1),
    ("Transport_Severe",   0, 2),
    ("Transport_Stop",     0, 3),
    ("Compound_Severe",    2, 2),
    ("Extreme_Compound",   3, 3),
]


def ci95(values):
    """Normal-approximation 95% CI half-width."""
    x = np.asarray(values, dtype=np.float64)

    if x.size <= 1:
        return 0.0

    return 1.96 * float(
        np.std(x, ddof=1) / np.sqrt(x.size)
    )


def run_one_replication(
    scenario_name,
    A_state,
    U_state,
    seed,
    horizon,
    stress_days,
    action,
):
    """
    Run one controlled replication.

    Days [1, stress_days]:
        A/U are fixed at the scenario values.

    Days [stress_days+1, horizon]:
        A/U are fixed at Normal A0/U0.

    P_A and P_U are never edited.
    """

    env = SupplyChainEnv(
        seed=seed,
        max_steps=horizon,
    )

    decoded = env.decode_action(action)

    # Force disruption before reset so the first observation also contains
    # the intended scenario state.
    env.set_disruption_override(
        A_t=A_state,
        U_t=U_state,
    )

    obs, _ = env.reset(
        seed=seed
    )

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

    steps = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):

        # At the start of the first post-stress decision day, switch to Normal.
        # For Normal scenario this simply keeps A0/U0 throughout.
        if steps == stress_days:
            env.set_disruption_override(
                A_t=0,
                U_t=0,
            )

        mask = env.action_masks()

        if not bool(mask[action]):
            raise RuntimeError(
                f"Diagnostic action {action}={decoded} is invalid "
                f"in scenario={scenario_name}, A={env.A_t}, U={env.U_t}. "
                "Use an action valid under every scenario; action 0 is recommended."
            )

        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        # Episode totals.
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

        # Split stress / recovery periods.
        if steps < stress_days:
            prefix = "stress"
        else:
            prefix = "recovery"

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

        supplier_backlog_history.append(
            float(info["supplier_backlog_units"])
        )
        capacity_backlog_history.append(
            float(info["capacity_backlog_units"])
        )
        transport_backlog_history.append(
            float(info["transport_backlog_units"])
        )
        pipeline_history.append(
            float(info["pipeline_units"])
        )
        cw_inventory_history.append(
            float(info["cw_inventory_total"])
        )
        fw_inventory_history.append(
            float(info["fw_inventory_total"])
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

    result = {
        "scenario": scenario_name,
        "A_state": int(A_state),
        "U_state": int(U_state),
        "seed": int(seed),
        "horizon": int(horizon),
        "stress_days": int(stress_days),
        "action": int(action),
        "l_t": int(decoded[0]),
        "e_t": int(decoded[1]),
        "g_t": int(decoded[2]),
        "steps": int(steps),

        **totals,

        "fill_rate": float(fill_rate),
        "stress_fill_rate": float(stress_fill_rate),
        "recovery_fill_rate": float(recovery_fill_rate),

        "mean_supplier_backlog": float(
            np.mean(supplier_backlog_history)
        ),
        "max_supplier_backlog": float(
            np.max(supplier_backlog_history)
        ),
        "end_supplier_backlog": float(
            supplier_backlog_history[-1]
        ),

        "mean_capacity_backlog": float(
            np.mean(capacity_backlog_history)
        ),
        "max_capacity_backlog": float(
            np.max(capacity_backlog_history)
        ),

        "mean_transport_backlog": float(
            np.mean(transport_backlog_history)
        ),
        "max_transport_backlog": float(
            np.max(transport_backlog_history)
        ),

        "mean_pipeline_units": float(
            np.mean(pipeline_history)
        ),
        "mean_cw_inventory": float(
            np.mean(cw_inventory_history)
        ),
        "mean_fw_inventory": float(
            np.mean(fw_inventory_history)
        ),
    }

    return result


def build_summary(raw_df):
    metrics = [
        "profit",
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
        "mean_transport_backlog",
        "mean_pipeline_units",
        "mean_cw_inventory",
        "mean_fw_inventory",
    ]

    rows = []

    for scenario_name, A_state, U_state in SCENARIOS:
        sub = raw_df[
            raw_df["scenario"] == scenario_name
        ]

        row = {
            "scenario": scenario_name,
            "A_state": A_state,
            "U_state": U_state,
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
    Pair each scenario replication with Normal using the same seed.

    This is useful because the environment has a separate demand RNG, so the
    customer-demand path is directly comparable across scenarios for a given
    seed. It should still be described as paired-by-seed rather than perfect
    CRN for every exogenous draw, because disruption-induced ordering changes
    can change how lead-time draws are consumed.
    """

    normal = (
        raw_df[
            raw_df["scenario"] == "Normal"
        ][
            [
                "seed",
                "profit",
                "fill_rate",
                "lost_units",
            ]
        ]
        .rename(
            columns={
                "profit": "normal_profit",
                "fill_rate": "normal_fill_rate",
                "lost_units": "normal_lost_units",
            }
        )
    )

    delta_rows = []

    for scenario_name, _, _ in SCENARIOS:
        sub = raw_df[
            raw_df["scenario"] == scenario_name
        ][
            [
                "seed",
                "profit",
                "fill_rate",
                "lost_units",
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

        delta_rows.append(
            {
                "scenario": scenario_name,

                # Negative is worse than Normal.
                "delta_profit_vs_normal_mean":
                    float(np.mean(d_profit)),
                "delta_profit_vs_normal_ci95":
                    ci95(d_profit),

                # Negative is worse than Normal.
                "delta_fill_vs_normal_mean":
                    float(np.mean(d_fill)),
                "delta_fill_vs_normal_ci95":
                    ci95(d_fill),

                # Positive is worse than Normal.
                "delta_lost_vs_normal_mean":
                    float(np.mean(d_lost)),
                "delta_lost_vs_normal_ci95":
                    ci95(d_lost),
            }
        )

    delta_df = pd.DataFrame(delta_rows)

    return summary_df.merge(
        delta_df,
        on="scenario",
        how="left",
    )


def print_compact_summary(summary_df):
    cols = [
        "scenario",
        "profit_mean",
        "fill_rate_mean",
        "lost_units_mean",
        "stress_fill_rate_mean",
        "stress_lost_units_mean",
        "max_supplier_backlog_mean",
        "delta_profit_vs_normal_mean",
    ]

    view = summary_df[cols].copy()

    view["profit_mean"] = (
        view["profit_mean"]
        .map(lambda x: f"{x:,.0f}")
    )

    view["fill_rate_mean"] = (
        view["fill_rate_mean"]
        .map(lambda x: f"{x:.4f}")
    )

    view["lost_units_mean"] = (
        view["lost_units_mean"]
        .map(lambda x: f"{x:,.1f}")
    )

    view["stress_fill_rate_mean"] = (
        view["stress_fill_rate_mean"]
        .map(lambda x: f"{x:.4f}")
    )

    view["stress_lost_units_mean"] = (
        view["stress_lost_units_mean"]
        .map(lambda x: f"{x:,.1f}")
    )

    view["max_supplier_backlog_mean"] = (
        view["max_supplier_backlog_mean"]
        .map(lambda x: f"{x:,.1f}")
    )

    view["delta_profit_vs_normal_mean"] = (
        view["delta_profit_vs_normal_mean"]
        .map(lambda x: f"{x:,.0f}")
    )

    print("\n" + "=" * 120)
    print("STRESS SEVERITY SUMMARY")
    print("=" * 120)

    print(
        view.to_string(
            index=False
        )
    )


def check_monotonicity(summary_df):
    """
    Diagnostic warnings only.
    Do NOT fail the script merely because a stochastic mean is not perfectly
    monotone.
    """

    s = summary_df.set_index(
        "scenario"
    )

    checks = []

    # Supplier severity: Normal -> A1 -> A2 -> A3
    supplier_order = [
        "Normal",
        "Supplier_Mild",
        "Supplier_Severe",
        "Supplier_Shutdown",
    ]

    # Transport severity: Normal -> U1 -> U2 -> U3
    transport_order = [
        "Normal",
        "Transport_Mild",
        "Transport_Severe",
        "Transport_Stop",
    ]

    def nonincreasing(names, metric):
        vals = [
            float(s.loc[name, metric])
            for name in names
        ]
        return all(
            vals[i] >= vals[i + 1]
            for i in range(len(vals) - 1)
        ), vals

    def nondecreasing(names, metric):
        vals = [
            float(s.loc[name, metric])
            for name in names
        ]
        return all(
            vals[i] <= vals[i + 1]
            for i in range(len(vals) - 1)
        ), vals

    ok, vals = nonincreasing(
        supplier_order,
        "profit_mean",
    )
    checks.append(
        (
            "Supplier profit decreases with severity",
            ok,
            vals,
        )
    )

    ok, vals = nondecreasing(
        supplier_order,
        "lost_units_mean",
    )
    checks.append(
        (
            "Supplier lost sales increase with severity",
            ok,
            vals,
        )
    )

    ok, vals = nonincreasing(
        transport_order,
        "profit_mean",
    )
    checks.append(
        (
            "Transport profit decreases with severity",
            ok,
            vals,
        )
    )

    ok, vals = nondecreasing(
        transport_order,
        "lost_units_mean",
    )
    checks.append(
        (
            "Transport lost sales increase with severity",
            ok,
            vals,
        )
    )

    extreme_profit = float(
        s.loc["Extreme_Compound", "profit_mean"]
    )
    normal_profit = float(
        s.loc["Normal", "profit_mean"]
    )

    checks.append(
        (
            "Extreme compound profit < Normal profit",
            extreme_profit < normal_profit,
            [normal_profit, extreme_profit],
        )
    )

    extreme_fill = float(
        s.loc["Extreme_Compound", "fill_rate_mean"]
    )
    normal_fill = float(
        s.loc["Normal", "fill_rate_mean"]
    )

    checks.append(
        (
            "Extreme compound fill rate < Normal fill rate",
            extreme_fill < normal_fill,
            [normal_fill, extreme_fill],
        )
    )

    print("\n" + "=" * 120)
    print("SANITY-CHECK DIAGNOSTICS")
    print("=" * 120)

    all_ok = True

    for name, ok, vals in checks:
        mark = "PASS" if ok else "CHECK"
        print(
            f"[{mark:5s}] {name}"
            f" | values={vals}"
        )
        all_ok = all_ok and ok

    if all_ok:
        print(
            "\nOverall diagnostic: PASS. "
            "Stress severity is directionally visible under the fixed baseline."
        )
    else:
        print(
            "\nOverall diagnostic: CHECK. "
            "At least one expected severity ordering is weak or reversed. "
            "Do not change parameters yet; inspect the printed results first."
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Controlled stress-severity validation for SupplyChainEnv."
        )
    )

    parser.add_argument(
        "--reps",
        type=int,
        default=100,
        help="Replications per scenario. Default: 100.",
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=90,
        help="Episode length in days. Default: 90.",
    )

    parser.add_argument(
        "--stress-days",
        type=int,
        default=30,
        help=(
            "Number of initial days for which the scenario is forced. "
            "Remaining days are forced Normal. Default: 30."
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
            "Fixed diagnostic action. Default 0 = (l=0,e=0,g=0)."
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
        raise ValueError("--reps must be positive.")

    if args.horizon <= 0:
        raise ValueError("--horizon must be positive.")

    if not 0 <= args.stress_days <= args.horizon:
        raise ValueError(
            "--stress-days must be between 0 and --horizon."
        )

    # Validate action once.
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
    print("CONTROLLED STRESS-SEVERITY VALIDATION")
    print("=" * 120)
    print("Replications per scenario :", args.reps)
    print("Horizon                   :", args.horizon, "days")
    print("Forced stress window       :", args.stress_days, "days")
    print(
        "Recovery window            :",
        args.horizon - args.stress_days,
        "days forced A0/U0",
    )
    print("Fixed diagnostic action    :", args.action, "->", decoded)
    print("IMPORTANT                  : P_A and P_U are NOT modified.")
    print(
        "Scenario design            : controlled override for validation, "
        "not Natural-Markov evaluation."
    )

    rows = []

    for scenario_name, A_state, U_state in SCENARIOS:
        print(
            f"\nRunning {scenario_name:20s} "
            f"(A{A_state}/U{U_state}) ..."
        )

        for rep in range(args.reps):
            seed = args.base_seed + rep

            result = run_one_replication(
                scenario_name=scenario_name,
                A_state=A_state,
                U_state=U_state,
                seed=seed,
                horizon=args.horizon,
                stress_days=args.stress_days,
                action=args.action,
            )

            rows.append(result)

        print("  complete")

    raw_df = pd.DataFrame(rows)

    summary_df = build_summary(
        raw_df
    )

    summary_df = add_paired_normal_deltas(
        raw_df,
        summary_df,
    )

    raw_path = output_dir / "stress_severity_raw.csv"
    summary_path = output_dir / "stress_severity_summary.csv"

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

    check_monotonicity(
        summary_df
    )

    print("\n" + "=" * 120)
    print("FILES WRITTEN")
    print("=" * 120)
    print("Raw results    :", os.path.abspath(raw_path))
    print("Summary results:", os.path.abspath(summary_path))

    print(
        "\nInterpretation rule:"
        "\n  1) First inspect stress_fill_rate / stress_lost_units."
        "\n  2) Then inspect full-episode profit and backlog."
        "\n  3) If A3/U3 still looks almost identical to Normal, "
        "the inventory/reorder calibration is too protective or the shock "
        "mechanism is too weak for the research question."
        "\n  4) Do NOT tune parameters merely to force a preferred RL ranking."
    )


if __name__ == "__main__":
    main()
