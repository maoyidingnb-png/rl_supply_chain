"""
evaluate_models.py

Formal evaluation for the frozen supply-chain RL environment.

What this script does
---------------------
1) Loads all trained checkpoints:
   DQN / DDQN / Dueling DQN / D3QN / PPO
   training seeds 42,43,44,45,46 by default.

2) Uses deterministic policies during evaluation:
   - Q-learning family: argmax valid Q action
   - PPO: highest-logit valid action
   No learning and no exploration.

3) Evaluates on:
   A. Natural_Markov:
      no forced override; A_t and U_t follow the original P_A / P_U.
   B. Control_Normal:
      forced A0/U0 for the full episode.
   C. Guide-aligned representative forced disruptions:
      A1 14d, A2 14d, A3 7d,
      U1 5d, U2 5d, U3 7d,
      A2+U2 5d, A3+U3 3d,
      followed by forced A0/U0 recovery.

4) Uses the same evaluation seed schedule for every trained model
   (paired / common-random-number style comparison).

5) Statistical summary:
   First averages evaluation replications WITHIN each training seed.
   Then computes algorithm mean and 95% CI ACROSS the independent
   training seeds. This avoids treating environment replications as
   independent trained models.

Outputs
-------
evaluation_results/
    evaluation_raw.csv
    evaluation_by_train_seed.csv
    evaluation_algorithm_summary.csv
    evaluation_action_summary.csv
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from env import SupplyChainEnv
from models import (
    DQNAgent,
    DDQNAgent,
    DuelingDQNAgent,
    D3QNAgent,
    PPOAgent,
)


ALGORITHMS = ["dqn", "ddqn", "dueling", "d3qn", "ppo"]

Q_AGENT_CLASSES = {
    "dqn": DQNAgent,
    "ddqn": DDQNAgent,
    "dueling": DuelingDQNAgent,
    "d3qn": D3QNAgent,
}

DEFAULT_TRAIN_SEEDS = [42, 43, 44, 45, 46]

# Maximum/representative durations from the Stage-2 guide's forced-event table.
SCENARIOS = [
    {
        "scenario": "Natural_Markov",
        "kind": "natural",
        "A": None,
        "U": None,
        "duration": 0,
    },
    {
        "scenario": "Control_Normal",
        "kind": "control",
        "A": 0,
        "U": 0,
        "duration": -1,  # full horizon forced normal
    },
    {
        "scenario": "Supplier_A1_14d",
        "kind": "stress",
        "A": 1,
        "U": 0,
        "duration": 14,
    },
    {
        "scenario": "Supplier_A2_14d",
        "kind": "stress",
        "A": 2,
        "U": 0,
        "duration": 14,
    },
    {
        "scenario": "Supplier_A3_7d",
        "kind": "stress",
        "A": 3,
        "U": 0,
        "duration": 7,
    },
    {
        "scenario": "Transport_U1_5d",
        "kind": "stress",
        "A": 0,
        "U": 1,
        "duration": 5,
    },
    {
        "scenario": "Transport_U2_5d",
        "kind": "stress",
        "A": 0,
        "U": 2,
        "duration": 5,
    },
    {
        "scenario": "Transport_U3_7d",
        "kind": "stress",
        "A": 0,
        "U": 3,
        "duration": 7,
    },
    {
        "scenario": "Compound_A2U2_5d",
        "kind": "stress",
        "A": 2,
        "U": 2,
        "duration": 5,
    },
    {
        "scenario": "Compound_A3U3_3d",
        "kind": "stress",
        "A": 3,
        "U": 3,
        "duration": 3,
    },
]


def resolve_device(requested):
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable.")
    return requested


def checkpoint_path(checkpoint_dir, algorithm, train_seed, training_mode):
    return (
        Path(checkpoint_dir)
        / f"{algorithm}_{training_mode}_seed{train_seed}.pth"
    )


def build_and_load_agent(
    algorithm,
    train_seed,
    state_dim,
    action_dim,
    checkpoint_dir,
    training_mode,
    device,
):
    path = checkpoint_path(
        checkpoint_dir,
        algorithm,
        train_seed,
        training_mode,
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {path}\n"
            "Do not retrain automatically; first verify the filename and folder."
        )

    if algorithm == "ppo":
        agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
        )
        agent.load(
            str(path),
            load_optimizer=False,
        )
        agent.net.eval()
    else:
        cls = Q_AGENT_CLASSES[algorithm]
        agent = cls(
            state_dim=state_dim,
            action_dim=action_dim,
            epsilon=0.0,
            epsilon_min=0.0,
            epsilon_decay=1.0,
            device=device,
        )
        agent.load(
            str(path),
            load_optimizer=False,
        )
        agent.policy_net.eval()
        agent.target_net.eval()

    return agent, path


def safe(info, key, default=0.0):
    value = info.get(key, default)
    if value is None:
        return float(default)
    return float(value)


def configure_scenario_before_reset(env, spec):
    kind = spec["kind"]

    if kind == "natural":
        env.clear_disruption_override()
    elif kind == "control":
        env.set_disruption_override(A_t=0, U_t=0)
    elif kind == "stress":
        env.set_disruption_override(
            A_t=int(spec["A"]),
            U_t=int(spec["U"]),
        )
    else:
        raise ValueError(f"Unknown scenario kind: {kind}")


def maybe_switch_to_recovery(env, spec, steps):
    if spec["kind"] != "stress":
        return

    duration = int(spec["duration"])

    if steps == duration:
        # Immediately put the decision state into Normal and keep it Normal
        # for the remainder of the episode.
        env.set_disruption_override(A_t=0, U_t=0)


def evaluate_one_replication(
    agent,
    algorithm,
    train_seed,
    eval_seed,
    spec,
    horizon,
):
    env = SupplyChainEnv(
        seed=eval_seed,
        max_steps=horizon,
    )

    configure_scenario_before_reset(env, spec)

    obs, _ = env.reset(seed=eval_seed)

    totals = {
        "profit": 0.0,
        "revenue": 0.0,
        "lost_sales_penalty": 0.0,
        "holding_cost": 0.0,
        "holding_cost_cw": 0.0,
        "holding_cost_fw": 0.0,
        "expedite_cost": 0.0,
        "demand_units": 0.0,
        "sales_units": 0.0,
        "lost_units": 0.0,
    }

    stress = {
        "profit": 0.0,
        "demand_units": 0.0,
        "sales_units": 0.0,
        "lost_units": 0.0,
    }

    recovery = {
        "profit": 0.0,
        "demand_units": 0.0,
        "sales_units": 0.0,
        "lost_units": 0.0,
    }

    supplier_backlog = []
    capacity_backlog = []
    transport_backlog = []
    pipeline = []
    cw_inventory = []
    fw_inventory = []

    action_counts = np.zeros(18, dtype=np.int64)
    l_counts = np.zeros(3, dtype=np.int64)
    e_requested_counts = np.zeros(2, dtype=np.int64)
    e_executed_count = 0
    g_counts = np.zeros(3, dtype=np.int64)

    A_counts = np.zeros(4, dtype=np.int64)
    U_counts = np.zeros(4, dtype=np.int64)

    steps = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        maybe_switch_to_recovery(
            env,
            spec,
            steps,
        )

        mask = env.action_masks()

        action = int(
            agent.predict(
                obs,
                mask=mask,
            )
        )

        if not bool(mask[action]):
            raise RuntimeError(
                f"Agent selected invalid action {action} "
                f"in {spec['scenario']}."
            )

        l_t, e_t, g_t = env.decode_action(action)

        (
            next_obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        action_counts[action] += 1
        l_counts[l_t] += 1
        e_requested_counts[e_t] += 1
        g_counts[g_t] += 1
        e_executed_count += int(
            round(safe(info, "executed_e_t", 0.0))
        )

        current_A = int(round(safe(info, "A_t", 0.0)))
        current_U = int(round(safe(info, "U_t", 0.0)))
        A_counts[current_A] += 1
        U_counts[current_U] += 1

        for key in totals:
            totals[key] += safe(info, key, 0.0)

        is_stress_day = (
            spec["kind"] == "stress"
            and steps < int(spec["duration"])
        )

        target = stress if is_stress_day else recovery

        target["profit"] += safe(info, "profit", 0.0)
        target["demand_units"] += safe(info, "demand_units", 0.0)
        target["sales_units"] += safe(info, "sales_units", 0.0)
        target["lost_units"] += safe(info, "lost_units", 0.0)

        supplier_backlog.append(
            safe(info, "supplier_backlog_units", 0.0)
        )
        capacity_backlog.append(
            safe(info, "capacity_backlog_units", 0.0)
        )
        transport_backlog.append(
            safe(info, "transport_backlog_units", 0.0)
        )
        pipeline.append(
            safe(info, "pipeline_units", 0.0)
        )
        cw_inventory.append(
            safe(info, "cw_inventory_total", 0.0)
        )
        fw_inventory.append(
            safe(info, "fw_inventory_total", 0.0)
        )

        obs = next_obs
        steps += 1

    fill_rate = (
        totals["sales_units"] / totals["demand_units"]
        if totals["demand_units"] > 0
        else 1.0
    )

    stress_fill_rate = (
        stress["sales_units"] / stress["demand_units"]
        if stress["demand_units"] > 0
        else np.nan
    )

    recovery_fill_rate = (
        recovery["sales_units"] / recovery["demand_units"]
        if recovery["demand_units"] > 0
        else np.nan
    )

    def mean0(x):
        return float(np.mean(x)) if x else 0.0

    def max0(x):
        return float(np.max(x)) if x else 0.0

    row = {
        "algorithm": algorithm,
        "train_seed": int(train_seed),
        "eval_seed": int(eval_seed),
        "scenario": spec["scenario"],
        "scenario_kind": spec["kind"],
        "forced_A": spec["A"],
        "forced_U": spec["U"],
        "stress_duration": int(spec["duration"]),
        "horizon": int(horizon),
        "steps": int(steps),

        **totals,

        "fill_rate": float(fill_rate),

        "stress_profit": float(stress["profit"]),
        "stress_demand_units": float(stress["demand_units"]),
        "stress_sales_units": float(stress["sales_units"]),
        "stress_lost_units": float(stress["lost_units"]),
        "stress_fill_rate": float(stress_fill_rate)
        if not np.isnan(stress_fill_rate)
        else np.nan,

        "recovery_profit": float(recovery["profit"]),
        "recovery_demand_units": float(recovery["demand_units"]),
        "recovery_sales_units": float(recovery["sales_units"]),
        "recovery_lost_units": float(recovery["lost_units"]),
        "recovery_fill_rate": float(recovery_fill_rate)
        if not np.isnan(recovery_fill_rate)
        else np.nan,

        "mean_supplier_backlog": mean0(supplier_backlog),
        "max_supplier_backlog": max0(supplier_backlog),
        "mean_capacity_backlog": mean0(capacity_backlog),
        "max_capacity_backlog": max0(capacity_backlog),
        "mean_transport_backlog": mean0(transport_backlog),
        "max_transport_backlog": max0(transport_backlog),
        "mean_pipeline_units": mean0(pipeline),
        "mean_cw_inventory": mean0(cw_inventory),
        "mean_fw_inventory": mean0(fw_inventory),

        "requested_expedite_days": int(e_requested_counts[1]),
        "executed_expedite_days": int(e_executed_count),
    }

    for i, count in enumerate(action_counts):
        row[f"action_{i}_days"] = int(count)

    for i, count in enumerate(l_counts):
        row[f"l{i}_days"] = int(count)

    for i, count in enumerate(g_counts):
        row[f"g{i}_days"] = int(count)

    for i, count in enumerate(A_counts):
        row[f"A{i}_days"] = int(count)

    for i, count in enumerate(U_counts):
        row[f"U{i}_days"] = int(count)

    return row


def ci95_across_training_seeds(values):
    """
    95% CI half-width across independently trained seeds.

    For n=5, Student-t critical value with df=4 is 2.776.
    For other n, use a small lookup; fall back to normal 1.96 for n>30.
    """
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]

    n = len(x)
    if n <= 1:
        return 0.0

    tcrit_lookup = {
        2: 12.706,
        3: 4.303,
        4: 3.182,
        5: 2.776,
        6: 2.571,
        7: 2.447,
        8: 2.365,
        9: 2.306,
        10: 2.262,
        11: 2.228,
        12: 2.201,
        13: 2.179,
        14: 2.160,
        15: 2.145,
        16: 2.131,
        17: 2.120,
        18: 2.110,
        19: 2.101,
        20: 2.093,
        21: 2.086,
        22: 2.080,
        23: 2.074,
        24: 2.069,
        25: 2.064,
        26: 2.060,
        27: 2.056,
        28: 2.052,
        29: 2.048,
        30: 2.045,
    }

    tcrit = tcrit_lookup.get(n, 1.96)
    se = np.std(x, ddof=1) / np.sqrt(n)
    return float(tcrit * se)


SUMMARY_METRICS = [
    "profit",
    "revenue",
    "fill_rate",
    "lost_units",
    "lost_sales_penalty",
    "holding_cost",
    "holding_cost_cw",
    "holding_cost_fw",
    "expedite_cost",
    "stress_profit",
    "stress_fill_rate",
    "stress_lost_units",
    "recovery_profit",
    "recovery_fill_rate",
    "recovery_lost_units",
    "mean_supplier_backlog",
    "max_supplier_backlog",
    "mean_capacity_backlog",
    "max_capacity_backlog",
    "mean_transport_backlog",
    "max_transport_backlog",
    "mean_pipeline_units",
    "mean_cw_inventory",
    "mean_fw_inventory",
    "requested_expedite_days",
    "executed_expedite_days",
]


def build_by_train_seed(raw):
    """
    Average environment replications within each independently trained model.
    """
    group_cols = [
        "algorithm",
        "train_seed",
        "scenario",
        "scenario_kind",
        "forced_A",
        "forced_U",
        "stress_duration",
        "horizon",
    ]

    metric_cols = SUMMARY_METRICS.copy()

    action_cols = [
        c for c in raw.columns
        if c.startswith("action_")
        or c.startswith("l")
        and c.endswith("_days")
        or c.startswith("g")
        and c.endswith("_days")
    ]

    metric_cols += [
        c for c in action_cols
        if c not in metric_cols
    ]

    grouped = (
        raw.groupby(
            group_cols,
            dropna=False,
            as_index=False,
        )[metric_cols]
        .mean()
    )

    return grouped


def build_algorithm_summary(by_seed):
    """
    Mean / SD / 95% CI across independent training seeds.
    """
    rows = []

    scenario_order = {
        spec["scenario"]: i
        for i, spec in enumerate(SCENARIOS)
    }

    for (scenario, algorithm), sub in by_seed.groupby(
        ["scenario", "algorithm"],
        sort=False,
    ):
        row = {
            "scenario": scenario,
            "algorithm": algorithm,
            "n_train_seeds": int(sub["train_seed"].nunique()),
        }

        for metric in SUMMARY_METRICS:
            vals = sub[metric].to_numpy(dtype=np.float64)
            finite = vals[np.isfinite(vals)]

            row[f"{metric}_mean"] = (
                float(np.mean(finite))
                if len(finite)
                else np.nan
            )

            row[f"{metric}_std"] = (
                float(np.std(finite, ddof=1))
                if len(finite) > 1
                else 0.0
            )

            row[f"{metric}_ci95"] = (
                ci95_across_training_seeds(finite)
                if len(finite)
                else np.nan
            )

        rows.append(row)

    out = pd.DataFrame(rows)

    out["_scenario_order"] = out["scenario"].map(scenario_order)
    out["_algo_order"] = out["algorithm"].map(
        {a: i for i, a in enumerate(ALGORITHMS)}
    )

    out = (
        out.sort_values(
            ["_scenario_order", "_algo_order"]
        )
        .drop(columns=["_scenario_order", "_algo_order"])
        .reset_index(drop=True)
    )

    return out


def build_action_summary(raw):
    action_cols = [
        f"action_{i}_days"
        for i in range(18)
    ]

    rows = []

    for (scenario, algorithm), sub in raw.groupby(
        ["scenario", "algorithm"],
        sort=False,
    ):
        totals = sub[action_cols].sum(axis=0).to_numpy(dtype=np.float64)
        denom = totals.sum()

        row = {
            "scenario": scenario,
            "algorithm": algorithm,
            "total_decision_days": int(denom),
        }

        if denom > 0:
            freqs = totals / denom
        else:
            freqs = np.zeros(18, dtype=np.float64)

        for a in range(18):
            row[f"action_{a}_share"] = float(freqs[a])

        rows.append(row)

    return pd.DataFrame(rows)


def print_compact_summary(summary):
    view = summary[
        [
            "scenario",
            "algorithm",
            "profit_mean",
            "profit_ci95",
            "fill_rate_mean",
            "fill_rate_ci95",
            "lost_units_mean",
            "lost_units_ci95",
            "expedite_cost_mean",
        ]
    ].copy()

    view["profit_mean"] = view["profit_mean"].map(
        lambda x: f"{x:,.0f}"
    )
    view["profit_ci95"] = view["profit_ci95"].map(
        lambda x: f"{x:,.0f}"
    )
    view["fill_rate_mean"] = view["fill_rate_mean"].map(
        lambda x: f"{x:.5f}"
    )
    view["fill_rate_ci95"] = view["fill_rate_ci95"].map(
        lambda x: f"{x:.5f}"
    )
    view["lost_units_mean"] = view["lost_units_mean"].map(
        lambda x: f"{x:,.1f}"
    )
    view["lost_units_ci95"] = view["lost_units_ci95"].map(
        lambda x: f"{x:,.1f}"
    )
    view["expedite_cost_mean"] = view["expedite_cost_mean"].map(
        lambda x: f"{x:,.0f}"
    )

    print("\n" + "=" * 150)
    print("FORMAL EVALUATION SUMMARY")
    print("mean +/- 95% CI across independent TRAINING seeds")
    print("=" * 150)
    print(view.to_string(index=False))


def check_checkpoints(
    checkpoint_dir,
    train_seeds,
    training_mode,
    algorithms,
):
    missing = []

    for algorithm in algorithms:
        for seed in train_seeds:
            p = checkpoint_path(
                checkpoint_dir,
                algorithm,
                seed,
                training_mode,
            )
            if not p.exists():
                missing.append(str(p))

    if missing:
        print("\nMissing checkpoints:")
        for p in missing:
            print("  ", p)
        raise FileNotFoundError(
            f"{len(missing)} required checkpoints are missing."
        )

    print(
        f"Checkpoint check: PASS "
        f"({len(algorithms) * len(train_seeds)} files found)"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Formal deterministic evaluation of trained RL checkpoints."
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
    )

    parser.add_argument(
        "--training-mode",
        type=str,
        default="enriched",
        choices=["natural", "enriched", "adaptive"],
    )

    parser.add_argument(
        "--algorithms",
        type=str,
        nargs="+",
        default=ALGORITHMS,
        choices=ALGORITHMS,
        help=(
            "Subset of algorithms to evaluate. "
            "Example: --algorithms dueling"
        ),
    )

    parser.add_argument(
        "--train-seeds",
        type=int,
        nargs="+",
        default=DEFAULT_TRAIN_SEEDS,
    )

    parser.add_argument(
        "--eval-reps",
        type=int,
        default=20,
        help=(
            "Environment replications per checkpoint per scenario. "
            "Default 20 => 100 replications per algorithm/scenario "
            "across five training seeds."
        ),
    )

    parser.add_argument(
        "--base-eval-seed",
        type=int,
        default=20260818,
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=90,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Fast compatibility test: all checkpoints, "
            "1 eval replication, Natural_Markov only."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.eval_reps <= 0:
        raise ValueError("--eval-reps must be positive.")

    if args.horizon <= 0:
        raise ValueError("--horizon must be positive.")

    device = resolve_device(args.device)

    check_checkpoints(
        args.checkpoint_dir,
        args.train_seeds,
        args.training_mode,
        args.algorithms,
    )

    probe = SupplyChainEnv(
        seed=args.base_eval_seed,
        max_steps=args.horizon,
    )

    state_dim = int(probe.observation_space.shape[0])
    action_dim = int(probe.action_space.n)

    if state_dim != 94:
        raise RuntimeError(
            f"Expected state_dim=94, got {state_dim}. "
            "You may be evaluating with the wrong env.py."
        )

    if action_dim != 18:
        raise RuntimeError(
            f"Expected action_dim=18, got {action_dim}."
        )

    scenarios = SCENARIOS
    eval_reps = args.eval_reps

    if args.smoke:
        scenarios = [
            spec for spec in SCENARIOS
            if spec["scenario"] == "Natural_Markov"
        ]
        eval_reps = 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 110)
    print("FORMAL RL EVALUATION")
    print("=" * 110)
    print("Algorithms        :", args.algorithms)
    print("Training seeds    :", args.train_seeds)
    print("Training mode     :", args.training_mode)
    print("Evaluation reps   :", eval_reps)
    print("Scenarios         :", [s["scenario"] for s in scenarios])
    print("Horizon           :", args.horizon)
    print("Device            :", device)
    print("Policy            : deterministic; NO exploration / NO training")
    print("P_A / P_U         : unchanged in Natural_Markov")
    print(
        "Statistics        : env reps averaged within train seed, "
        "95% CI across train seeds"
    )

    raw_rows = []

    total_models = len(args.algorithms) * len(args.train_seeds)
    model_idx = 0

    for algorithm in args.algorithms:
        for train_seed in args.train_seeds:
            model_idx += 1

            agent, ckpt = build_and_load_agent(
                algorithm=algorithm,
                train_seed=train_seed,
                state_dim=state_dim,
                action_dim=action_dim,
                checkpoint_dir=args.checkpoint_dir,
                training_mode=args.training_mode,
                device=device,
            )

            print(
                f"\n[{model_idx:02d}/{total_models}] "
                f"{algorithm.upper():8s} seed={train_seed} "
                f"| loaded {ckpt.name}"
            )

            for spec in scenarios:
                for rep in range(eval_reps):
                    eval_seed = (
                        int(args.base_eval_seed)
                        + rep
                    )

                    row = evaluate_one_replication(
                        agent=agent,
                        algorithm=algorithm,
                        train_seed=train_seed,
                        eval_seed=eval_seed,
                        spec=spec,
                        horizon=args.horizon,
                    )

                    raw_rows.append(row)

                print(
                    f"  {spec['scenario']:22s} "
                    f"{eval_reps} reps complete"
                )

    raw = pd.DataFrame(raw_rows)

    by_seed = build_by_train_seed(raw)
    summary = build_algorithm_summary(by_seed)
    actions = build_action_summary(raw)

    raw_path = output_dir / "evaluation_raw.csv"
    seed_path = output_dir / "evaluation_by_train_seed.csv"
    summary_path = output_dir / "evaluation_algorithm_summary.csv"
    action_path = output_dir / "evaluation_action_summary.csv"

    raw.to_csv(
        raw_path,
        index=False,
        encoding="utf-8-sig",
    )

    by_seed.to_csv(
        seed_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    actions.to_csv(
        action_path,
        index=False,
        encoding="utf-8-sig",
    )

    print_compact_summary(summary)

    print("\n" + "=" * 110)
    print("FILES WRITTEN")
    print("=" * 110)
    print("Raw               :", os.path.abspath(raw_path))
    print("By training seed  :", os.path.abspath(seed_path))
    print("Algorithm summary :", os.path.abspath(summary_path))
    print("Action summary    :", os.path.abspath(action_path))

    if args.smoke:
        print(
            "\nSMOKE TEST COMPLETE. "
            "If no exception occurred and all 25 checkpoints loaded, "
            "run the full command without --smoke."
        )


if __name__ == "__main__":
    main()
