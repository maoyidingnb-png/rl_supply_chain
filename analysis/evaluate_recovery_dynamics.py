"""
evaluate_recovery_dynamics.py

Focused time-resolved diagnostic for the trained RL policies.

Purpose
-------
The 7-day stress-window test showed that customer fill rate can remain ~1.0
during the shock itself because existing inventory buffers absorb the disruption.
This script therefore follows each policy through:

    Days 1-7   : forced severe disruption
    Days 8-30  : forced A0/U0 recovery

and records DAILY decisions and operating outcomes.

Scenarios
---------
1) Supplier_A3_7d : A3/U0 for 7 days, then A0/U0
2) Transport_U3_7d: A0/U3 for 7 days, then A0/U0

No training. No parameter tuning.

Outputs
-------
evaluation_results/
    recovery_dynamics_raw.csv
    recovery_dynamics_daily.csv
    recovery_dynamics_phase_summary.csv
"""

import argparse
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
TRAIN_SEEDS = [42, 43, 44, 45, 46]

Q_AGENT_CLASSES = {
    "dqn": DQNAgent,
    "ddqn": DDQNAgent,
    "dueling": DuelingDQNAgent,
    "d3qn": D3QNAgent,
}

SCENARIOS = [
    ("Supplier_A3_7d", 3, 0),
    ("Transport_U3_7d", 0, 3),
]


def resolve_device(requested):
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    return requested


def checkpoint_path(checkpoint_dir, algorithm, seed):
    return Path(checkpoint_dir) / f"{algorithm}_enriched_seed{seed}.pth"


def load_agent(algorithm, seed, state_dim, action_dim, checkpoint_dir, device):
    path = checkpoint_path(checkpoint_dir, algorithm, seed)

    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")

    if algorithm == "ppo":
        agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
        )
        agent.load(str(path), load_optimizer=False)
        agent.net.eval()
    else:
        agent = Q_AGENT_CLASSES[algorithm](
            state_dim=state_dim,
            action_dim=action_dim,
            epsilon=0.0,
            epsilon_min=0.0,
            epsilon_decay=1.0,
            device=device,
        )
        agent.load(str(path), load_optimizer=False)
        agent.policy_net.eval()

    return agent


def get_float(info, key, default=0.0):
    v = info.get(key, default)
    if v is None:
        return float(default)
    return float(v)


def phase_name(day):
    if 1 <= day <= 7:
        return "stress_d1_7"
    if 8 <= day <= 14:
        return "recovery_d8_14"
    return "recovery_d15_30"


def run_one(agent, algorithm, train_seed, eval_seed, scenario, A, U, horizon):
    env = SupplyChainEnv(seed=eval_seed, max_steps=horizon)

    # First observation already contains the severe disruption.
    env.set_disruption_override(A_t=A, U_t=U)
    obs, _ = env.reset(seed=eval_seed)

    rows = []

    for day in range(1, horizon + 1):
        # Recovery begins before the day-8 decision.
        if day == 8:
            env.set_disruption_override(A_t=0, U_t=0)

        mask = env.action_masks()
        action = int(agent.predict(obs, mask=mask))

        if not bool(mask[action]):
            raise RuntimeError(
                f"Invalid action {action} selected by "
                f"{algorithm} seed={train_seed} day={day} {scenario}"
            )

        l_t, e_t, g_t = env.decode_action(action)

        obs, reward, terminated, truncated, info = env.step(action)

        demand = get_float(info, "demand_units")
        sales = get_float(info, "sales_units")
        fill = sales / demand if demand > 0 else 1.0

        rows.append(
            {
                "algorithm": algorithm,
                "train_seed": int(train_seed),
                "eval_seed": int(eval_seed),
                "scenario": scenario,
                "day": int(day),
                "phase": phase_name(day),
                "action": int(action),
                "l": int(l_t),
                "e_requested": int(e_t),
                "g": int(g_t),
                "A_t": int(round(get_float(info, "A_t"))),
                "U_t": int(round(get_float(info, "U_t"))),
                "profit": get_float(info, "profit"),
                "demand_units": demand,
                "sales_units": sales,
                "lost_units": get_float(info, "lost_units"),
                "fill_rate": float(fill),
                "expedite_cost": get_float(info, "expedite_cost"),
            }
        )

        if terminated or truncated:
            break

    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--eval-reps", type=int, default=20)
    p.add_argument("--base-eval-seed", type=int, default=20260818)
    p.add_argument("--horizon", type=int, default=30)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--output-dir", default="evaluation_results")
    args = p.parse_args()

    if args.horizon < 15:
        raise ValueError("--horizon should be at least 15 days.")

    device = resolve_device(args.device)

    probe = SupplyChainEnv(seed=args.base_eval_seed, max_steps=args.horizon)
    state_dim = int(probe.observation_space.shape[0])
    action_dim = int(probe.action_space.n)

    if state_dim != 94 or action_dim != 18:
        raise RuntimeError(
            f"Unexpected environment: state_dim={state_dim}, action_dim={action_dim}"
        )

    print("=" * 105)
    print("TIME-RESOLVED SHOCK / RECOVERY DIAGNOSTIC")
    print("=" * 105)
    print("Algorithms :", ALGORITHMS)
    print("Train seeds:", TRAIN_SEEDS)
    print("Eval reps  :", args.eval_reps)
    print("Horizon    :", args.horizon)
    print("Shock      : days 1-7")
    print("Recovery   : days 8-30 forced A0/U0")
    print("Policy     : deterministic, NO training")

    rows = []

    for algorithm in ALGORITHMS:
        for train_seed in TRAIN_SEEDS:
            agent = load_agent(
                algorithm,
                train_seed,
                state_dim,
                action_dim,
                args.checkpoint_dir,
                device,
            )

            print(f"\n{algorithm.upper():8s} seed={train_seed}")

            for scenario, A, U in SCENARIOS:
                for rep in range(args.eval_reps):
                    eval_seed = args.base_eval_seed + rep

                    rows.extend(
                        run_one(
                            agent,
                            algorithm,
                            train_seed,
                            eval_seed,
                            scenario,
                            A,
                            U,
                            args.horizon,
                        )
                    )

                print(f"  {scenario:22s} complete")

    raw = pd.DataFrame(rows)

    # Daily mean across evaluation reps and training seeds.
    daily = (
        raw.groupby(
            ["algorithm", "scenario", "day"],
            as_index=False,
        )
        .agg(
            profit_mean=("profit", "mean"),
            fill_rate_mean=("fill_rate", "mean"),
            lost_units_mean=("lost_units", "mean"),
            expedite_cost_mean=("expedite_cost", "mean"),
            mean_l=("l", "mean"),
            mean_e_requested=("e_requested", "mean"),
            mean_g=("g", "mean"),
        )
    )

    # Phase-level operational and action summaries.
    phase_basic = (
        raw.groupby(
            ["algorithm", "train_seed", "scenario", "phase"],
            as_index=False,
        )
        .agg(
            profit=("profit", "sum"),
            demand_units=("demand_units", "sum"),
            sales_units=("sales_units", "sum"),
            lost_units=("lost_units", "sum"),
            expedite_cost=("expedite_cost", "sum"),
            e_requested_share=("e_requested", "mean"),
            l_mean=("l", "mean"),
            g_mean=("g", "mean"),
        )
    )

    phase_basic["fill_rate"] = np.where(
        phase_basic["demand_units"] > 0,
        phase_basic["sales_units"] / phase_basic["demand_units"],
        1.0,
    )

    # Average evaluation replications within each train seed first.
    by_seed = (
        phase_basic.groupby(
            ["algorithm", "train_seed", "scenario", "phase"],
            as_index=False,
        )
        .mean(numeric_only=True)
    )

    # Then summarize across the 5 independently trained seeds.
    phase_summary = (
        by_seed.groupby(
            ["algorithm", "scenario", "phase"],
            as_index=False,
        )
        .agg(
            profit_mean=("profit", "mean"),
            fill_rate_mean=("fill_rate", "mean"),
            lost_units_mean=("lost_units", "mean"),
            expedite_cost_mean=("expedite_cost", "mean"),
            e_requested_share=("e_requested_share", "mean"),
            l_mean=("l_mean", "mean"),
            g_mean=("g_mean", "mean"),
        )
    )

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw_path = outdir / "recovery_dynamics_raw.csv"
    daily_path = outdir / "recovery_dynamics_daily.csv"
    phase_path = outdir / "recovery_dynamics_phase_summary.csv"

    raw.to_csv(raw_path, index=False, encoding="utf-8-sig")
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    phase_summary.to_csv(phase_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 140)
    print("PHASE SUMMARY")
    print("=" * 140)

    view = phase_summary.copy()
    for c in ["profit_mean", "expedite_cost_mean"]:
        view[c] = view[c].round(0)
    for c in ["fill_rate_mean", "e_requested_share", "l_mean", "g_mean"]:
        view[c] = view[c].round(4)
    view["lost_units_mean"] = view["lost_units_mean"].round(1)

    print(view.to_string(index=False))

    print("\nFiles written:")
    print(raw_path)
    print(daily_path)
    print(phase_path)


if __name__ == "__main__":
    main()
