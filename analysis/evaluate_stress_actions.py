"""
evaluate_stress_actions.py

Focused diagnostic:
Compare deterministic action choices during the FIRST 7 DAYS under:
1) forced Normal A0/U0
2) forced Supplier shutdown A3/U0
3) forced Transport stop A0/U3

Uses the already-trained enriched checkpoints.
No training. No parameter tuning.

Outputs:
evaluation_results/
    stress_action_raw.csv
    stress_action_by_train_seed.csv
    stress_action_algorithm_summary.csv
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
Q_AGENT_CLASSES = {
    "dqn": DQNAgent,
    "ddqn": DDQNAgent,
    "dueling": DuelingDQNAgent,
    "d3qn": D3QNAgent,
}
TRAIN_SEEDS = [42, 43, 44, 45, 46]

SCENARIOS = [
    ("Normal_7d", 0, 0),
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
        raise FileNotFoundError(path)

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


def run_first_7_days(agent, algorithm, train_seed, eval_seed, scenario, A, U):
    env = SupplyChainEnv(seed=eval_seed, max_steps=90)

    # Force scenario before reset so the first observation contains it.
    env.set_disruption_override(A_t=A, U_t=U)
    obs, _ = env.reset(seed=eval_seed)

    action_counts = np.zeros(18, dtype=np.int64)

    total_profit = 0.0
    total_lost = 0.0
    total_demand = 0.0
    total_sales = 0.0
    total_expedite = 0.0

    for day in range(7):
        mask = env.action_masks()
        action = int(agent.predict(obs, mask=mask))

        if not mask[action]:
            raise RuntimeError(
                f"Invalid action {action} selected by "
                f"{algorithm} seed={train_seed} in {scenario}."
            )

        obs, reward, terminated, truncated, info = env.step(action)

        action_counts[action] += 1
        total_profit += float(info.get("profit", 0.0))
        total_lost += float(info.get("lost_units", 0.0))
        total_demand += float(info.get("demand_units", 0.0))
        total_sales += float(info.get("sales_units", 0.0))
        total_expedite += float(info.get("expedite_cost", 0.0))

        if terminated or truncated:
            break

    row = {
        "algorithm": algorithm,
        "train_seed": train_seed,
        "eval_seed": eval_seed,
        "scenario": scenario,
        "A": A,
        "U": U,
        "days": int(action_counts.sum()),
        "profit_7d": total_profit,
        "lost_units_7d": total_lost,
        "fill_rate_7d": (
            total_sales / total_demand
            if total_demand > 0 else 1.0
        ),
        "expedite_cost_7d": total_expedite,
    }

    for a in range(18):
        row[f"action_{a}_days"] = int(action_counts[a])

    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--eval-reps", type=int, default=20)
    p.add_argument("--base-eval-seed", type=int, default=20260818)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--output-dir", default="evaluation_results")
    args = p.parse_args()

    device = resolve_device(args.device)

    probe = SupplyChainEnv(seed=args.base_eval_seed)
    state_dim = int(probe.observation_space.shape[0])
    action_dim = int(probe.action_space.n)

    if state_dim != 94 or action_dim != 18:
        raise RuntimeError(
            f"Unexpected environment: state_dim={state_dim}, action_dim={action_dim}"
        )

    rows = []

    print("=" * 100)
    print("STRESS-WINDOW ACTION DIAGNOSTIC")
    print("=" * 100)
    print("Scenarios:", [s[0] for s in SCENARIOS])
    print("Train seeds:", TRAIN_SEEDS)
    print("Eval reps:", args.eval_reps)
    print("Policy: deterministic, NO training")

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
                    rows.append(
                        run_first_7_days(
                            agent,
                            algorithm,
                            train_seed,
                            eval_seed,
                            scenario,
                            A,
                            U,
                        )
                    )
                print(f"  {scenario:20s} complete")

    raw = pd.DataFrame(rows)

    action_cols = [f"action_{a}_days" for a in range(18)]

    by_seed = (
        raw.groupby(
            ["algorithm", "train_seed", "scenario"],
            as_index=False,
        )[
            ["profit_7d", "lost_units_7d", "fill_rate_7d",
             "expedite_cost_7d"] + action_cols
        ]
        .mean()
    )

    # Convert mean action days over reps to shares over the 7-day window.
    for a in range(18):
        by_seed[f"action_{a}_share"] = (
            by_seed[f"action_{a}_days"] / 7.0
        )

    # Dominant action per train seed / scenario.
    share_cols = [f"action_{a}_share" for a in range(18)]
    by_seed["dominant_action"] = (
        by_seed[share_cols]
        .to_numpy()
        .argmax(axis=1)
    )
    by_seed["dominant_share"] = (
        by_seed[share_cols]
        .max(axis=1)
    )

    # Algorithm-level mean shares across independent training seeds.
    summary_cols = [
        "profit_7d",
        "lost_units_7d",
        "fill_rate_7d",
        "expedite_cost_7d",
    ] + share_cols

    summary = (
        by_seed.groupby(
            ["algorithm", "scenario"],
            as_index=False,
        )[summary_cols]
        .mean()
    )

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    raw_path = outdir / "stress_action_raw.csv"
    seed_path = outdir / "stress_action_by_train_seed.csv"
    summary_path = outdir / "stress_action_algorithm_summary.csv"

    raw.to_csv(raw_path, index=False, encoding="utf-8-sig")
    by_seed.to_csv(seed_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print("DOMINANT ACTION BY TRAINING SEED")
    print("=" * 100)

    view = by_seed[
        [
            "algorithm",
            "train_seed",
            "scenario",
            "dominant_action",
            "dominant_share",
            "profit_7d",
            "fill_rate_7d",
            "lost_units_7d",
            "expedite_cost_7d",
        ]
    ].copy()

    view["dominant_share"] = view["dominant_share"].round(3)
    view["profit_7d"] = view["profit_7d"].round(0)
    view["fill_rate_7d"] = view["fill_rate_7d"].round(5)
    view["lost_units_7d"] = view["lost_units_7d"].round(1)
    view["expedite_cost_7d"] = view["expedite_cost_7d"].round(0)

    print(view.to_string(index=False))

    print("\nFiles written:")
    print(raw_path)
    print(seed_path)
    print(summary_path)


if __name__ == "__main__":
    main()
