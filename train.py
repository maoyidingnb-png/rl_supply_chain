import argparse
import csv
import os
import random

import numpy as np
import torch

from env import SupplyChainEnv
from models import (
    DQNAgent,
    DDQNAgent,
    DuelingDQNAgent,
    D3QNAgent,
    PPOAgent,
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(CURRENT_DIR, "training_logs")
CHECKPOINT_DIR = os.path.join(CURRENT_DIR, "checkpoints")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

Q_AGENT_CLASSES = {
    "dqn": DQNAgent,
    "ddqn": DDQNAgent,
    "dueling": DuelingDQNAgent,
    "d3qn": D3QNAgent,
}
ALL_ALGORITHMS = ["dqn", "ddqn", "dueling", "d3qn", "ppo"]


def set_global_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(device_arg):
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA unavailable; falling back to CPU.")
        return "cpu"
    return device_arg


def episode_seed(base_seed, episode_idx):
    return int(base_seed) * 10000 + int(episode_idx)


# =============================================================================
# Training distribution
# =============================================================================

# [TRAINING DESIGN ASSUMPTION]
# Only used to enrich TRAINING data with rare disruption states.
# P_A / P_U are NOT changed.
RISK_START_STATES = [
    (1, 0),
    (2, 0),
    (3, 0),
    (0, 1),
    (0, 2),
    (0, 3),
    (2, 2),
    (3, 3),
]

ENRICHED_RISK_EPISODE_PROB = 0.30


def choose_initial_disruption(training_mode, ep_seed):
    if training_mode == "natural":
        return 0, 0, False

    if training_mode != "enriched":
        raise ValueError(
            f"Unknown training_mode={training_mode}; "
            "expected 'natural' or 'enriched'."
        )

    rng = np.random.default_rng(int(ep_seed) + 20260817)

    if rng.random() >= ENRICHED_RISK_EPISODE_PROB:
        return 0, 0, False

    idx = int(rng.integers(0, len(RISK_START_STATES)))
    initial_A, initial_U = RISK_START_STATES[idx]

    return int(initial_A), int(initial_U), True


def reset_training_episode(env, ep_seed, training_mode):
    initial_A, initial_U, risk_start_episode = choose_initial_disruption(
        training_mode,
        ep_seed,
    )

    # Force only the initial state.
    env.set_disruption_override(
        A_t=initial_A,
        U_t=initial_U,
    )

    obs, info = env.reset(seed=ep_seed)

    # From the first step onward, the original Markov matrices apply again.
    env.clear_disruption_override()

    return obs, info, initial_A, initial_U, risk_start_episode


# =============================================================================
# Logging
# =============================================================================

def log_header(action_dim):
    header = [
        "algorithm",
        "seed",
        "episode",
        "episode_seed",
        "training_mode",
        "risk_start_episode",
        "initial_A",
        "initial_U",
        "steps",
        "total_reward",
        "profit",
        "revenue",
        "lost_sales_penalty",
        "holding_cost",
        "expedite_cost",
        "demand_units",
        "sales_units",
        "lost_units",
        "fill_rate",
        "mean_train_loss",
        "epsilon",
    ]

    header += [f"A{i}_days" for i in range(4)]
    header += [f"U{i}_days" for i in range(4)]
    header += [f"action_{a}_count" for a in range(action_dim)]

    return header


def build_log_row(
    *,
    algorithm,
    seed,
    episode,
    ep_seed,
    training_mode,
    risk_start_episode,
    initial_A,
    initial_U,
    steps,
    total_reward,
    profit,
    revenue,
    lost_sales_penalty,
    holding_cost,
    expedite_cost,
    demand_units,
    sales_units,
    lost_units,
    mean_train_loss,
    epsilon,
    A_counts,
    U_counts,
    action_counts,
):
    fill_rate = sales_units / demand_units if demand_units > 0 else 1.0

    row = [
        algorithm,
        seed,
        episode,
        ep_seed,
        training_mode,
        int(bool(risk_start_episode)),
        int(initial_A),
        int(initial_U),
        steps,
        total_reward,
        profit,
        revenue,
        lost_sales_penalty,
        holding_cost,
        expedite_cost,
        demand_units,
        sales_units,
        lost_units,
        fill_rate,
        mean_train_loss,
        epsilon,
    ]

    row += A_counts.tolist()
    row += U_counts.tolist()
    row += action_counts.tolist()
    return row


# =============================================================================
# Q-learning helpers
# =============================================================================

def compute_per_step_epsilon_decay(
    epsilon_start,
    epsilon_end,
    total_environment_steps,
    exploration_fraction=0.60,
):
    decay_steps = max(
        1,
        int(total_environment_steps * exploration_fraction),
    )

    return (epsilon_end / epsilon_start) ** (1.0 / decay_steps)


def build_q_agent(
    algorithm,
    state_dim,
    action_dim,
    episodes,
    max_steps,
    device,
):
    agent_cls = Q_AGENT_CLASSES[algorithm]

    epsilon_start = 1.0
    epsilon_end = 0.05
    total_environment_steps = episodes * max_steps

    epsilon_decay = compute_per_step_epsilon_decay(
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        total_environment_steps=total_environment_steps,
        exploration_fraction=0.60,
    )

    return agent_cls(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=1e-3,
        gamma=0.99,
        batch_size=64,
        replay_capacity=50000,
        epsilon=epsilon_start,
        epsilon_min=epsilon_end,
        epsilon_decay=epsilon_decay,
        device=device,
    )


# =============================================================================
# Q-learning training
# =============================================================================

def train_q_agent(
    algorithm,
    seed,
    episodes=300,
    target_update_episodes=10,
    device="cpu",
    print_every=25,
    training_mode="natural",
):
    if algorithm not in Q_AGENT_CLASSES:
        raise ValueError(f"Unknown Q-learning algorithm: {algorithm}")

    set_global_seed(seed)

    env = SupplyChainEnv(seed=seed)
    state_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.n)

    agent = build_q_agent(
        algorithm=algorithm,
        state_dim=state_dim,
        action_dim=action_dim,
        episodes=episodes,
        max_steps=env.max_steps,
        device=device,
    )

    log_path = os.path.join(
        LOG_DIR,
        f"training_log_{algorithm}_{training_mode}_seed{seed}.csv",
    )
    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        f"{algorithm}_{training_mode}_seed{seed}.pth",
    )

    print("\n" + "=" * 90)
    print(
        f"TRAINING {algorithm.upper()} "
        f"| seed={seed} "
        f"| episodes={episodes} "
        f"| state_dim={state_dim} "
        f"| action_dim={action_dim} "
        f"| mode={training_mode} "
        f"| device={device}"
    )
    print("=" * 90)
    print(f"epsilon per-step decay = {agent.epsilon_decay:.8f}")

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(log_header(action_dim))

        for episode_idx in range(episodes):
            ep_seed = episode_seed(seed, episode_idx)

            (
                obs,
                _,
                initial_A,
                initial_U,
                risk_start_episode,
            ) = reset_training_episode(
                env=env,
                ep_seed=ep_seed,
                training_mode=training_mode,
            )

            terminated = False
            truncated = False

            total_reward = 0.0
            ep_profit = 0.0
            ep_revenue = 0.0
            ep_lost_penalty = 0.0
            ep_holding = 0.0
            ep_expedite = 0.0
            ep_demand = 0.0
            ep_sales = 0.0
            ep_lost_units = 0.0

            losses = []
            A_counts = np.zeros(4, dtype=int)
            U_counts = np.zeros(4, dtype=int)
            action_counts = np.zeros(action_dim, dtype=int)
            steps = 0

            while not (terminated or truncated):
                current_mask = env.action_masks()

                action = agent.select_action(
                    obs,
                    mask=current_mask,
                )
                action_counts[action] += 1

                (
                    next_obs,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(action)

                done = bool(terminated or truncated)

                # s_{t+1} mask for the TD target
                next_mask = env.action_masks()

                agent.memory.push(
                    obs,
                    action,
                    reward,
                    next_obs,
                    done,
                    next_mask=next_mask,
                )

                loss = agent.train_step()
                if loss is not None:
                    losses.append(loss)

                total_reward += float(reward)
                ep_profit += float(info["profit"])
                ep_revenue += float(info["revenue"])
                ep_lost_penalty += float(info["lost_sales_penalty"])
                ep_holding += float(info["holding_cost"])
                ep_expedite += float(info["expedite_cost"])
                ep_demand += float(info["demand_units"])
                ep_sales += float(info["sales_units"])
                ep_lost_units += float(info["lost_units"])

                A_counts[int(info["A_t"])] += 1
                U_counts[int(info["U_t"])] += 1

                obs = next_obs
                steps += 1

            if (episode_idx + 1) % target_update_episodes == 0:
                agent.update_target()

            mean_loss = float(np.mean(losses)) if losses else 0.0

            writer.writerow(
                build_log_row(
                    algorithm=algorithm,
                    seed=seed,
                    episode=episode_idx + 1,
                    ep_seed=ep_seed,
                    training_mode=training_mode,
                    risk_start_episode=risk_start_episode,
                    initial_A=initial_A,
                    initial_U=initial_U,
                    steps=steps,
                    total_reward=total_reward,
                    profit=ep_profit,
                    revenue=ep_revenue,
                    lost_sales_penalty=ep_lost_penalty,
                    holding_cost=ep_holding,
                    expedite_cost=ep_expedite,
                    demand_units=ep_demand,
                    sales_units=ep_sales,
                    lost_units=ep_lost_units,
                    mean_train_loss=mean_loss,
                    epsilon=agent.epsilon,
                    A_counts=A_counts,
                    U_counts=U_counts,
                    action_counts=action_counts,
                )
            )

            f.flush()

            if episode_idx == 0 or (episode_idx + 1) % print_every == 0:
                fill_rate = ep_sales / ep_demand if ep_demand > 0 else 1.0
                start_label = (
                    f"A{initial_A}/U{initial_U}"
                    if risk_start_episode
                    else "A0/U0"
                )

                print(
                    f"Episode {episode_idx + 1:4d}/{episodes}"
                    f" | Start={start_label:7s}"
                    f" | Profit={ep_profit:12,.0f}"
                    f" | Fill={fill_rate:.4f}"
                    f" | Lost={ep_lost_units:8.1f}"
                    f" | eps={agent.epsilon:.4f}"
                    f" | loss={mean_loss:.5f}"
                )

    agent.save(checkpoint_path)

    print(f"\n{algorithm.upper()} complete.")
    print("Log:", log_path)
    print("Checkpoint:", checkpoint_path)

    return {
        "algorithm": algorithm,
        "seed": seed,
        "training_mode": training_mode,
        "log_path": log_path,
        "checkpoint_path": checkpoint_path,
    }


# =============================================================================
# PPO training
# =============================================================================

def train_ppo(
    seed,
    episodes=300,
    device="cpu",
    print_every=25,
    training_mode="natural",
):
    set_global_seed(seed)

    env = SupplyChainEnv(seed=seed)
    state_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.n)

    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
    )

    log_path = os.path.join(
        LOG_DIR,
        f"training_log_ppo_{training_mode}_seed{seed}.csv",
    )
    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        f"ppo_{training_mode}_seed{seed}.pth",
    )

    print("\n" + "=" * 90)
    print(
        f"TRAINING PPO "
        f"| seed={seed} "
        f"| episodes={episodes} "
        f"| state_dim={state_dim} "
        f"| action_dim={action_dim} "
        f"| mode={training_mode} "
        f"| device={device}"
    )
    print("=" * 90)

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(log_header(action_dim))

        for episode_idx in range(episodes):
            ep_seed = episode_seed(seed, episode_idx)

            (
                obs,
                _,
                initial_A,
                initial_U,
                risk_start_episode,
            ) = reset_training_episode(
                env=env,
                ep_seed=ep_seed,
                training_mode=training_mode,
            )

            terminated = False
            truncated = False

            total_reward = 0.0
            ep_profit = 0.0
            ep_revenue = 0.0
            ep_lost_penalty = 0.0
            ep_holding = 0.0
            ep_expedite = 0.0
            ep_demand = 0.0
            ep_sales = 0.0
            ep_lost_units = 0.0

            A_counts = np.zeros(4, dtype=int)
            U_counts = np.zeros(4, dtype=int)
            action_counts = np.zeros(action_dim, dtype=int)
            steps = 0

            while not (terminated or truncated):
                current_mask = env.action_masks()

                action, log_prob, value = agent.select_action(
                    obs,
                    mask=current_mask,
                )
                action_counts[action] += 1

                (
                    next_obs,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(action)

                done = bool(terminated or truncated)

                agent.buffer.push(
                    obs,
                    action,
                    log_prob,
                    reward,
                    done,
                    value,
                    mask=current_mask,
                )

                total_reward += float(reward)
                ep_profit += float(info["profit"])
                ep_revenue += float(info["revenue"])
                ep_lost_penalty += float(info["lost_sales_penalty"])
                ep_holding += float(info["holding_cost"])
                ep_expedite += float(info["expedite_cost"])
                ep_demand += float(info["demand_units"])
                ep_sales += float(info["sales_units"])
                ep_lost_units += float(info["lost_units"])

                A_counts[int(info["A_t"])] += 1
                U_counts[int(info["U_t"])] += 1

                obs = next_obs
                steps += 1

            mean_loss = agent.train_step(last_value=0.0)
            if mean_loss is None:
                mean_loss = 0.0

            writer.writerow(
                build_log_row(
                    algorithm="ppo",
                    seed=seed,
                    episode=episode_idx + 1,
                    ep_seed=ep_seed,
                    training_mode=training_mode,
                    risk_start_episode=risk_start_episode,
                    initial_A=initial_A,
                    initial_U=initial_U,
                    steps=steps,
                    total_reward=total_reward,
                    profit=ep_profit,
                    revenue=ep_revenue,
                    lost_sales_penalty=ep_lost_penalty,
                    holding_cost=ep_holding,
                    expedite_cost=ep_expedite,
                    demand_units=ep_demand,
                    sales_units=ep_sales,
                    lost_units=ep_lost_units,
                    mean_train_loss=float(mean_loss),
                    epsilon="",
                    A_counts=A_counts,
                    U_counts=U_counts,
                    action_counts=action_counts,
                )
            )

            f.flush()

            if episode_idx == 0 or (episode_idx + 1) % print_every == 0:
                fill_rate = ep_sales / ep_demand if ep_demand > 0 else 1.0
                start_label = (
                    f"A{initial_A}/U{initial_U}"
                    if risk_start_episode
                    else "A0/U0"
                )

                print(
                    f"Episode {episode_idx + 1:4d}/{episodes}"
                    f" | Start={start_label:7s}"
                    f" | Profit={ep_profit:12,.0f}"
                    f" | Fill={fill_rate:.4f}"
                    f" | Lost={ep_lost_units:8.1f}"
                    f" | PPO loss={float(mean_loss):.5f}"
                )

    agent.save(checkpoint_path)

    print("\nPPO complete.")
    print("Log:", log_path)
    print("Checkpoint:", checkpoint_path)

    return {
        "algorithm": "ppo",
        "seed": seed,
        "training_mode": training_mode,
        "log_path": log_path,
        "checkpoint_path": checkpoint_path,
    }


def train_one(
    algorithm,
    seed,
    episodes,
    target_update_episodes,
    device,
    print_every,
    training_mode,
):
    if algorithm == "ppo":
        return train_ppo(
            seed=seed,
            episodes=episodes,
            device=device,
            print_every=print_every,
            training_mode=training_mode,
        )

    return train_q_agent(
        algorithm=algorithm,
        seed=seed,
        episodes=episodes,
        target_update_episodes=target_update_episodes,
        device=device,
        print_every=print_every,
        training_mode=training_mode,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train DQN / DDQN / Dueling DQN / D3QN / PPO "
            "on the guide-aligned supply-chain environment."
        )
    )

    parser.add_argument(
        "--algo",
        type=str,
        default="d3qn",
        choices=ALL_ALGORITHMS + ["all"],
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
    )

    parser.add_argument(
        "--training-mode",
        type=str,
        default="natural",
        choices=["natural", "enriched"],
        help=(
            "natural: all episodes start at A0/U0; "
            "enriched: 70%% A0/U0 + 30%% risk-start episodes. "
            "P_A/P_U are unchanged after reset."
        ),
    )

    parser.add_argument(
        "--target-update-episodes",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )

    parser.add_argument(
        "--print-every",
        type=int,
        default=25,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.episodes <= 0:
        raise ValueError("--episodes must be positive.")

    if args.target_update_episodes <= 0:
        raise ValueError(
            "--target-update-episodes must be positive."
        )

    device = resolve_device(args.device)

    algorithms = (
        ALL_ALGORITHMS
        if args.algo == "all"
        else [args.algo]
    )

    print("=" * 90)
    print("UNIFIED RL TRAINING")
    print("=" * 90)
    print("Algorithms:", algorithms)
    print("Seeds:", args.seeds)
    print("Episodes:", args.episodes)
    print("Training mode:", args.training_mode)

    if args.training_mode == "enriched":
        print(
            "Risk-start probability:",
            ENRICHED_RISK_EPISODE_PROB,
        )
        print("NOTE: P_A and P_U are NOT modified.")

    print("Device:", device)

    results = []

    for algorithm in algorithms:
        for seed in args.seeds:
            result = train_one(
                algorithm=algorithm,
                seed=seed,
                episodes=args.episodes,
                target_update_episodes=args.target_update_episodes,
                device=device,
                print_every=args.print_every,
                training_mode=args.training_mode,
            )
            results.append(result)

    print("\n" + "=" * 90)
    print("ALL REQUESTED TRAINING RUNS COMPLETED")
    print("=" * 90)

    for result in results:
        print(
            f"{result['algorithm'].upper():8s} "
            f"mode={result['training_mode']:8s} "
            f"seed={result['seed']} "
            f"-> {result['checkpoint_path']}"
        )


if __name__ == "__main__":
    main()