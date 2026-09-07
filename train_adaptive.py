
"""
train_adaptive.py

Adaptive rare-disruption training for the existing RL supply-chain project.

Place this file in:
    D:\暑假科研\rl_supply_chain\train_adaptive.py

It reuses the existing project's unified training helpers from train.py
(or train_corrected.py when present), and writes NEW logs/checkpoints with
training_mode="adaptive".

Core design
-----------
- 70% of episodes: normal start A0/U0.
- 30% of episodes: risk start.
- Only the conditional distribution INSIDE the 30% risk branch adapts.
- Adaptive probabilities are a mixture of:
      uniform base distribution
      + difficulty-based softmax distribution
- Per-risk probabilities have lower/upper bounds.
- Difficulty is smoothed with EMA.
- P_A and P_U are NEVER modified.
- Risk override is applied only at reset; original Markov dynamics resume immediately.

This lets the paper compare:
    natural vs enriched vs adaptive

without changing any existing checkpoints.

IMPORTANT
---------
The adaptive hyperparameters in this file are RESEARCH-DESIGN choices, not
teacher/file-prescribed parameters. They should be reported as such in the paper.
"""

import argparse
import csv
import os
from dataclasses import dataclass

import numpy as np

# Reuse the project's existing unified training helpers.
# Some project versions call the file train_corrected.py, while the user's
# actual project uses train.py. Support both without changing old experiments.
try:
    from train_corrected import (
        ALL_ALGORITHMS,
        Q_AGENT_CLASSES,
        SupplyChainEnv,
        PPOAgent,
        build_log_row,
        build_q_agent,
        episode_seed,
        log_header,
        resolve_device,
        set_global_seed,
    )
except ModuleNotFoundError:
    from train import (
        ALL_ALGORITHMS,
        Q_AGENT_CLASSES,
        SupplyChainEnv,
        PPOAgent,
        build_log_row,
        build_q_agent,
        episode_seed,
        log_header,
        resolve_device,
        set_global_seed,
    )

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(CURRENT_DIR, "training_logs")
CHECKPOINT_DIR = os.path.join(CURRENT_DIR, "checkpoints")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Same risk-start states as the existing fixed-enriched training.
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

RISK_LABELS = [
    "A1U0",
    "A2U0",
    "A3U0",
    "A0U1",
    "A0U2",
    "A0U3",
    "A2U2",
    "A3U3",
]


def bounded_simplex(raw, lower, upper, max_iter=200):
    """
    Project positive weights to a probability vector with
    lower <= p_i <= upper and sum(p)=1.

    This is deliberately simple because K=8.
    """
    x = np.asarray(raw, dtype=np.float64)
    if np.any(~np.isfinite(x)) or np.all(x <= 0):
        x = np.ones_like(x)

    x = np.maximum(x, 1e-12)
    x = x / x.sum()

    n = len(x)
    if lower * n > 1.0 + 1e-12:
        raise ValueError("lower bound is infeasible.")
    if upper * n < 1.0 - 1e-12:
        raise ValueError("upper bound is infeasible.")

    p = x.copy()

    for _ in range(max_iter):
        old = p.copy()

        p = np.clip(p, lower, upper)
        total = p.sum()

        if abs(total - 1.0) < 1e-12:
            break

        if total < 1.0:
            room = np.maximum(upper - p, 0.0)
            room_sum = room.sum()
            if room_sum <= 1e-15:
                break
            p += (1.0 - total) * room / room_sum
        else:
            removable = np.maximum(p - lower, 0.0)
            removable_sum = removable.sum()
            if removable_sum <= 1e-15:
                break
            p -= (total - 1.0) * removable / removable_sum

        if np.max(np.abs(p - old)) < 1e-12:
            break

    # Numerical clean-up.
    p = np.clip(p, lower, upper)
    p = p / p.sum()

    return p


@dataclass
class CurriculumConfig:
    # Overall start distribution
    risk_episode_prob: float = 0.30

    # Keep a uniform component to prevent scenario starvation.
    adaptive_mix_beta: float = 0.50

    # Softmax sensitivity to difficulty.
    temperature: float = 4.0

    # EMA smoothing.
    ema_rho: float = 0.85

    # Do not adapt too early.
    warmup_episodes: int = 60

    # Recompute sampling distribution every N episodes.
    update_interval: int = 20

    # Conditional probabilities within the risk branch.
    min_risk_cond_prob: float = 0.03
    max_risk_cond_prob: float = 0.30

    # Difficulty weights.
    w_profit: float = 0.45
    w_fill: float = 0.35
    w_lost: float = 0.20

    # Normal baseline EMA.
    normal_profit_ema_rho: float = 0.90


class AdaptiveRiskCurriculum:
    def __init__(self, seed, cfg):
        self.cfg = cfg
        self.rng = np.random.default_rng(int(seed) + 910273)

        self.k = len(RISK_START_STATES)
        self.uniform = np.full(self.k, 1.0 / self.k, dtype=np.float64)

        # Start with equal difficulty -> uniform risk sampling.
        self.difficulty_ema = np.zeros(self.k, dtype=np.float64)
        self.seen_count = np.zeros(self.k, dtype=np.int64)
        self.probs = self.uniform.copy()

        self.normal_profit_ema = None

    def _softmax(self, x):
        z = self.cfg.temperature * np.asarray(x, dtype=np.float64)
        z = z - np.max(z)
        exp_z = np.exp(np.clip(z, -50.0, 50.0))
        return exp_z / exp_z.sum()

    def refresh_probs(self, episode_number):
        """
        Adapt only after warmup. The adaptive distribution is mixed with
        the uniform base distribution, then bounded.
        """
        if episode_number <= self.cfg.warmup_episodes:
            self.probs = self.uniform.copy()
            return

        adaptive = self._softmax(self.difficulty_ema)

        raw = (
            (1.0 - self.cfg.adaptive_mix_beta) * self.uniform
            + self.cfg.adaptive_mix_beta * adaptive
        )

        self.probs = bounded_simplex(
            raw,
            lower=self.cfg.min_risk_cond_prob,
            upper=self.cfg.max_risk_cond_prob,
        )

    def choose_start(self, ep_seed):
        """
        70% normal by default; 30% risk by default.
        ep_seed is used to make the start decision reproducible.
        """
        local_rng = np.random.default_rng(int(ep_seed) + 20260817)

        if local_rng.random() >= self.cfg.risk_episode_prob:
            return 0, 0, False, -1

        idx = int(local_rng.choice(self.k, p=self.probs))
        a, u = RISK_START_STATES[idx]
        return int(a), int(u), True, idx

    def observe_episode(
        self,
        *,
        risk_idx,
        is_risk,
        profit,
        fill_rate,
        lost_units,
        demand_units,
    ):
        """
        Update the normal-profit baseline or the difficulty of the sampled risk.

        Difficulty components:
          profit_deficit = relative shortfall vs normal-start EMA
          fill_deficit   = 1 - fill_rate
          lost_ratio     = lost_units / demand_units

        All are non-negative. This is a training signal, not the RL reward.
        """
        profit = float(profit)
        fill_rate = float(fill_rate)
        lost_units = float(lost_units)
        demand_units = float(demand_units)

        if not is_risk:
            if self.normal_profit_ema is None:
                self.normal_profit_ema = profit
            else:
                rho = self.cfg.normal_profit_ema_rho
                self.normal_profit_ema = (
                    rho * self.normal_profit_ema
                    + (1.0 - rho) * profit
                )
            return

        if risk_idx < 0:
            return

        fill_deficit = max(0.0, 1.0 - fill_rate)
        lost_ratio = max(
            0.0,
            lost_units / max(demand_units, 1.0),
        )

        if self.normal_profit_ema is None:
            profit_deficit = 0.0
        else:
            denom = max(abs(self.normal_profit_ema), 1.0)
            profit_deficit = max(
                0.0,
                (self.normal_profit_ema - profit) / denom,
            )

        raw_difficulty = (
            self.cfg.w_profit * profit_deficit
            + self.cfg.w_fill * fill_deficit
            + self.cfg.w_lost * lost_ratio
        )

        # First observation uses the raw value; later use EMA.
        if self.seen_count[risk_idx] == 0:
            self.difficulty_ema[risk_idx] = raw_difficulty
        else:
            rho = self.cfg.ema_rho
            self.difficulty_ema[risk_idx] = (
                rho * self.difficulty_ema[risk_idx]
                + (1.0 - rho) * raw_difficulty
            )

        self.seen_count[risk_idx] += 1

    def curriculum_row(self, episode, algorithm, seed):
        row = {
            "algorithm": algorithm,
            "seed": seed,
            "episode": episode,
            "normal_profit_ema": (
                np.nan
                if self.normal_profit_ema is None
                else self.normal_profit_ema
            ),
        }

        for i, label in enumerate(RISK_LABELS):
            row[f"{label}_prob"] = self.probs[i]
            row[f"{label}_difficulty"] = self.difficulty_ema[i]
            row[f"{label}_seen"] = int(self.seen_count[i])

        return row


def reset_adaptive_episode(env, ep_seed, curriculum):
    initial_A, initial_U, is_risk, risk_idx = curriculum.choose_start(ep_seed)

    # Exactly as in the fixed-enriched version:
    # force ONLY the initial state.
    env.set_disruption_override(
        A_t=initial_A,
        U_t=initial_U,
    )

    obs, info = env.reset(seed=ep_seed)

    # From the first step onward, original Markov matrices apply.
    env.clear_disruption_override()

    return obs, info, initial_A, initial_U, is_risk, risk_idx


def curriculum_header():
    cols = [
        "algorithm",
        "seed",
        "episode",
        "normal_profit_ema",
    ]
    for label in RISK_LABELS:
        cols += [
            f"{label}_prob",
            f"{label}_difficulty",
            f"{label}_seen",
        ]
    return cols


def train_q_adaptive(
    *,
    algorithm,
    seed,
    episodes,
    target_update_episodes,
    device,
    print_every,
    cfg,
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

    curriculum = AdaptiveRiskCurriculum(seed=seed, cfg=cfg)

    training_mode = "adaptive"

    log_path = os.path.join(
        LOG_DIR,
        f"training_log_{algorithm}_{training_mode}_seed{seed}.csv",
    )
    curriculum_path = os.path.join(
        LOG_DIR,
        f"curriculum_log_{algorithm}_{training_mode}_seed{seed}.csv",
    )
    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        f"{algorithm}_{training_mode}_seed{seed}.pth",
    )

    print("\n" + "=" * 100)
    print(
        f"ADAPTIVE TRAINING {algorithm.upper()} "
        f"| seed={seed} | episodes={episodes} "
        f"| risk_prob={cfg.risk_episode_prob:.2f} "
        f"| beta={cfg.adaptive_mix_beta:.2f} "
        f"| warmup={cfg.warmup_episodes}"
    )
    print("=" * 100)
    print("Risk states:", RISK_LABELS)
    print("P_A / P_U are NOT modified.")

    with (
        open(log_path, "w", newline="", encoding="utf-8") as f,
        open(curriculum_path, "w", newline="", encoding="utf-8") as cf,
    ):
        writer = csv.writer(f)
        writer.writerow(log_header(action_dim))

        curriculum_writer = csv.DictWriter(
            cf,
            fieldnames=curriculum_header(),
        )
        curriculum_writer.writeheader()

        for episode_idx in range(episodes):
            episode_number = episode_idx + 1

            if (
                episode_number == 1
                or (
                    episode_number > cfg.warmup_episodes
                    and (episode_number - 1) % cfg.update_interval == 0
                )
            ):
                curriculum.refresh_probs(episode_number)

            ep_seed = episode_seed(seed, episode_idx)

            (
                obs,
                _,
                initial_A,
                initial_U,
                risk_start_episode,
                risk_idx,
            ) = reset_adaptive_episode(
                env=env,
                ep_seed=ep_seed,
                curriculum=curriculum,
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

            if episode_number % target_update_episodes == 0:
                agent.update_target()

            mean_loss = float(np.mean(losses)) if losses else 0.0
            fill_rate = ep_sales / ep_demand if ep_demand > 0 else 1.0

            curriculum.observe_episode(
                risk_idx=risk_idx,
                is_risk=risk_start_episode,
                profit=ep_profit,
                fill_rate=fill_rate,
                lost_units=ep_lost_units,
                demand_units=ep_demand,
            )

            # Recompute for logging at the end of an update block.
            if (
                episode_number >= cfg.warmup_episodes
                and episode_number % cfg.update_interval == 0
            ):
                curriculum.refresh_probs(episode_number)

            writer.writerow(
                build_log_row(
                    algorithm=algorithm,
                    seed=seed,
                    episode=episode_number,
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

            curriculum_writer.writerow(
                curriculum.curriculum_row(
                    episode=episode_number,
                    algorithm=algorithm,
                    seed=seed,
                )
            )

            f.flush()
            cf.flush()

            if (
                episode_number == 1
                or episode_number % print_every == 0
            ):
                start_label = (
                    f"A{initial_A}/U{initial_U}"
                    if risk_start_episode
                    else "A0/U0"
                )
                hardest_idx = int(np.argmax(curriculum.probs))
                print(
                    f"Episode {episode_number:4d}/{episodes}"
                    f" | Start={start_label:7s}"
                    f" | Profit={ep_profit:12,.0f}"
                    f" | Fill={fill_rate:.4f}"
                    f" | Lost={ep_lost_units:8.1f}"
                    f" | eps={agent.epsilon:.4f}"
                    f" | top-risk={RISK_LABELS[hardest_idx]}"
                    f"({curriculum.probs[hardest_idx]:.3f})"
                )

    agent.save(checkpoint_path)

    print("\nAdaptive training complete.")
    print("Training log   :", log_path)
    print("Curriculum log :", curriculum_path)
    print("Checkpoint     :", checkpoint_path)

    return {
        "algorithm": algorithm,
        "seed": seed,
        "training_mode": training_mode,
        "log_path": log_path,
        "curriculum_path": curriculum_path,
        "checkpoint_path": checkpoint_path,
    }


def train_ppo_adaptive(
    *,
    seed,
    episodes,
    device,
    print_every,
    cfg,
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

    curriculum = AdaptiveRiskCurriculum(seed=seed, cfg=cfg)
    training_mode = "adaptive"

    log_path = os.path.join(
        LOG_DIR,
        f"training_log_ppo_{training_mode}_seed{seed}.csv",
    )
    curriculum_path = os.path.join(
        LOG_DIR,
        f"curriculum_log_ppo_{training_mode}_seed{seed}.csv",
    )
    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        f"ppo_{training_mode}_seed{seed}.pth",
    )

    print("\n" + "=" * 100)
    print(
        f"ADAPTIVE TRAINING PPO "
        f"| seed={seed} | episodes={episodes} "
        f"| risk_prob={cfg.risk_episode_prob:.2f} "
        f"| beta={cfg.adaptive_mix_beta:.2f} "
        f"| warmup={cfg.warmup_episodes}"
    )
    print("=" * 100)
    print("Risk states:", RISK_LABELS)
    print("P_A / P_U are NOT modified.")

    with (
        open(log_path, "w", newline="", encoding="utf-8") as f,
        open(curriculum_path, "w", newline="", encoding="utf-8") as cf,
    ):
        writer = csv.writer(f)
        writer.writerow(log_header(action_dim))

        curriculum_writer = csv.DictWriter(
            cf,
            fieldnames=curriculum_header(),
        )
        curriculum_writer.writeheader()

        for episode_idx in range(episodes):
            episode_number = episode_idx + 1

            if (
                episode_number == 1
                or (
                    episode_number > cfg.warmup_episodes
                    and (episode_number - 1) % cfg.update_interval == 0
                )
            ):
                curriculum.refresh_probs(episode_number)

            ep_seed = episode_seed(seed, episode_idx)

            (
                obs,
                _,
                initial_A,
                initial_U,
                risk_start_episode,
                risk_idx,
            ) = reset_adaptive_episode(
                env=env,
                ep_seed=ep_seed,
                curriculum=curriculum,
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

            fill_rate = ep_sales / ep_demand if ep_demand > 0 else 1.0

            curriculum.observe_episode(
                risk_idx=risk_idx,
                is_risk=risk_start_episode,
                profit=ep_profit,
                fill_rate=fill_rate,
                lost_units=ep_lost_units,
                demand_units=ep_demand,
            )

            if (
                episode_number >= cfg.warmup_episodes
                and episode_number % cfg.update_interval == 0
            ):
                curriculum.refresh_probs(episode_number)

            writer.writerow(
                build_log_row(
                    algorithm="ppo",
                    seed=seed,
                    episode=episode_number,
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

            curriculum_writer.writerow(
                curriculum.curriculum_row(
                    episode=episode_number,
                    algorithm="ppo",
                    seed=seed,
                )
            )

            f.flush()
            cf.flush()

            if (
                episode_number == 1
                or episode_number % print_every == 0
            ):
                start_label = (
                    f"A{initial_A}/U{initial_U}"
                    if risk_start_episode
                    else "A0/U0"
                )
                hardest_idx = int(np.argmax(curriculum.probs))
                print(
                    f"Episode {episode_number:4d}/{episodes}"
                    f" | Start={start_label:7s}"
                    f" | Profit={ep_profit:12,.0f}"
                    f" | Fill={fill_rate:.4f}"
                    f" | Lost={ep_lost_units:8.1f}"
                    f" | top-risk={RISK_LABELS[hardest_idx]}"
                    f"({curriculum.probs[hardest_idx]:.3f})"
                )

    agent.save(checkpoint_path)

    print("\nAdaptive PPO training complete.")
    print("Training log   :", log_path)
    print("Curriculum log :", curriculum_path)
    print("Checkpoint     :", checkpoint_path)

    return {
        "algorithm": "ppo",
        "seed": seed,
        "training_mode": training_mode,
        "log_path": log_path,
        "curriculum_path": curriculum_path,
        "checkpoint_path": checkpoint_path,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Adaptive rare-disruption RL training."
    )

    parser.add_argument(
        "--algo",
        type=str,
        default="dueling",
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
        default=[42, 43, 44, 45, 46],
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

    # Adaptive curriculum hyperparameters.
    parser.add_argument("--risk-prob", type=float, default=0.30)
    parser.add_argument("--beta", type=float, default=0.50)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--ema-rho", type=float, default=0.85)
    parser.add_argument("--warmup", type=int, default=60)
    parser.add_argument("--update-interval", type=int, default=20)
    parser.add_argument("--min-risk-prob", type=float, default=0.03)
    parser.add_argument("--max-risk-prob", type=float, default=0.30)
    parser.add_argument("--w-profit", type=float, default=0.45)
    parser.add_argument("--w-fill", type=float, default=0.35)
    parser.add_argument("--w-lost", type=float, default=0.20)

    return parser.parse_args()


def validate_args(args):
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive.")
    if not (0.0 <= args.risk_prob <= 1.0):
        raise ValueError("--risk-prob must be in [0,1].")
    if not (0.0 <= args.beta <= 1.0):
        raise ValueError("--beta must be in [0,1].")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive.")
    if not (0.0 <= args.ema_rho < 1.0):
        raise ValueError("--ema-rho must be in [0,1).")
    if args.warmup < 0:
        raise ValueError("--warmup must be >=0.")
    if args.update_interval <= 0:
        raise ValueError("--update-interval must be positive.")

    weight_sum = args.w_profit + args.w_fill + args.w_lost
    if abs(weight_sum - 1.0) > 1e-8:
        raise ValueError(
            "--w-profit + --w-fill + --w-lost must sum to 1."
        )


def main():
    args = parse_args()
    validate_args(args)

    device = resolve_device(args.device)

    cfg = CurriculumConfig(
        risk_episode_prob=args.risk_prob,
        adaptive_mix_beta=args.beta,
        temperature=args.temperature,
        ema_rho=args.ema_rho,
        warmup_episodes=args.warmup,
        update_interval=args.update_interval,
        min_risk_cond_prob=args.min_risk_prob,
        max_risk_cond_prob=args.max_risk_prob,
        w_profit=args.w_profit,
        w_fill=args.w_fill,
        w_lost=args.w_lost,
    )

    algorithms = (
        ALL_ALGORITHMS
        if args.algo == "all"
        else [args.algo]
    )

    print("=" * 100)
    print("ADAPTIVE RARE-DISRUPTION TRAINING")
    print("=" * 100)
    print("Algorithms          :", algorithms)
    print("Seeds               :", args.seeds)
    print("Episodes            :", args.episodes)
    print("Device              :", device)
    print("Normal episode prob :", 1.0 - cfg.risk_episode_prob)
    print("Risk episode prob   :", cfg.risk_episode_prob)
    print("Adaptive beta       :", cfg.adaptive_mix_beta)
    print("Warmup episodes     :", cfg.warmup_episodes)
    print("Update interval     :", cfg.update_interval)
    print("EMA rho             :", cfg.ema_rho)
    print("Risk prob bounds    :", (
        cfg.min_risk_cond_prob,
        cfg.max_risk_cond_prob,
    ))
    print("Difficulty weights  :", (
        cfg.w_profit,
        cfg.w_fill,
        cfg.w_lost,
    ))
    print("NOTE: P_A/P_U remain unchanged.")

    results = []

    for algorithm in algorithms:
        for seed in args.seeds:
            if algorithm == "ppo":
                result = train_ppo_adaptive(
                    seed=seed,
                    episodes=args.episodes,
                    device=device,
                    print_every=args.print_every,
                    cfg=cfg,
                )
            else:
                result = train_q_adaptive(
                    algorithm=algorithm,
                    seed=seed,
                    episodes=args.episodes,
                    target_update_episodes=args.target_update_episodes,
                    device=device,
                    print_every=args.print_every,
                    cfg=cfg,
                )

            results.append(result)

    print("\n" + "=" * 100)
    print("ALL ADAPTIVE RUNS COMPLETED")
    print("=" * 100)

    for result in results:
        print(
            f"{result['algorithm'].upper():8s} "
            f"seed={result['seed']} "
            f"-> {result['checkpoint_path']}"
        )


if __name__ == "__main__":
    main()
