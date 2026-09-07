import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# =============================================================================
# Utilities
# =============================================================================

def _to_bool_mask(mask, action_dim):
    """
    Convert a single action mask to a NumPy bool array of shape (action_dim,).

    If mask is None, all actions are treated as valid.
    """
    if mask is None:
        return np.ones(action_dim, dtype=bool)

    mask = np.asarray(mask, dtype=bool)

    if mask.shape != (action_dim,):
        raise ValueError(
            f"Action mask must have shape ({action_dim},), got {mask.shape}"
        )

    if not mask.any():
        raise ValueError("Action mask contains no valid action.")

    return mask


def _masked_q_values(q_values, masks):
    """
    Apply a batch of boolean masks to Q-values.

    q_values:
        Tensor [batch, action_dim]

    masks:
        BoolTensor [batch, action_dim]

    Invalid actions are assigned a very negative value so that argmax/max
    cannot select them.
    """
    if masks.dtype != torch.bool:
        masks = masks.bool()

    if q_values.shape != masks.shape:
        raise ValueError(
            f"Q-value shape {q_values.shape} does not match mask shape {masks.shape}"
        )

    if not torch.all(masks.any(dim=1)):
        raise ValueError("At least one sampled next-state mask has no valid action.")

    return q_values.masked_fill(~masks, -1e9)


# =============================================================================
# Replay Buffer
# =============================================================================

class ReplayBuffer:
    """
    Replay buffer for DQN-family agents.

    Important:
        next_mask stores the feasible-action set of s_{t+1}.
        This is required to apply Action Masking inside the TD target.

    Backward compatibility:
        push(..., next_mask=None) is still accepted. In that case all actions
        are considered valid. For the final experiments, train.py should always
        pass the true next-state mask.
    """

    def __init__(self, capacity=50000, action_dim=18):
        self.capacity = int(capacity)
        self.action_dim = int(action_dim)

        self.buffer = []
        self.ptr = 0

    def push(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        next_mask=None,
    ):
        next_mask = _to_bool_mask(
            next_mask,
            self.action_dim,
        )

        transition = (
            np.asarray(state, dtype=np.float32).copy(),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32).copy(),
            float(done),
            next_mask.copy(),
        )

        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.ptr] = transition
        self.ptr = (self.ptr + 1) % self.capacity

    def sample(self, batch_size, device=None):
        batch = random.sample(
            self.buffer,
            batch_size,
        )

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            next_masks,
        ) = zip(*batch)

        states = torch.as_tensor(
            np.asarray(states),
            dtype=torch.float32,
            device=device,
        )

        actions = torch.as_tensor(
            actions,
            dtype=torch.long,
            device=device,
        )

        rewards = torch.as_tensor(
            rewards,
            dtype=torch.float32,
            device=device,
        )

        next_states = torch.as_tensor(
            np.asarray(next_states),
            dtype=torch.float32,
            device=device,
        )

        dones = torch.as_tensor(
            dones,
            dtype=torch.float32,
            device=device,
        )

        next_masks = torch.as_tensor(
            np.asarray(next_masks),
            dtype=torch.bool,
            device=device,
        )

        return (
            states,
            actions,
            rewards,
            next_states,
            dones,
            next_masks,
        )

    def __len__(self):
        return len(self.buffer)


# =============================================================================
# Q Networks
# =============================================================================

class StandardQNet(nn.Module):
    """
    Standard Q-network used by:
        - DQN
        - DDQN

    Architecture:
        state -> 128 -> 128 -> 64 -> action_dim
    """

    def __init__(self, state_dim, action_dim=18):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.q_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        feat = self.feature(x)
        return self.q_head(feat)


class DuelingQNet(nn.Module):
    """
    Dueling Q-network used by:
        - Dueling DQN
        - D3QN

    Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)
    """

    def __init__(self, state_dim, action_dim=18):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        feat = self.feature(x)

        value = self.value_stream(feat)
        advantage = self.advantage_stream(feat)

        q_values = value + (
            advantage
            - advantage.mean(
                dim=-1,
                keepdim=True,
            )
        )

        return q_values


# Backward-compatible name used by the original project.
D3QNNet = DuelingQNet


# =============================================================================
# DQN-family Base Agent
# =============================================================================

class QLearningAgent:
    """
    Shared implementation for the four ablation models.

    The two switches have precise algorithmic meanings:

        double_q=False, dueling=False -> DQN
        double_q=True,  dueling=False -> DDQN
        double_q=False, dueling=True  -> Dueling DQN
        double_q=True,  dueling=True  -> D3QN

    Therefore Double and Dueling are no longer conflated.
    """

    def __init__(
        self,
        state_dim,
        action_dim=18,
        *,
        double_q=False,
        dueling=False,
        lr=1e-3,
        gamma=0.99,
        batch_size=64,
        replay_capacity=50000,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        device=None,
    ):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)

        self.double_q = bool(double_q)
        self.dueling = bool(dueling)

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(device)

        net_cls = (
            DuelingQNet
            if self.dueling
            else StandardQNet
        )

        self.policy_net = net_cls(
            self.state_dim,
            self.action_dim,
        ).to(self.device)

        self.target_net = net_cls(
            self.state_dim,
            self.action_dim,
        ).to(self.device)

        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )

        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=lr,
        )

        self.memory = ReplayBuffer(
            capacity=replay_capacity,
            action_dim=self.action_dim,
        )

        self.gamma = float(gamma)
        self.batch_size = int(batch_size)

        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)

    # -------------------------------------------------------------------------
    # Action selection
    # -------------------------------------------------------------------------

    def select_action(
        self,
        state,
        mask=None,
        deterministic=False,
    ):
        """
        Epsilon-greedy action selection with current-state Action Masking.

        deterministic=True:
            ignore epsilon and choose the best valid action.
            Useful for evaluation.
        """
        mask = _to_bool_mask(
            mask,
            self.action_dim,
        )

        valid_actions = np.flatnonzero(mask)

        if (
            not deterministic
            and random.random() < self.epsilon
        ):
            return int(
                random.choice(
                    valid_actions.tolist()
                )
            )

        state_t = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = (
                self.policy_net(state_t)
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
            )

        q_values = q_values.copy()
        q_values[~mask] = -1e9

        return int(
            np.argmax(q_values)
        )

    def predict(
        self,
        state,
        mask=None,
    ):
        """
        Deterministic evaluation interface.
        """
        return self.select_action(
            state,
            mask=mask,
            deterministic=True,
        )

    # -------------------------------------------------------------------------
    # TD update
    # -------------------------------------------------------------------------

    def train_step(self):
        """
        Perform one mini-batch TD update.

        Target logic:

        DQN / Dueling DQN:
            y = r + gamma * max_{valid a'} Q_target(s', a')

        DDQN / D3QN:
            a* = argmax_{valid a'} Q_policy(s', a')
            y  = r + gamma * Q_target(s', a*)

        Thus next-state Action Masking is applied in BOTH standard and
        Double-Q targets.
        """
        if len(self.memory) < self.batch_size:
            return None

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            next_masks,
        ) = self.memory.sample(
            self.batch_size,
            device=self.device,
        )

        # Q(s_t, a_t)
        q_values = self.policy_net(
            states
        )

        state_action_values = q_values.gather(
            1,
            actions.unsqueeze(1),
        ).squeeze(1)

        with torch.no_grad():

            if self.double_q:
                # -------------------------------------------------------------
                # Double Q:
                # policy chooses, target evaluates.
                # Crucially: policy argmax is restricted to valid next actions.
                # -------------------------------------------------------------
                next_policy_q = self.policy_net(
                    next_states
                )

                next_policy_q = _masked_q_values(
                    next_policy_q,
                    next_masks,
                )

                next_actions = (
                    next_policy_q
                    .argmax(
                        dim=1,
                        keepdim=True,
                    )
                )

                next_target_q = self.target_net(
                    next_states
                )

                next_q_values = (
                    next_target_q
                    .gather(
                        1,
                        next_actions,
                    )
                    .squeeze(1)
                )

            else:
                # -------------------------------------------------------------
                # Standard DQN target:
                # max is also restricted to valid next actions.
                # -------------------------------------------------------------
                next_target_q = self.target_net(
                    next_states
                )

                next_target_q = _masked_q_values(
                    next_target_q,
                    next_masks,
                )

                next_q_values = (
                    next_target_q
                    .max(dim=1)
                    .values
                )

            expected_q_values = (
                rewards
                + (
                    1.0 - dones
                )
                * self.gamma
                * next_q_values
            )

        loss = nn.MSELoss()(
            state_action_values,
            expected_q_values,
        )

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(),
            max_norm=10.0,
        )

        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon = max(
                self.epsilon_min,
                self.epsilon
                * self.epsilon_decay,
            )

        return float(
            loss.item()
        )

    # -------------------------------------------------------------------------
    # Target network
    # -------------------------------------------------------------------------

    def update_target(self):
        """
        Hard target-network synchronization.
        train.py decides how often this is called.
        """
        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )

    # Backward-compatible alias.
    sync_target = update_target

    # -------------------------------------------------------------------------
    # Save / Load
    # -------------------------------------------------------------------------

    def save(self, path):
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "double_q": self.double_q,
                "dueling": self.dueling,
            },
            path,
        )

    def load(
        self,
        path,
        load_optimizer=True,
    ):
        checkpoint = torch.load(
            path,
            map_location=self.device,
        )

        # Support both the new checkpoint dictionary and a legacy plain
        # state_dict file.
        if (
            isinstance(checkpoint, dict)
            and "policy_net" in checkpoint
        ):
            self.policy_net.load_state_dict(
                checkpoint["policy_net"]
            )

            if "target_net" in checkpoint:
                self.target_net.load_state_dict(
                    checkpoint["target_net"]
                )
            else:
                self.update_target()

            if (
                load_optimizer
                and "optimizer" in checkpoint
            ):
                self.optimizer.load_state_dict(
                    checkpoint["optimizer"]
                )

            if "epsilon" in checkpoint:
                self.epsilon = float(
                    checkpoint["epsilon"]
                )

        else:
            self.policy_net.load_state_dict(
                checkpoint
            )
            self.update_target()

        self.target_net.eval()


# =============================================================================
# Four clean ablation agents
# =============================================================================

class DQNAgent(QLearningAgent):
    """
    Standard DQN:
        Double = False
        Dueling = False
    """

    def __init__(
        self,
        state_dim,
        action_dim=18,
        **kwargs,
    ):
        super().__init__(
            state_dim,
            action_dim,
            double_q=False,
            dueling=False,
            **kwargs,
        )


class DDQNAgent(QLearningAgent):
    """
    Double DQN:
        Double = True
        Dueling = False
    """

    def __init__(
        self,
        state_dim,
        action_dim=18,
        **kwargs,
    ):
        super().__init__(
            state_dim,
            action_dim,
            double_q=True,
            dueling=False,
            **kwargs,
        )


class DuelingDQNAgent(QLearningAgent):
    """
    Dueling DQN:
        Double = False
        Dueling = True
    """

    def __init__(
        self,
        state_dim,
        action_dim=18,
        **kwargs,
    ):
        super().__init__(
            state_dim,
            action_dim,
            double_q=False,
            dueling=True,
            **kwargs,
        )


class D3QNAgent(QLearningAgent):
    """
    Dueling Double DQN (D3QN):
        Double = True
        Dueling = True

    Backward compatibility:
        The old project sometimes constructed:
            D3QNAgent(..., is_d3qn=False)
        and treated that as "DQN".

        To avoid immediately breaking such code:
            is_d3qn=True  -> real D3QN
            is_d3qn=False -> real standard DQN

        For the final experiments, use the explicit DQNAgent / DDQNAgent /
        DuelingDQNAgent / D3QNAgent classes instead.
    """

    def __init__(
        self,
        state_dim,
        action_dim=18,
        is_d3qn=True,
        **kwargs,
    ):
        if is_d3qn:
            double_q = True
            dueling = True
        else:
            double_q = False
            dueling = False

        QLearningAgent.__init__(
            self,
            state_dim,
            action_dim,
            double_q=double_q,
            dueling=dueling,
            **kwargs,
        )


# =============================================================================
# PPO
# =============================================================================

class PPONet(nn.Module):
    """
    PPO Actor-Critic with a shared feature extractor.
    """

    def __init__(
        self,
        state_dim,
        action_dim=18,
    ):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        feat = self.shared(x)

        logits = self.actor(feat)
        value = self.critic(feat)

        return logits, value

    def get_action_distribution(
        self,
        obs_tensor,
        mask_tensor=None,
    ):
        logits, value = self.forward(
            obs_tensor
        )

        if mask_tensor is not None:
            if mask_tensor.dtype != torch.bool:
                mask_tensor = mask_tensor.bool()

            logits = logits.masked_fill(
                ~mask_tensor,
                -1e9,
            )

        dist = torch.distributions.Categorical(
            logits=logits
        )

        return dist, value


class RolloutBuffer:
    """
    PPO on-policy rollout buffer.

    Masks are stored because PPO must evaluate each sampled action under the
    SAME feasible-action set that was used when the old log-probability was
    generated.
    """

    def __init__(
        self,
        action_dim=18,
    ):
        self.action_dim = int(
            action_dim
        )

        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.masks = []

    def push(
        self,
        state,
        action,
        log_prob,
        reward,
        done,
        value,
        mask=None,
    ):
        mask = _to_bool_mask(
            mask,
            self.action_dim,
        )

        self.states.append(
            np.asarray(
                state,
                dtype=np.float32,
            ).copy()
        )

        self.actions.append(
            int(action)
        )

        self.log_probs.append(
            float(log_prob)
        )

        self.rewards.append(
            float(reward)
        )

        self.dones.append(
            float(done)
        )

        self.values.append(
            float(value)
        )

        self.masks.append(
            mask.copy()
        )

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.masks.clear()

    def get_tensors(
        self,
        device=None,
    ):
        return (
            torch.as_tensor(
                np.asarray(self.states),
                dtype=torch.float32,
                device=device,
            ),
            torch.as_tensor(
                self.actions,
                dtype=torch.long,
                device=device,
            ),
            torch.as_tensor(
                self.log_probs,
                dtype=torch.float32,
                device=device,
            ),
            torch.as_tensor(
                self.rewards,
                dtype=torch.float32,
                device=device,
            ),
            torch.as_tensor(
                self.dones,
                dtype=torch.float32,
                device=device,
            ),
            torch.as_tensor(
                self.values,
                dtype=torch.float32,
                device=device,
            ),
            torch.as_tensor(
                np.asarray(self.masks),
                dtype=torch.bool,
                device=device,
            ),
        )

    def __len__(self):
        return len(
            self.states
        )


class PPOAgent:
    """
    PPO-Clip Agent with consistent Action Masking in:
        1. action sampling,
        2. deterministic prediction,
        3. PPO update / new log-probability calculation.
    """

    def __init__(
        self,
        state_dim,
        action_dim=18,
        device=None,
    ):
        self.state_dim = int(
            state_dim
        )

        self.action_dim = int(
            action_dim
        )

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(
            device
        )

        self.net = PPONet(
            self.state_dim,
            self.action_dim,
        ).to(
            self.device
        )

        self.optimizer = optim.Adam(
            self.net.parameters(),
            lr=3e-4,
        )

        self.buffer = RolloutBuffer(
            action_dim=self.action_dim
        )

        # PPO hyperparameters
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.entropy_coef = 0.01
        self.value_coef = 0.5
        self.ppo_epochs = 10
        self.batch_size = 64

    # -------------------------------------------------------------------------
    # Mask helpers
    # -------------------------------------------------------------------------

    def _mask_tensor(
        self,
        mask,
        batch_size=1,
    ):
        mask = _to_bool_mask(
            mask,
            self.action_dim,
        )

        mask_t = torch.as_tensor(
            mask,
            dtype=torch.bool,
            device=self.device,
        ).unsqueeze(0)

        if batch_size > 1:
            mask_t = mask_t.expand(
                batch_size,
                -1,
            )

        return mask_t

    # -------------------------------------------------------------------------
    # Action selection
    # -------------------------------------------------------------------------

    def select_action(
        self,
        state,
        mask=None,
    ):
        """
        Sample from the masked policy.

        Returns:
            action, log_prob, value
        """
        state_t = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        mask_t = self._mask_tensor(
            mask,
            batch_size=1,
        )

        with torch.no_grad():
            dist, value = (
                self.net.get_action_distribution(
                    state_t,
                    mask_t,
                )
            )

            action = dist.sample()
            log_prob = dist.log_prob(
                action
            )

        return (
            int(action.item()),
            float(log_prob.item()),
            float(value.squeeze(-1).item()),
        )

    def predict(
        self,
        state,
        mask=None,
    ):
        """
        Deterministic evaluation: choose the highest-logit valid action.
        """
        state_t = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        mask_t = self._mask_tensor(
            mask,
            batch_size=1,
        )

        with torch.no_grad():
            logits, _ = self.net(
                state_t
            )

            logits = logits.masked_fill(
                ~mask_t,
                -1e9,
            )

        return int(
            logits.argmax(
                dim=-1
            ).item()
        )

    # -------------------------------------------------------------------------
    # GAE
    # -------------------------------------------------------------------------

    def compute_gae(
        self,
        rewards,
        values,
        dones,
        last_value=0.0,
    ):
        """
        Generalized Advantage Estimation.

        last_value:
            bootstrap value after the final rollout transition.
            Use 0 for a terminal episode.
            For a non-terminal rollout cut, train.py may provide V(s_{T+1}).
        """
        advantages = []
        gae = 0.0

        for t in reversed(
            range(
                len(rewards)
            )
        ):
            if t == len(rewards) - 1:
                next_value = (
                    0.0
                    if dones[t]
                    else float(last_value)
                )
            else:
                next_value = (
                    0.0
                    if dones[t]
                    else values[t + 1]
                )

            delta = (
                rewards[t]
                + self.gamma
                * next_value
                * (
                    1.0 - dones[t]
                )
                - values[t]
            )

            gae = (
                delta
                + self.gamma
                * self.gae_lambda
                * (
                    1.0 - dones[t]
                )
                * gae
            )

            advantages.insert(
                0,
                gae,
            )

        returns = [
            adv + val
            for adv, val
            in zip(
                advantages,
                values,
            )
        ]

        return (
            advantages,
            returns,
        )

    # -------------------------------------------------------------------------
    # PPO update
    # -------------------------------------------------------------------------

    def train_step(
        self,
        last_value=0.0,
    ):
        """
        Run multiple PPO epochs over the current rollout.

        Crucial fix:
            new_log_probs are computed under the SAME stored action masks
            used when old_log_probs were generated.
        """
        if len(
            self.buffer
        ) == 0:
            return None

        (
            states,
            actions,
            old_log_probs,
            rewards,
            dones,
            values,
            masks,
        ) = self.buffer.get_tensors(
            device=self.device
        )

        old_log_probs = (
            old_log_probs.detach()
        )

        rewards_np = (
            rewards.detach()
            .cpu()
            .numpy()
            .tolist()
        )

        dones_np = (
            dones.detach()
            .cpu()
            .numpy()
            .tolist()
        )

        values_np = (
            values.detach()
            .cpu()
            .numpy()
            .tolist()
        )

        (
            advantages,
            returns,
        ) = self.compute_gae(
            rewards_np,
            values_np,
            dones_np,
            last_value=last_value,
        )

        advantages = torch.as_tensor(
            advantages,
            dtype=torch.float32,
            device=self.device,
        )

        returns = torch.as_tensor(
            returns,
            dtype=torch.float32,
            device=self.device,
        )

        if len(
            advantages
        ) > 1:
            advantages = (
                advantages
                - advantages.mean()
            ) / (
                advantages.std(
                    unbiased=False
                )
                + 1e-8
            )

        dataset = (
            torch.utils.data.TensorDataset(
                states,
                actions,
                old_log_probs,
                advantages,
                returns,
                masks,
            )
        )

        dataloader = (
            torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
            )
        )

        total_loss = 0.0
        num_updates = 0

        for _ in range(
            self.ppo_epochs
        ):
            for batch in dataloader:

                (
                    s_b,
                    a_b,
                    old_lp_b,
                    adv_b,
                    ret_b,
                    mask_b,
                ) = batch

                logits, value_pred = (
                    self.net(
                        s_b
                    )
                )

                # Same mask as behavior policy.
                masked_logits = (
                    logits.masked_fill(
                        ~mask_b,
                        -1e9,
                    )
                )

                dist = (
                    torch.distributions.Categorical(
                        logits=masked_logits
                    )
                )

                new_log_probs = (
                    dist.log_prob(
                        a_b
                    )
                )

                entropy = (
                    dist.entropy()
                    .mean()
                )

                ratio = torch.exp(
                    new_log_probs
                    - old_lp_b
                )

                surr1 = (
                    ratio
                    * adv_b
                )

                surr2 = (
                    torch.clamp(
                        ratio,
                        1.0
                        - self.clip_epsilon,
                        1.0
                        + self.clip_epsilon,
                    )
                    * adv_b
                )

                policy_loss = (
                    -torch.min(
                        surr1,
                        surr2,
                    ).mean()
                )

                value_loss = (
                    nn.MSELoss()(
                        value_pred.squeeze(-1),
                        ret_b,
                    )
                )

                loss = (
                    policy_loss
                    + self.value_coef
                    * value_loss
                    - self.entropy_coef
                    * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.net.parameters(),
                    max_norm=0.5,
                )

                self.optimizer.step()

                total_loss += float(
                    loss.item()
                )

                num_updates += 1

        self.buffer.clear()

        if num_updates == 0:
            return 0.0

        return (
            total_loss
            / num_updates
        )

    # -------------------------------------------------------------------------
    # Value bootstrap helper
    # -------------------------------------------------------------------------

    def get_value(
        self,
        state,
    ):
        state_t = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            _, value = self.net(
                state_t
            )

        return float(
            value.squeeze(-1).item()
        )

    # -------------------------------------------------------------------------
    # Save / Load
    # -------------------------------------------------------------------------

    def save(
        self,
        path,
    ):
        torch.save(
            {
                "net": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
            },
            path,
        )

    def load(
        self,
        path,
        load_optimizer=True,
    ):
        checkpoint = torch.load(
            path,
            map_location=self.device,
        )

        if (
            isinstance(checkpoint, dict)
            and "net" in checkpoint
        ):
            self.net.load_state_dict(
                checkpoint["net"]
            )

            if (
                load_optimizer
                and "optimizer" in checkpoint
            ):
                self.optimizer.load_state_dict(
                    checkpoint["optimizer"]
                )

        else:
            # Legacy plain state_dict
            self.net.load_state_dict(
                checkpoint
            )