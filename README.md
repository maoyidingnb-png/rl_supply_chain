# Reinforcement Learning for Disruption-Resilient Supply Chains

This repository studies inventory and allocation decisions in a two-echelon instant-retail supply chain under rare supplier and transportation disruptions. It provides a Gymnasium environment, five reinforcement-learning agents, reproducible training and evaluation scripts, validation experiments, trained checkpoints, and publication-ready results.

## Research question

Can reinforcement learning improve service and profit when a central warehouse must replenish ten fulfillment warehouses while supplier capacity and transportation availability evolve as Markov disruption processes?

The environment models five products, one central warehouse (CW), ten fulfillment warehouses (FWs), stochastic demand, lead-time pipelines, supplier backlog, capacity limits, and two four-state disruption processes. Each daily decision selects one of 18 combinations of:

- inventory-protection level (`l`);
- upstream expediting decision (`e`);
- CW-to-FW allocation rule (`g`).

The normalized observation has 94 dimensions. The reward is daily profit: revenue minus lost-sales penalties, holding costs, and expediting costs.

## Algorithms

- DQN
- Double DQN (DDQN)
- Dueling DQN
- Dueling Double DQN (D3QN)
- PPO

Two rare-event training approaches are included: a fixed enriched-start distribution and an adaptive curriculum that updates the conditional mix of disruption scenarios while leaving the underlying Markov transition matrices unchanged.

## Repository structure

```text
rl_supply_chain/
|-- env.py                         # Gymnasium supply-chain environment
|-- models.py                      # DQN-family and PPO implementations
|-- baselines.py                   # Fixed and rule-based policies
|-- train.py                       # Natural/fixed-enriched training
|-- train_adaptive.py              # Adaptive rare-disruption curriculum
|-- evaluate_models.py             # Formal multi-seed evaluation
|-- evaluate_models_adaptive.py    # Adaptive-model evaluation
|-- analysis/                      # Calibration, diagnostics, and figures
|-- tests/                         # Executable environment validations
|-- checkpoints/                   # Trained model weights
|-- training_logs/                 # Per-episode training histories
|-- evaluation_results/            # Raw and aggregated evaluation data
|-- validation_results/            # Controlled validation outputs
|-- paper_figures_v2/              # Publication-ready PDF/PNG figures
`-- adaptive_figures/              # Adaptive-curriculum diagnostics
```

Generated results are intentionally included so the main findings can be inspected without retraining. The complete checkpoint set is about 23 MB.

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Quick verification

Run the guide-aligned environment validation:

```bash
python -m tests.validate_environment
```

This checks the 94-dimensional observation, 18-action mapping, action masking, inventory-protection multipliers, allocation rules, Markov matrices, lead times, transport backlog, accounting identities, and capacity constraints.

Run a one-episode training smoke test:

```bash
python train.py --algo dqn --episodes 1 --seeds 999 --device cpu --print-every 1
```

The seed-999 smoke artifacts are ignored by Git.

## Reproduce the experiments

Train all five algorithms with five seeds under fixed enriched starts:

```bash
python train.py --algo all --episodes 300 --seeds 42 43 44 45 46 --training-mode enriched
```

Train the adaptive curriculum (example: Dueling DQN):

```bash
python train_adaptive.py --algo dueling --episodes 300 --seeds 42 43 44 45 46
```

Run a fast compatibility evaluation of the archived enriched checkpoints:

```bash
python evaluate_models.py --training-mode enriched --train-seeds 42 43 44 45 46 --smoke --device cpu
```

Run the complete deterministic evaluation:

```bash
python evaluate_models.py --training-mode enriched --train-seeds 42 43 44 45 46 --eval-reps 20
```

Generate the paper figures from archived CSV results:

```bash
python -m analysis.make_paper_figures
```

All commands should be launched from the repository root. Random seeds, deterministic evaluation actions, confidence-interval aggregation, and scenario definitions are recorded in the scripts.

## Selected archived result

In the natural Markov scenario, the archived five-seed enriched-model evaluation reports mean 90-day profits of approximately 6.71M (DQN), 6.74M (DDQN), and 6.81M for Dueling DQN, D3QN, and PPO, with fill rates above 99.99%. These values are summaries of simulation output, not claims about a deployed supply chain. See `evaluation_results/evaluation_algorithm_summary.csv` for every scenario and metric.

## Analysis entry points

- `analysis/calibrate_parameters.py`: environment calibration experiments.
- `analysis/evaluate_stress_actions.py`: policy behavior during severe shocks.
- `analysis/evaluate_recovery_dynamics.py`: daily recovery trajectories after a seven-day disruption.
- `analysis/compare_training_modes.py`: natural vs. fixed-enriched vs. adaptive ablation.
- `analysis/make_paper_figures.py`: publication figures from archived results.

## Reproducibility notes

- Evaluation is deterministic with respect to action selection; stochastic scenarios use explicit seed schedules.
- Training and evaluation outputs are written to `training_logs/`, `checkpoints/`, and `evaluation_results/`.
- PyTorch checkpoints should only be loaded from trusted sources.
- The environment contains clearly marked calibration assumptions where source material did not specify numerical parameters.

## License

No license has been selected yet. Add an explicit open-source license before inviting reuse; until then, normal copyright restrictions apply.
