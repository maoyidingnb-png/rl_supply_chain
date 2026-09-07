
"""
plot_adaptive_curriculum.py

Visualize how the adaptive risk distribution changes during training.

Run:
    python plot_adaptive_curriculum.py --algorithm dueling --seed 42
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

RISK_LABELS = [
    "A1U0", "A2U0", "A3U0",
    "A0U1", "A0U2", "A0U3",
    "A2U2", "A3U3",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", default="dueling")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    path = Path("training_logs") / (
        f"curriculum_log_{args.algorithm}_adaptive_seed{args.seed}.csv"
    )
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    out = Path("adaptive_figures")
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    for label in RISK_LABELS:
        ax.plot(
            df["episode"],
            df[f"{label}_prob"] * 100,
            label=label,
        )

    ax.set_xlabel("Training episode")
    ax.set_ylabel("Conditional probability inside risk branch (%)")
    ax.set_title(
        f"{args.algorithm} seed={args.seed}: adaptive risk curriculum"
    )
    ax.legend(frameon=False, ncol=4)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()

    fig.savefig(
        out / f"{args.algorithm}_seed{args.seed}_curriculum_probs.pdf",
        bbox_inches="tight",
    )
    fig.savefig(
        out / f"{args.algorithm}_seed{args.seed}_curriculum_probs.png",
        dpi=250,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("Created:", out.resolve())


if __name__ == "__main__":
    main()
