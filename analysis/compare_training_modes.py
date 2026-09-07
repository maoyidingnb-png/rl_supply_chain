
"""
compare_training_modes.py

Compare Natural vs Fixed-Enriched vs Adaptive training for the SAME algorithm.

Expected folders:
    evaluation_results_natural/evaluation_algorithm_summary.csv
    evaluation_results_enriched/evaluation_algorithm_summary.csv
    evaluation_results_adaptive/evaluation_algorithm_summary.csv

Run:
    python compare_training_modes.py --algorithm dueling

Outputs:
    ablation_results/
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MODES = ["natural", "enriched", "adaptive"]
MODE_LABELS = {
    "natural": "Natural",
    "enriched": "Fixed-Enriched",
    "adaptive": "Adaptive",
}

KEY_SCENARIOS = [
    "Control_Normal",
    "Natural_Markov",
    "Supplier_A3_7d",
    "Transport_U3_7d",
    "Compound_A3U3_3d",
]


def load_mode(mode):
    path = Path(f"evaluation_results_{mode}") / "evaluation_algorithm_summary.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run evaluate_models_adaptive.py "
            f"with --training-mode {mode} --output-dir evaluation_results_{mode}"
        )
    df = pd.read_csv(path)
    df["training_mode"] = mode
    df["algorithm"] = df["algorithm"].str.lower()
    return df


def save(fig, out, stem):
    fig.tight_layout()
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{stem}.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", default="dueling")
    args = parser.parse_args()

    algorithm = args.algorithm.lower()
    out = Path("ablation_results")
    out.mkdir(parents=True, exist_ok=True)

    frames = [load_mode(mode) for mode in MODES]
    all_df = pd.concat(frames, ignore_index=True)

    sub = all_df[all_df["algorithm"] == algorithm].copy()
    if sub.empty:
        raise ValueError(f"No rows found for algorithm={algorithm}")

    metrics = [
        "profit_mean",
        "profit_ci95",
        "fill_rate_mean",
        "fill_rate_ci95",
        "lost_units_mean",
        "lost_units_ci95",
        "expedite_cost_mean",
        "expedite_cost_ci95",
    ]
    keep = [
        "training_mode",
        "algorithm",
        "scenario",
    ] + [m for m in metrics if m in sub.columns]

    result = sub[keep].sort_values(
        ["scenario", "training_mode"]
    )
    result.to_csv(
        out / f"{algorithm}_training_mode_ablation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ------------------------------------------------------------
    # Figure 1: Normal-efficiency vs severe-resilience trade-off
    # x = Control_Normal profit
    # y = mean A3/U3 fill rate
    # ------------------------------------------------------------
    rows = []
    for mode in MODES:
        m = sub[sub["training_mode"] == mode]

        control = m[m["scenario"] == "Control_Normal"]
        a3 = m[m["scenario"] == "Supplier_A3_7d"]
        u3 = m[m["scenario"] == "Transport_U3_7d"]

        if control.empty or a3.empty or u3.empty:
            continue

        rows.append({
            "mode": mode,
            "control_profit": float(control.iloc[0]["profit_mean"]),
            "severe_fill": 0.5 * (
                float(a3.iloc[0]["fill_rate_mean"])
                + float(u3.iloc[0]["fill_rate_mean"])
            ),
            "severe_profit": 0.5 * (
                float(a3.iloc[0]["profit_mean"])
                + float(u3.iloc[0]["profit_mean"])
            ),
            "severe_lost": 0.5 * (
                float(a3.iloc[0]["lost_units_mean"])
                + float(u3.iloc[0]["lost_units_mean"])
            ),
        })

    trade = pd.DataFrame(rows)
    trade.to_csv(
        out / f"{algorithm}_efficiency_resilience_tradeoff.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if len(trade):
        fig, ax = plt.subplots(figsize=(7.5, 5.2))
        for _, r in trade.iterrows():
            ax.scatter(
                r["control_profit"] / 1e6,
                r["severe_fill"] * 100,
                s=65,
            )
            ax.annotate(
                MODE_LABELS[r["mode"]],
                (r["control_profit"] / 1e6, r["severe_fill"] * 100),
                xytext=(6, 5),
                textcoords="offset points",
            )
        ax.set_xlabel("Control-Normal operating contribution (million)")
        ax.set_ylabel("Mean fill rate under A3/U3 severe shocks (%)")
        ax.set_title(
            f"{algorithm}: normal efficiency vs severe-shock resilience"
        )
        ax.grid(alpha=0.2)
        save(fig, out, f"{algorithm}_efficiency_vs_resilience")

    # ------------------------------------------------------------
    # Figure 2: Severe scenario profit comparison
    # ------------------------------------------------------------
    severe_scenarios = ["Supplier_A3_7d", "Transport_U3_7d"]
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    x = np.arange(len(severe_scenarios))
    width = 0.24

    for j, mode in enumerate(MODES):
        vals = []
        cis = []
        for sc in severe_scenarios:
            m = sub[
                (sub["training_mode"] == mode)
                & (sub["scenario"] == sc)
            ]
            if m.empty:
                vals.append(np.nan)
                cis.append(0.0)
            else:
                vals.append(float(m.iloc[0]["profit_mean"]) / 1e6)
                cis.append(float(m.iloc[0].get("profit_ci95", 0.0)) / 1e6)

        ax.bar(
            x + (j - 1) * width,
            vals,
            width=width,
            yerr=cis,
            capsize=3,
            label=MODE_LABELS[mode],
        )

    ax.set_xticks(x, ["Supplier A3, 7d", "Transport U3, 7d"])
    ax.set_ylabel("Operating contribution (million)")
    ax.set_title(f"{algorithm}: training-distribution ablation")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    save(fig, out, f"{algorithm}_severe_profit_ablation")

    # ------------------------------------------------------------
    # Figure 3: Key scenario profit degradation relative to
    # each mode's own Control_Normal.
    # ------------------------------------------------------------
    heat_scenarios = [
        "Natural_Markov",
        "Supplier_A3_7d",
        "Transport_U3_7d",
        "Compound_A3U3_3d",
    ]

    mat = np.full((len(MODES), len(heat_scenarios)), np.nan)

    for i, mode in enumerate(MODES):
        m = sub[sub["training_mode"] == mode]
        control = m[m["scenario"] == "Control_Normal"]
        if control.empty:
            continue

        base = float(control.iloc[0]["profit_mean"])

        for j, sc in enumerate(heat_scenarios):
            row = m[m["scenario"] == sc]
            if row.empty:
                continue
            val = float(row.iloc[0]["profit_mean"])
            mat[i, j] = 100.0 * (val - base) / base

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    im = ax.imshow(mat, aspect="auto")
    ax.set_yticks(
        np.arange(len(MODES)),
        [MODE_LABELS[m] for m in MODES],
    )
    ax.set_xticks(
        np.arange(len(heat_scenarios)),
        heat_scenarios,
        rotation=25,
        ha="right",
    )
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                ax.text(
                    j,
                    i,
                    f"{mat[i,j]:.2f}%",
                    ha="center",
                    va="center",
                    fontsize=9,
                )
    fig.colorbar(im, ax=ax, label="Profit change vs Control_Normal (%)")
    ax.set_title(f"{algorithm}: robustness degradation by training mode")
    save(fig, out, f"{algorithm}_robustness_heatmap")

    print("=" * 72)
    print("ABLATION COMPARISON COMPLETE")
    print("=" * 72)
    print("Output:", out.resolve())
    for p in sorted(out.glob("*")):
        print(" ", p.name)


if __name__ == "__main__":
    main()
