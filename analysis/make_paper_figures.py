
"""
make_paper_figures_v2.py

Run from:
    D:\暑假科研\rl_supply_chain

Command:
    python make_paper_figures_v2.py

Required:
    pandas
    numpy
    matplotlib

Reads existing experiment CSV files only. It does NOT train or re-evaluate models.

Expected inputs:
    evaluation_results/evaluation_algorithm_summary.csv
    evaluation_results/evaluation_by_train_seed.csv
    evaluation_results/evaluation_action_summary.csv
    training_logs/training_log_<algo>_enriched_seed<seed>.csv

Optional recovery inputs (first existing file is used):
    evaluation_results/recovery_daily_summary_corrected.csv
    evaluation_results/recovery_dynamics_daily.csv

Outputs:
    paper_figures_v2/*.pdf
    paper_figures_v2/*.png
    paper_figures_v2/robustness_vs_control.csv
    paper_figures_v2/paired_algorithm_differences.csv
"""

from pathlib import Path
import math
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(".")
EVAL_DIR = ROOT / "evaluation_results"
TRAIN_DIR = ROOT / "training_logs"
OUT = ROOT / "paper_figures_v2"
OUT.mkdir(parents=True, exist_ok=True)

ALGO_ORDER = ["dqn", "ddqn", "dueling", "d3qn", "ppo"]
ALGO_LABEL = {
    "dqn": "DQN",
    "ddqn": "DDQN",
    "dueling": "Dueling DQN",
    "d3qn": "D3QN",
    "ppo": "PPO",
}
SEEDS = [42, 43, 44, 45, 46]
TCRIT_5 = 2.7764451051977987

SCENARIO_ORDER = [
    "Natural_Markov",
    "Control_Normal",
    "Supplier_A1_14d",
    "Supplier_A2_14d",
    "Supplier_A3_7d",
    "Transport_U1_5d",
    "Transport_U2_5d",
    "Transport_U3_7d",
    "Compound_A2U2_5d",
    "Compound_A3U3_3d",
]

def save(fig, stem):
    fig.tight_layout()
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=250, bbox_inches="tight")
    plt.close(fig)

def ci95(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) <= 1:
        return 0.0
    return TCRIT_5 * vals.std(ddof=1) / np.sqrt(len(vals))

def require(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return path

summary_path = require(EVAL_DIR / "evaluation_algorithm_summary.csv")
by_seed_path = require(EVAL_DIR / "evaluation_by_train_seed.csv")
action_path = require(EVAL_DIR / "evaluation_action_summary.csv")

summary = pd.read_csv(summary_path)
by_seed = pd.read_csv(by_seed_path)
actions = pd.read_csv(action_path)

summary["algorithm"] = summary["algorithm"].str.lower()
by_seed["algorithm"] = by_seed["algorithm"].str.lower()
actions["algorithm"] = actions["algorithm"].str.lower()

# -------------------------------------------------------------------
# Figure 1: training convergence, rolling-20-episode profit
# -------------------------------------------------------------------
train_frames = []
for algo in ALGO_ORDER:
    for seed in SEEDS:
        candidates = [
            TRAIN_DIR / f"training_log_{algo}_enriched_seed{seed}.csv",
            TRAIN_DIR / f"training_log_{algo}_seed{seed}.csv",
        ]
        f = next((p for p in candidates if p.exists()), None)
        if f is None:
            warnings.warn(f"Training log missing for {algo}, seed {seed}; training curve may be skipped.")
            continue
        df = pd.read_csv(f)
        if not {"episode", "profit"}.issubset(df.columns):
            warnings.warn(f"{f} lacks episode/profit columns.")
            continue
        tmp = df[["episode", "profit"]].copy()
        tmp["profit_roll"] = tmp["profit"].rolling(20, min_periods=1).mean()
        tmp["algorithm"] = algo
        tmp["train_seed"] = seed
        train_frames.append(tmp)

if train_frames:
    tr = pd.concat(train_frames, ignore_index=True)
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    for algo in ALGO_ORDER:
        sub = tr[tr["algorithm"] == algo]
        if sub.empty:
            continue
        pivot = sub.pivot(index="episode", columns="train_seed", values="profit_roll")
        mean = pivot.mean(axis=1)
        n = pivot.count(axis=1)
        sd = pivot.std(axis=1, ddof=1)
        ci = TCRIT_5 * sd / np.sqrt(n)
        line, = ax.plot(mean.index, mean.values / 1e6, label=ALGO_LABEL[algo])
        ax.fill_between(
            mean.index,
            (mean - ci).values / 1e6,
            (mean + ci).values / 1e6,
            alpha=0.12,
            color=line.get_color(),
        )
    ax.set_xlabel("Training episode")
    ax.set_ylabel("20-episode rolling operating contribution (million)")
    ax.set_title("Training convergence across independent seeds")
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.2)
    save(fig, "fig01_training_profit_curve")

# -------------------------------------------------------------------
# Figure 2: Natural Markov profit with 95% CI
# -------------------------------------------------------------------
nat = summary[summary["scenario"] == "Natural_Markov"].copy()
nat["order"] = nat["algorithm"].map({a:i for i,a in enumerate(ALGO_ORDER)})
nat = nat.sort_values("order")

fig, ax = plt.subplots(figsize=(8.2, 5.2))
x = np.arange(len(nat))
ax.errorbar(
    x,
    nat["profit_mean"].to_numpy() / 1e6,
    yerr=nat["profit_ci95"].to_numpy() / 1e6,
    fmt="o",
    capsize=5,
)
ax.set_xticks(x, [ALGO_LABEL[a] for a in nat["algorithm"]], rotation=15)
ax.set_ylabel("Operating contribution (million)")
ax.set_title("Natural Markov: mean ± 95% CI across training seeds")
ax.grid(axis="y", alpha=0.2)
save(fig, "fig02_natural_profit_ci")

# -------------------------------------------------------------------
# Figure 3: Profit degradation relative to each algorithm's Control_Normal
# -------------------------------------------------------------------
control = (
    summary[summary["scenario"] == "Control_Normal"]
    .set_index("algorithm")["profit_mean"]
)

rob_rows = []
for _, row in summary.iterrows():
    algo = row["algorithm"]
    if algo not in control.index:
        continue
    base = float(control.loc[algo])
    pct_change = 100.0 * (float(row["profit_mean"]) - base) / base
    rob_rows.append({
        "algorithm": algo,
        "scenario": row["scenario"],
        "profit_change_vs_control_pct": pct_change,
        "profit_mean": row["profit_mean"],
        "fill_rate_mean": row.get("fill_rate_mean", np.nan),
        "lost_units_mean": row.get("lost_units_mean", np.nan),
        "expedite_cost_mean": row.get("expedite_cost_mean", np.nan),
    })
rob = pd.DataFrame(rob_rows)
rob.to_csv(OUT / "robustness_vs_control.csv", index=False, encoding="utf-8-sig")

heat_scenarios = [s for s in SCENARIO_ORDER if s not in ("Natural_Markov", "Control_Normal")]
mat = np.full((len(ALGO_ORDER), len(heat_scenarios)), np.nan)
for i, algo in enumerate(ALGO_ORDER):
    for j, sc in enumerate(heat_scenarios):
        m = rob[(rob["algorithm"] == algo) & (rob["scenario"] == sc)]
        if len(m):
            mat[i, j] = float(m.iloc[0]["profit_change_vs_control_pct"])

fig, ax = plt.subplots(figsize=(11.0, 4.6))
im = ax.imshow(mat, aspect="auto")
ax.set_yticks(np.arange(len(ALGO_ORDER)), [ALGO_LABEL[a] for a in ALGO_ORDER])
ax.set_xticks(np.arange(len(heat_scenarios)), heat_scenarios, rotation=35, ha="right")
ax.set_title("Profit change relative to each policy's Control_Normal baseline (%)")
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        if np.isfinite(mat[i, j]):
            ax.text(j, i, f"{mat[i,j]:.2f}%", ha="center", va="center", fontsize=8)
fig.colorbar(im, ax=ax, label="Profit change (%)")
save(fig, "fig03_profit_degradation_heatmap")

# -------------------------------------------------------------------
# Figure 4: severe A3/U3 absolute profit
# -------------------------------------------------------------------
severe = summary[summary["scenario"].isin(["Supplier_A3_7d", "Transport_U3_7d"])].copy()
fig, ax = plt.subplots(figsize=(8.5, 5.2))
width = 0.36
x = np.arange(len(ALGO_ORDER))
for offset, sc in [(-width/2, "Supplier_A3_7d"), (width/2, "Transport_U3_7d")]:
    vals = []
    for algo in ALGO_ORDER:
        m = severe[(severe["scenario"] == sc) & (severe["algorithm"] == algo)]
        vals.append(float(m.iloc[0]["profit_mean"]) / 1e6)
    ax.bar(x + offset, vals, width=width, label=sc)
ax.set_xticks(x, [ALGO_LABEL[a] for a in ALGO_ORDER], rotation=15)
ax.set_ylabel("Operating contribution (million)")
ax.set_title("Severe disruption: absolute economic performance")
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.2)
save(fig, "fig04_severe_profit")

# -------------------------------------------------------------------
# Figure 5: cost-service trade-off in severe A3
# -------------------------------------------------------------------
a3 = summary[summary["scenario"] == "Supplier_A3_7d"].copy()
fig, ax = plt.subplots(figsize=(7.6, 5.2))
for _, row in a3.iterrows():
    ax.scatter(row["expedite_cost_mean"]/1000, row["lost_units_mean"])
    ax.annotate(
        ALGO_LABEL[row["algorithm"]],
        (row["expedite_cost_mean"]/1000, row["lost_units_mean"]),
        xytext=(5, 4),
        textcoords="offset points",
    )
ax.set_xlabel("Expedite cost (thousand)")
ax.set_ylabel("Lost units")
ax.set_title("Supplier A3 (7d): cost-service trade-off")
ax.grid(alpha=0.2)
save(fig, "fig05_a3_cost_service_tradeoff")

# -------------------------------------------------------------------
# Figure 6: action decomposition from 18 actions
# decode action: l = a//6, e=(a%6)//3, g=(a%6)%3
# -------------------------------------------------------------------
def action_component_shares(action_row):
    shares = np.array([float(action_row[f"action_{a}_share"]) for a in range(18)])
    l = np.zeros(3)
    e = np.zeros(2)
    g = np.zeros(3)
    for a, s in enumerate(shares):
        ll = a // 6
        rem = a % 6
        ee = rem // 3
        gg = rem % 3
        l[ll] += s
        e[ee] += s
        g[gg] += s
    return l, e, g

for scenario, stem in [
    ("Natural_Markov", "natural"),
    ("Supplier_A3_7d", "a3"),
    ("Transport_U3_7d", "u3"),
]:
    sub = actions[actions["scenario"] == scenario].copy()
    if sub.empty:
        continue
    components = {"l": [], "e": [], "g": []}
    present_algos = []
    for algo in ALGO_ORDER:
        m = sub[sub["algorithm"] == algo]
        if m.empty:
            continue
        l, e, g = action_component_shares(m.iloc[0])
        components["l"].append(l)
        components["e"].append(e)
        components["g"].append(g)
        present_algos.append(algo)

    for key, labels, title in [
        ("l", ["l0", "l1", "l2"], "Inventory-buffer choice"),
        ("e", ["e0", "e1"], "Expedition request"),
        ("g", ["g0", "g1", "g2"], "CW allocation rule"),
    ]:
        data = np.vstack(components[key])
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        bottom = np.zeros(len(present_algos))
        xx = np.arange(len(present_algos))
        for j, lab in enumerate(labels):
            ax.bar(xx, data[:, j] * 100, bottom=bottom * 100, label=lab)
            bottom += data[:, j]
        ax.set_xticks(xx, [ALGO_LABEL[a] for a in present_algos], rotation=15)
        ax.set_ylabel("Decision share (%)")
        ax.set_ylim(0, 100)
        ax.set_title(f"{scenario}: {title}")
        ax.legend(frameon=False, ncol=len(labels))
        save(fig, f"fig06_{stem}_{key}_share")

# -------------------------------------------------------------------
# Figure 7: stress-vs-recovery metrics already stored by formal evaluation
# Whole-horizon values can dilute short shocks, so expose phase metrics.
# -------------------------------------------------------------------
for sc in ["Supplier_A3_7d", "Transport_U3_7d"]:
    sub = summary[summary["scenario"] == sc].copy()
    if {"stress_fill_rate_mean", "recovery_fill_rate_mean"}.issubset(sub.columns):
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        xx = np.arange(len(ALGO_ORDER))
        width = 0.36
        stress_vals, recovery_vals = [], []
        for algo in ALGO_ORDER:
            m = sub[sub["algorithm"] == algo]
            stress_vals.append(float(m.iloc[0]["stress_fill_rate_mean"]) * 100)
            recovery_vals.append(float(m.iloc[0]["recovery_fill_rate_mean"]) * 100)
        ax.bar(xx-width/2, stress_vals, width=width, label="Stress phase")
        ax.bar(xx+width/2, recovery_vals, width=width, label="Remaining recovery phase")
        ax.set_xticks(xx, [ALGO_LABEL[a] for a in ALGO_ORDER], rotation=15)
        ax.set_ylabel("Fill rate (%)")
        ax.set_title(f"{sc}: phase-specific service level")
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.2)
        save(fig, f"fig07_{sc.lower()}_phase_fill")

# -------------------------------------------------------------------
# Figure 8: corrected daily recovery curve, if available
# -------------------------------------------------------------------
recovery_candidates = [
    EVAL_DIR / "recovery_daily_summary_corrected.csv",
    EVAL_DIR / "recovery_dynamics_daily.csv",
]
recovery_file = next((p for p in recovery_candidates if p.exists()), None)

if recovery_file is not None:
    rd = pd.read_csv(recovery_file)
    rd["algorithm"] = rd["algorithm"].str.lower()

    # Normalize column names where possible.
    if "fill_rate_mean" not in rd.columns and "fill_rate" in rd.columns:
        # If raw-ish daily data exists, aggregate eval reps within train seed first,
        # then obtain mean and 95% CI across training seeds.
        if {"train_seed", "scenario", "day", "fill_rate"}.issubset(rd.columns):
            seed_daily = (
                rd.groupby(["algorithm","train_seed","scenario","day"], as_index=False)
                  ["fill_rate"].mean()
            )
            rows = []
            for keys, g in seed_daily.groupby(["algorithm","scenario","day"]):
                algo, sc, day = keys
                vals = g["fill_rate"].to_numpy(float)
                rows.append({
                    "algorithm": algo,
                    "scenario": sc,
                    "day": day,
                    "fill_rate_mean": vals.mean(),
                    "fill_rate_ci95": ci95(vals),
                })
            rd = pd.DataFrame(rows)

    if {"algorithm","scenario","day","fill_rate_mean"}.issubset(rd.columns):
        for sc in ["Supplier_A3_7d", "Transport_U3_7d"]:
            sub = rd[rd["scenario"] == sc].copy()
            if sub.empty:
                continue
            fig, ax = plt.subplots(figsize=(9.0, 5.2))
            for algo in ALGO_ORDER:
                m = sub[sub["algorithm"] == algo].sort_values("day")
                if m.empty:
                    continue
                line, = ax.plot(m["day"], m["fill_rate_mean"]*100, label=ALGO_LABEL[algo])
                if "fill_rate_ci95" in m.columns:
                    ax.fill_between(
                        m["day"],
                        (m["fill_rate_mean"]-m["fill_rate_ci95"])*100,
                        (m["fill_rate_mean"]+m["fill_rate_ci95"])*100,
                        alpha=0.10,
                        color=line.get_color(),
                    )
            ax.axvspan(1, 7, alpha=0.08)
            ax.axvline(7.5, linestyle="--", linewidth=1)
            ax.set_xlabel("Day")
            ax.set_ylabel("Fill rate (%)")
            ax.set_title(f"{sc}: day-by-day shock and recovery")
            ax.legend(frameon=False, ncol=2)
            ax.grid(axis="y", alpha=0.2)
            save(fig, f"fig08_{sc.lower()}_daily_recovery")
else:
    warnings.warn(
        "No recovery daily CSV found. Run corrected recovery summarization first "
        "to generate day-by-day recovery figures."
    )

# -------------------------------------------------------------------
# Paired algorithm differences across matched training seed IDs
# This is descriptive because n=5.
# Reference = Dueling DQN; metrics in key scenarios.
# -------------------------------------------------------------------
paired_rows = []
reference = "dueling"
for sc in ["Natural_Markov", "Supplier_A3_7d", "Transport_U3_7d"]:
    sub = by_seed[by_seed["scenario"] == sc]
    ref = sub[sub["algorithm"] == reference].set_index("train_seed")
    for algo in ALGO_ORDER:
        if algo == reference:
            continue
        cur = sub[sub["algorithm"] == algo].set_index("train_seed")
        common = sorted(set(ref.index) & set(cur.index))
        for metric in ["profit", "fill_rate", "lost_units", "expedite_cost"]:
            if metric not in ref.columns or metric not in cur.columns or not common:
                continue
            # Defined as algorithm - Dueling.
            diff = cur.loc[common, metric].to_numpy(float) - ref.loc[common, metric].to_numpy(float)
            paired_rows.append({
                "scenario": sc,
                "algorithm": algo,
                "reference": reference,
                "metric": metric,
                "n_pairs": len(diff),
                "mean_difference_algo_minus_reference": diff.mean(),
                "ci95_halfwidth": ci95(diff),
            })

pd.DataFrame(paired_rows).to_csv(
    OUT / "paired_algorithm_differences.csv",
    index=False,
    encoding="utf-8-sig",
)

print("=" * 78)
print("PAPER FIGURES V2 COMPLETE")
print("=" * 78)
print("Output folder:", OUT.resolve())
for p in sorted(OUT.glob("*.pdf")):
    print("  ", p.name)
print("\nTables:")
print("  robustness_vs_control.csv")
print("  paired_algorithm_differences.csv")
