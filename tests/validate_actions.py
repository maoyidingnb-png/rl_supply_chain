import numpy as np
import pandas as pd

from env import SupplyChainEnv


N_REPLICATIONS = 20
BASE_SEED = 2000


def configure_env(seed, A_t, U_t):

    env = SupplyChainEnv(seed=seed)

    # ==========================================
    # Moderate calibrated environment
    # ==========================================

    env.price = 100.0

    env.holding_cost_cdc = (
        100.0 * 0.20 / 365.0
    )

    env.holding_cost_rdc = (
        100.0 * 0.30 / 365.0
    )

    env.penalty_lost_sales = 150.0

    env.expedite_cost_per_order = 500.0

    env.supplier_capacity_per_product = (
        np.array(
            [225.0, 150.0, 75.0],
            dtype=np.float64
        )
    )

    env.cdc_capacity = 3000.0
    env.rdc_capacity = 300.0

    env.s_cdc_base = 600.0
    env.S_cdc_base = 1500.0

    env.s_rdc_base = 30.0
    env.S_rdc_base = 75.0

    # 固定场景
    env.set_disruption_override(
        A_t=A_t,
        U_t=U_t
    )

    env.reset(seed=seed)

    # Moderate initial inventory
    env.cdc_inventory[:] = 900.0
    env.rdc_inventory[:] = 60.0
    env.cdc_pipeline[:] = 0.0

    return env


def run_one(action, A_t, U_t, seed):

    env = configure_env(
        seed=seed,
        A_t=A_t,
        U_t=U_t
    )

    total_profit = 0.0
    total_demand = 0.0
    total_sales = 0.0
    total_lost = 0.0

    total_holding = 0.0
    total_expedite = 0.0

    l, e, g = env.decode_action(action)

    while True:

        (
            obs,
            reward,
            done,
            truncated,
            info
        ) = env.step(action)

        total_profit += info["profit"]

        total_demand += (
            info["demand_units"]
        )

        total_sales += (
            info["sales_units"]
        )

        total_lost += (
            info["lost_units"]
        )

        total_holding += (
            info["holding_cost"]
        )

        total_expedite += (
            info["expedite_cost"]
        )

        if done or truncated:
            break

    fill_rate = (
        total_sales / total_demand
    )

    return {
        "action": action,
        "l": l,
        "e": e,
        "g": g,

        "profit": total_profit,
        "fill_rate": fill_rate,
        "lost_units": total_lost,

        "holding_cost":
            total_holding,

        "expedite_cost":
            total_expedite,
    }


def evaluate_scenario(
    scenario_name,
    A_t,
    U_t
):

    print("\n" + "=" * 100)
    print("SCENARIO:", scenario_name)
    print("=" * 100)

    results = []

    for action in range(18):

        # 完全中断下跳过非法e=1动作
        env_temp = configure_env(
            BASE_SEED,
            A_t,
            U_t
        )

        mask = env_temp.action_masks()

        if not mask[action]:
            continue

        for rep in range(
            N_REPLICATIONS
        ):

            seed = (
                BASE_SEED + rep
            )

            result = run_one(
                action=action,
                A_t=A_t,
                U_t=U_t,
                seed=seed
            )

            result["scenario"] = (
                scenario_name
            )

            result["seed"] = seed

            results.append(result)

    df = pd.DataFrame(results)

    summary = (
        df.groupby(
            [
                "action",
                "l",
                "e",
                "g"
            ]
        )
        .agg(
            profit_mean=(
                "profit",
                "mean"
            ),

            fill_rate_mean=(
                "fill_rate",
                "mean"
            ),

            lost_mean=(
                "lost_units",
                "mean"
            ),

            holding_mean=(
                "holding_cost",
                "mean"
            ),

            expedite_mean=(
                "expedite_cost",
                "mean"
            ),
        )
        .reset_index()
    )

    summary = summary.sort_values(
        "profit_mean",
        ascending=False
    )

    print(
        summary.to_string(
            index=False,
            formatters={
                "profit_mean":
                    lambda x: f"{x:,.0f}",

                "fill_rate_mean":
                    lambda x: f"{x:.4f}",

                "lost_mean":
                    lambda x: f"{x:,.1f}",

                "holding_mean":
                    lambda x: f"{x:,.0f}",

                "expedite_mean":
                    lambda x: f"{x:,.0f}",
            }
        )
    )

    return df, summary


def main():

    scenarios = {

        "Normal": (
            0,
            0
        ),

        "Transport_Delay": (
            0,
            1
        ),

        "Severe_Capacity": (
            2,
            0
        ),

        "Compound": (
            2,
            1
        ),
    }

    all_raw = []
    all_summary = []

    for (
        name,
        (
            A_t,
            U_t
        )
    ) in scenarios.items():

        raw, summary = (
            evaluate_scenario(
                scenario_name=name,
                A_t=A_t,
                U_t=U_t
            )
        )

        raw["scenario"] = name
        summary["scenario"] = name

        all_raw.append(raw)
        all_summary.append(summary)

    raw_df = pd.concat(
        all_raw,
        ignore_index=True
    )

    summary_df = pd.concat(
        all_summary,
        ignore_index=True
    )

    raw_df.to_csv(
        "action_validation_raw.csv",
        index=False
    )

    summary_df.to_csv(
        "action_validation_summary.csv",
        index=False
    )

    print("\n" + "=" * 100)
    print("ACTION VALIDATION COMPLETE")
    print("=" * 100)

    print(
        "Saved: action_validation_raw.csv"
    )

    print(
        "Saved: action_validation_summary.csv"
    )


if __name__ == "__main__":
    main()
