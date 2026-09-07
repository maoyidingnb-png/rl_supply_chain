import numpy as np
import pandas as pd

from env import SupplyChainEnv


def run_scenario(
    name,
    seed=42,
    A_t=None,
    U_t=None,
    action=0,
    output_csv=None
):
    """
    运行一个90天场景，并检查经济核算是否一致。

    action=0:
        l=0
        e=0
        g=0

    因此可以看作最基础的固定(s,S)策略环境测试。
    """

    print("\n" + "=" * 75)
    print(name)
    print("=" * 75)

    env = SupplyChainEnv(seed=seed)

    # --------------------------------------------------
    # 场景设置
    # --------------------------------------------------

    if A_t is not None or U_t is not None:
        env.set_disruption_override(
            A_t=A_t,
            U_t=U_t
        )
    else:
        env.clear_disruption_override()

    obs, info = env.reset(seed=seed)

    records = []

    day = 0

    while True:

        obs, reward, done, truncated, info = env.step(action)

        day += 1

        # ==================================================
        # TEST 1
        # Demand accounting
        # ==================================================

        demand = info["demand_units"]
        sales = info["sales_units"]
        lost = info["lost_units"]

        assert np.isclose(
            sales + lost,
            demand,
            atol=1e-6
        ), (
            f"[FAIL] Day {day}: "
            f"sales + lost != demand\n"
            f"{sales} + {lost} != {demand}"
        )

        # ==================================================
        # TEST 2
        # Fill Rate
        # ==================================================

        expected_fill_rate = (
            sales / demand
            if demand > 0
            else 1.0
        )

        assert np.isclose(
            info["fill_rate"],
            expected_fill_rate,
            atol=1e-8
        ), (
            f"[FAIL] Day {day}: "
            "fill rate incorrect"
        )

        # ==================================================
        # TEST 3
        # Holding cost accounting
        # ==================================================

        expected_holding = (
            info["holding_cost_cdc"]
            + info["holding_cost_rdc"]
        )

        assert np.isclose(
            info["holding_cost"],
            expected_holding,
            atol=1e-6
        ), (
            f"[FAIL] Day {day}: "
            "holding cost incorrect"
        )

        # ==================================================
        # TEST 4
        # Profit accounting
        # ==================================================

        expected_profit = (
            info["revenue"]
            - info["lost_sales_penalty"]
            - info["holding_cost"]
            - info["expedite_cost"]
        )

        assert np.isclose(
            info["profit"],
            expected_profit,
            atol=1e-6
        ), (
            f"[FAIL] Day {day}: "
            "profit accounting incorrect\n"
            f"reported = {info['profit']}\n"
            f"expected = {expected_profit}"
        )

        # ==================================================
        # TEST 5
        # Reward scaling
        # ==================================================

        expected_reward = (
            info["profit"] / 10000.0
        )

        assert np.isclose(
            reward,
            expected_reward,
            atol=1e-8
        ), (
            f"[FAIL] Day {day}: "
            "reward scaling incorrect"
        )

        # ==================================================
        # 保存当天真实数据
        # ==================================================

        records.append({
            "day": day,

            "A_t": info["A_t"],
            "U_t": info["U_t"],

            "action": info["action"],
            "l_t": info["l_t"],
            "e_t": info["executed_e_t"],
            "g_t": info["g_t"],

            "demand_units": demand,
            "sales_units": sales,
            "lost_units": lost,
            "fill_rate": info["fill_rate"],

            "revenue": info["revenue"],
            "lost_sales_penalty":
                info["lost_sales_penalty"],

            "holding_cost_cdc":
                info["holding_cost_cdc"],

            "holding_cost_rdc":
                info["holding_cost_rdc"],

            "holding_cost":
                info["holding_cost"],

            "expedite_cost":
                info["expedite_cost"],

            "profit":
                info["profit"],

            "reward":
                reward,

            "upstream_order_units":
                info["upstream_order_units"],

            "cdc_arrival_units":
                info["cdc_arrival_units"],

            "rdc_shipment_units":
                info["rdc_shipment_units"],

            "cdc_inventory_total":
                np.sum(
                    info["cdc_inventory"]
                ),

            "rdc_inventory_total":
                np.sum(
                    info["rdc_inventory"]
                ),

            "pipeline_total":
                np.sum(
                    info["cdc_pipeline"]
                )
        })

        if done or truncated:
            break

    df = pd.DataFrame(records)

    # ==================================================
    # 保存CSV
    # ==================================================

    if output_csv is not None:
        df.to_csv(
            output_csv,
            index=False
        )

        print(
            "\nSaved:",
            output_csv
        )

    # ==================================================
    # 汇总指标
    # ==================================================

    total_demand = df["demand_units"].sum()
    total_sales = df["sales_units"].sum()
    total_lost = df["lost_units"].sum()

    overall_fill_rate = (
        total_sales / total_demand
    )

    total_revenue = df["revenue"].sum()

    total_lost_penalty = (
        df["lost_sales_penalty"].sum()
    )

    total_holding = (
        df["holding_cost"].sum()
    )

    total_expedite = (
        df["expedite_cost"].sum()
    )

    total_profit = (
        df["profit"].sum()
    )

    print("\n--- Demand / Service ---")

    print(
        "Days                  =",
        len(df)
    )

    print(
        "Average demand/day    =",
        round(
            df["demand_units"].mean(),
            2
        )
    )

    print(
        "Total demand          =",
        round(total_demand, 2)
    )

    print(
        "Total sales           =",
        round(total_sales, 2)
    )

    print(
        "Total lost units      =",
        round(total_lost, 2)
    )

    print(
        "Overall fill rate     =",
        round(
            overall_fill_rate,
            4
        )
    )

    print("\n--- Economic Results ---")

    print(
        "Total revenue         =",
        round(total_revenue, 2)
    )

    print(
        "Lost sales penalty    =",
        round(total_lost_penalty, 2)
    )

    print(
        "Holding cost          =",
        round(total_holding, 2)
    )

    print(
        "Expedite cost         =",
        round(total_expedite, 2)
    )

    print(
        "TOTAL PROFIT          =",
        round(total_profit, 2)
    )

    print(
        "Average profit/day    =",
        round(
            df["profit"].mean(),
            2
        )
    )

    # ==================================================
    # 理论需求数量级检查
    # ==================================================

    # 10 RDC × 5 products × Poisson(lambda=15)
    theoretical_demand = (
        env.num_rdc
        * env.num_products
        * 15
    )

    print("\n--- Demand sanity ---")

    print(
        "Theoretical E[D/day]  =",
        theoretical_demand
    )

    print(
        "Observed mean D/day   =",
        round(
            df["demand_units"].mean(),
            2
        )
    )

    print(
        "Difference            =",
        round(
            df["demand_units"].mean()
            - theoretical_demand,
            2
        )
    )

    print("\nPASS: all accounting identities verified")

    return df


def main():

    print("=" * 75)
    print("ECONOMIC SANITY CHECK")
    print("=" * 75)

    # ==================================================
    # Case 1
    # 完全正常环境
    # ==================================================

    normal_df = run_scenario(
        name="CASE 1: Normal environment",
        seed=42,
        A_t=0,
        U_t=0,
        action=0,
        output_csv="sanity_normal_daily.csv"
    )

    # ==================================================
    # Case 2
    # Markov随机环境
    # ==================================================

    markov_df = run_scenario(
        name="CASE 2: Markov disruption environment",
        seed=42,
        A_t=None,
        U_t=None,
        action=0,
        output_csv="sanity_markov_daily.csv"
    )

    print("\n" + "=" * 75)
    print("ALL ECONOMIC TESTS COMPLETED")
    print("=" * 75)


if __name__ == "__main__":
    main()
