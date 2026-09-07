import numpy as np
import pandas as pd

from env import SupplyChainEnv


# ============================================================
# 1. Experiment configuration
# ============================================================

N_REPLICATIONS = 20
BASE_SEED = 1000

# Fixed (s,S) baseline
#
# action 0:
# l=0, e=0, g=0

FIXED_ACTION = 0


# ============================================================
# 2. Economic calibration assumptions
#
# 【建议校准参数，不是原文件/老师规定】
# ============================================================

PRICE = 100.0

ANNUAL_HOLDING_RATE_CDC = 0.20
ANNUAL_HOLDING_RATE_RDC = 0.30

HOLDING_COST_CDC = (
    PRICE
    * ANNUAL_HOLDING_RATE_CDC
    / 365.0
)

HOLDING_COST_RDC = (
    PRICE
    * ANNUAL_HOLDING_RATE_RDC
    / 365.0
)

LOST_SALES_PENALTY = 150.0

EXPEDITE_COST = 500.0


# ============================================================
# 3. Supplier capacity
#
# 单位：
# units / product / day
#
# Expected demand：
# 10 RDC × lambda15
# = 150/product/day
#
# 【建议校准V1】
# ============================================================

SUPPLIER_CAPACITY = np.array(
    [
        225.0,  # Normal = 150%
        150.0,  # Mild = 100%
        75.0,   # Severe = 50%
    ],
    dtype=np.float64
)


# ============================================================
# 4. Candidate inventory systems
# ============================================================

PARAMETER_SETS = {

    "Lean": {

        "cdc_capacity":
            3000.0,

        "rdc_capacity":
            300.0,

        # CDC
        "s_cdc":
            450.0,     # 3 days

        "S_cdc":
            1200.0,    # 8 days

        # RDC
        "s_rdc":
            15.0,      # 1 day

        "S_rdc":
            60.0,      # 4 days

        # initial
        "initial_cdc":
            750.0,     # 5 days

        "initial_rdc":
            45.0,      # 3 days
    },

    "Moderate": {

        "cdc_capacity":
            3000.0,

        "rdc_capacity":
            300.0,

        "s_cdc":
            600.0,     # 4 days

        "S_cdc":
            1500.0,    # 10 days

        "s_rdc":
            30.0,      # 2 days

        "S_rdc":
            75.0,      # 5 days

        "initial_cdc":
            900.0,     # 6 days

        "initial_rdc":
            60.0,      # 4 days
    },

    "Conservative": {

        "cdc_capacity":
            3000.0,

        "rdc_capacity":
            300.0,

        "s_cdc":
            900.0,     # 6 days

        "S_cdc":
            1800.0,    # 12 days

        "s_rdc":
            45.0,      # 3 days

        "S_rdc":
            90.0,      # 6 days

        "initial_cdc":
            1200.0,    # 8 days

        "initial_rdc":
            75.0,      # 5 days
    },
}


# ============================================================
# 5. Scenarios
# ============================================================

SCENARIOS = {

    "Normal": {
        "A_t": 0,
        "U_t": 0,
    },

    "Mild_Capacity": {
        "A_t": 1,
        "U_t": 0,
    },

    "Severe_Capacity": {
        "A_t": 2,
        "U_t": 0,
    },

    "Transport_Delay": {
        "A_t": 0,
        "U_t": 1,
    },

    "Compound": {
        "A_t": 2,
        "U_t": 1,
    },
}


# ============================================================
# 6. Environment construction
# ============================================================

def create_environment(
    parameter_config,
    scenario_config,
    seed
):

    env = SupplyChainEnv(
        seed=seed
    )

    # --------------------------------------------------------
    # Economics
    # --------------------------------------------------------

    env.price = PRICE

    env.holding_cost_cdc = (
        HOLDING_COST_CDC
    )

    env.holding_cost_rdc = (
        HOLDING_COST_RDC
    )

    env.penalty_lost_sales = (
        LOST_SALES_PENALTY
    )

    env.expedite_cost_per_order = (
        EXPEDITE_COST
    )

    # --------------------------------------------------------
    # Supplier capacity
    # --------------------------------------------------------

    env.supplier_capacity_per_product = (
        SUPPLIER_CAPACITY.copy()
    )

    # --------------------------------------------------------
    # Physical capacity
    # --------------------------------------------------------

    env.cdc_capacity = (
        parameter_config[
            "cdc_capacity"
        ]
    )

    env.rdc_capacity = (
        parameter_config[
            "rdc_capacity"
        ]
    )

    # --------------------------------------------------------
    # (s,S)
    # --------------------------------------------------------

    env.s_cdc_base = (
        parameter_config[
            "s_cdc"
        ]
    )

    env.S_cdc_base = (
        parameter_config[
            "S_cdc"
        ]
    )

    env.s_rdc_base = (
        parameter_config[
            "s_rdc"
        ]
    )

    env.S_rdc_base = (
        parameter_config[
            "S_rdc"
        ]
    )

    # --------------------------------------------------------
    # Fixed scenario
    # --------------------------------------------------------

    env.set_disruption_override(
        A_t=scenario_config[
            "A_t"
        ],
        U_t=scenario_config[
            "U_t"
        ]
    )

    env.reset(
        seed=seed
    )

    # --------------------------------------------------------
    # Initial inventory
    # --------------------------------------------------------

    env.cdc_inventory[:] = (
        parameter_config[
            "initial_cdc"
        ]
    )

    env.rdc_inventory[:] = (
        parameter_config[
            "initial_rdc"
        ]
    )

    env.cdc_pipeline[:] = 0.0

    return env


# ============================================================
# 7. One replication
# ============================================================

def run_one_replication(
    parameter_name,
    parameter_config,
    scenario_name,
    scenario_config,
    seed
):

    env = create_environment(
        parameter_config,
        scenario_config,
        seed
    )

    total_profit = 0.0
    total_revenue = 0.0

    total_demand = 0.0
    total_sales = 0.0
    total_lost = 0.0

    total_holding = 0.0
    total_lost_penalty = 0.0

    total_requested_supply = 0.0
    total_actual_supply = 0.0
    total_capacity_shortfall = 0.0

    cdc_inventory_daily = []
    rdc_inventory_daily = []

    days = 0

    while True:

        (
            obs,
            reward,
            done,
            truncated,
            info
        ) = env.step(
            FIXED_ACTION
        )

        days += 1

        # --------------------------------------------------------
        # Accounting tests
        # --------------------------------------------------------

        assert np.isclose(
            info["sales_units"]
            + info["lost_units"],
            info["demand_units"]
        )

        expected_profit = (
            info["revenue"]
            - info["lost_sales_penalty"]
            - info["holding_cost"]
            - info["expedite_cost"]
        )

        assert np.isclose(
            info["profit"],
            expected_profit
        )

        # --------------------------------------------------------
        # Totals
        # --------------------------------------------------------

        total_profit += (
            info["profit"]
        )

        total_revenue += (
            info["revenue"]
        )

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

        total_lost_penalty += (
            info[
                "lost_sales_penalty"
            ]
        )

        total_requested_supply += (
            info[
                "requested_upstream_units"
            ]
        )

        total_actual_supply += (
            info[
                "supplier_fulfilled_units"
            ]
        )

        total_capacity_shortfall += (
            info[
                "supplier_unfulfilled_units"
            ]
        )

        cdc_inventory_daily.append(
            np.sum(
                info[
                    "cdc_inventory"
                ]
            )
        )

        rdc_inventory_daily.append(
            np.sum(
                info[
                    "rdc_inventory"
                ]
            )
        )

        if done or truncated:
            break

    fill_rate = (
        total_sales
        / total_demand
        if total_demand > 0
        else 1.0
    )

    return {

        "parameter_set":
            parameter_name,

        "scenario":
            scenario_name,

        "seed":
            seed,

        "days":
            days,

        # Demand/service
        "total_demand":
            total_demand,

        "total_sales":
            total_sales,

        "total_lost_units":
            total_lost,

        "fill_rate":
            fill_rate,

        # Economics
        "total_revenue":
            total_revenue,

        "total_lost_penalty":
            total_lost_penalty,

        "total_holding_cost":
            total_holding,

        "total_profit":
            total_profit,

        "avg_profit_per_day":
            total_profit / days,

        # Supplier
        "requested_supply":
            total_requested_supply,

        "actual_supply":
            total_actual_supply,

        "capacity_shortfall":
            total_capacity_shortfall,

        # Inventory
        "avg_cdc_inventory":
            np.mean(
                cdc_inventory_daily
            ),

        "avg_rdc_inventory":
            np.mean(
                rdc_inventory_daily
            ),
    }


# ============================================================
# 8. 95% confidence interval
# ============================================================

def ci95(series):

    x = np.asarray(
        series,
        dtype=float
    )

    if len(x) <= 1:
        return 0.0

    se = (
        np.std(
            x,
            ddof=1
        )
        / np.sqrt(
            len(x)
        )
    )

    return (
        1.96 * se
    )


# ============================================================
# 9. Summary
# ============================================================

def build_summary(
    raw_df
):

    rows = []

    groups = raw_df.groupby(
        [
            "parameter_set",
            "scenario"
        ],
        sort=False
    )

    for (
        parameter_set,
        scenario
    ), group in groups:

        rows.append({

            "parameter_set":
                parameter_set,

            "scenario":
                scenario,

            "n":
                len(group),

            # Fill
            "fill_rate_mean":
                group[
                    "fill_rate"
                ].mean(),

            "fill_rate_std":
                group[
                    "fill_rate"
                ].std(ddof=1),

            "fill_rate_ci95":
                ci95(
                    group[
                        "fill_rate"
                    ]
                ),

            # Profit
            "profit_mean":
                group[
                    "total_profit"
                ].mean(),

            "profit_std":
                group[
                    "total_profit"
                ].std(ddof=1),

            "profit_ci95":
                ci95(
                    group[
                        "total_profit"
                    ]
                ),

            # Lost
            "lost_units_mean":
                group[
                    "total_lost_units"
                ].mean(),

            # Holding
            "holding_cost_mean":
                group[
                    "total_holding_cost"
                ].mean(),

            # Lost penalty
            "lost_penalty_mean":
                group[
                    "total_lost_penalty"
                ].mean(),

            # Supplier
            "requested_supply_mean":
                group[
                    "requested_supply"
                ].mean(),

            "actual_supply_mean":
                group[
                    "actual_supply"
                ].mean(),

            "capacity_shortfall_mean":
                group[
                    "capacity_shortfall"
                ].mean(),

            # Inventory
            "avg_cdc_inventory":
                group[
                    "avg_cdc_inventory"
                ].mean(),

            "avg_rdc_inventory":
                group[
                    "avg_rdc_inventory"
                ].mean(),
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# 10. Console output
# ============================================================

def print_summary(
    summary_df
):

    print(
        "\n"
        + "=" * 120
    )

    print(
        "PARAMETER CALIBRATION SUMMARY"
    )

    print(
        "=" * 120
    )

    for parameter_name in (
        PARAMETER_SETS.keys()
    ):

        print(
            "\n"
            + "#" * 120
        )

        print(
            "PARAMETER SET:",
            parameter_name
        )

        print(
            "#" * 120
        )

        sub = summary_df[
            summary_df[
                "parameter_set"
            ]
            == parameter_name
        ]

        for _, row in (
            sub.iterrows()
        ):

            print(

                f"{row['scenario']:<20}"

                f" | Fill="
                f"{row['fill_rate_mean']:.4f}"
                f" ± "
                f"{row['fill_rate_ci95']:.4f}"

                f" | Profit="
                f"{row['profit_mean']:>12,.0f}"

                f" | Lost="
                f"{row['lost_units_mean']:>9,.1f}"

                f" | Holding="
                f"{row['holding_cost_mean']:>9,.0f}"

                f" | Supply="
                f"{row['actual_supply_mean']:>10,.0f}"

                f" | Shortfall="
                f"{row['capacity_shortfall_mean']:>10,.0f}"
            )


# ============================================================
# 11. Diagnostic
# ============================================================

def diagnostic_report(
    summary_df
):

    print(
        "\n"
        + "=" * 120
    )

    print(
        "CALIBRATION DIAGNOSTIC"
    )

    print(
        "=" * 120
    )

    for parameter_name in (
        PARAMETER_SETS.keys()
    ):

        sub = summary_df[
            summary_df[
                "parameter_set"
            ]
            == parameter_name
        ]

        normal = sub[
            sub["scenario"]
            == "Normal"
        ].iloc[0]

        mild = sub[
            sub["scenario"]
            == "Mild_Capacity"
        ].iloc[0]

        severe = sub[
            sub["scenario"]
            == "Severe_Capacity"
        ].iloc[0]

        compound = sub[
            sub["scenario"]
            == "Compound"
        ].iloc[0]

        print(
            "\n"
            + "-" * 120
        )

        print(
            parameter_name
        )

        print(
            "Normal Fill   =",
            round(
                normal[
                    "fill_rate_mean"
                ],
                4
            )
        )

        print(
            "Mild Fill     =",
            round(
                mild[
                    "fill_rate_mean"
                ],
                4
            )
        )

        print(
            "Severe Fill   =",
            round(
                severe[
                    "fill_rate_mean"
                ],
                4
            )
        )

        print(
            "Compound Fill =",
            round(
                compound[
                    "fill_rate_mean"
                ],
                4
            )
        )

        print(
            "Normal Profit =",
            f"{normal['profit_mean']:,.0f}"
        )

        print(
            "Severe Profit =",
            f"{severe['profit_mean']:,.0f}"
        )

        severe_drop = (
            normal[
                "fill_rate_mean"
            ]
            - severe[
                "fill_rate_mean"
            ]
        )

        compound_drop = (
            normal[
                "fill_rate_mean"
            ]
            - compound[
                "fill_rate_mean"
            ]
        )

        print(
            "Normal -> Severe Fill drop =",
            round(
                severe_drop,
                4
            )
        )

        print(
            "Normal -> Compound Fill drop =",
            round(
                compound_drop,
                4
            )
        )

        # --------------------------------------------------------
        # Calibration hints only
        # --------------------------------------------------------

        if (
            normal[
                "profit_mean"
            ]
            <= 0
        ):

            print(
                "[WARNING] "
                "Normal仍亏损。"
            )

        if (
            normal[
                "fill_rate_mean"
            ]
            < 0.98
        ):

            print(
                "[WARNING] "
                "Normal Fill Rate偏低。"
            )

        if (
            severe_drop < 0.01
        ):

            print(
                "[WARNING] "
                "Severe对服务水平影响仍然太小。"
            )

        if (
            severe[
                "fill_rate_mean"
            ]
            < 0.30
        ):

            print(
                "[WARNING] "
                "Severe可能设置得太困难。"
            )


# ============================================================
# 12. Main
# ============================================================

def main():

    print(
        "=" * 120
    )

    print(
        "SUPPLY CHAIN PARAMETER CALIBRATION"
    )

    print(
        "=" * 120
    )

    print(
        "Replications / setting =",
        N_REPLICATIONS
    )

    print(
        "Parameter sets         =",
        len(
            PARAMETER_SETS
        )
    )

    print(
        "Scenarios              =",
        len(
            SCENARIOS
        )
    )

    total_runs = (
        N_REPLICATIONS
        * len(
            PARAMETER_SETS
        )
        * len(
            SCENARIOS
        )
    )

    print(
        "Total runs             =",
        total_runs
    )

    print(
        "\nSupplier capacities:"
    )

    print(
        "Normal =",
        SUPPLIER_CAPACITY[0]
    )

    print(
        "Mild   =",
        SUPPLIER_CAPACITY[1]
    )

    print(
        "Severe =",
        SUPPLIER_CAPACITY[2]
    )

    results = []

    for (
        parameter_name,
        parameter_config
    ) in PARAMETER_SETS.items():

        print(
            "\n"
            + "=" * 120
        )

        print(
            "Testing:",
            parameter_name
        )

        print(
            "=" * 120
        )

        for (
            scenario_name,
            scenario_config
        ) in SCENARIOS.items():

            print(
                "\nScenario:",
                scenario_name
            )

            for replication in range(
                N_REPLICATIONS
            ):

                # ====================================================
                # CRN
                #
                # 所有策略/参数组同replication
                # 使用相同seed。
                # ====================================================

                seed = (
                    BASE_SEED
                    + replication
                )

                result = (
                    run_one_replication(
                        parameter_name=
                            parameter_name,

                        parameter_config=
                            parameter_config,

                        scenario_name=
                            scenario_name,

                        scenario_config=
                            scenario_config,

                        seed=
                            seed
                    )
                )

                results.append(
                    result
                )

            print(
                "completed:",
                N_REPLICATIONS
            )

    # ========================================================
    # Raw results
    # ========================================================

    raw_df = pd.DataFrame(
        results
    )

    raw_df.to_csv(
        "calibration_raw_results.csv",
        index=False
    )

    # ========================================================
    # Summary
    # ========================================================

    summary_df = build_summary(
        raw_df
    )

    summary_df.to_csv(
        "calibration_summary.csv",
        index=False
    )

    print_summary(
        summary_df
    )

    diagnostic_report(
        summary_df
    )

    print(
        "\n"
        + "=" * 120
    )

    print(
        "CALIBRATION COMPLETE"
    )

    print(
        "=" * 120
    )

    print(
        "\nSaved:"
    )

    print(
        "calibration_raw_results.csv"
    )

    print(
        "calibration_summary.csv"
    )


if __name__ == "__main__":
    main()
