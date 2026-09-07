import numpy as np

from env import SupplyChainEnv


def configure_moderate_environment(
    env
):
    """
    使用Moderate库存参数进行supplier capacity单元测试。

    【实验校准参数，不是原文件规定】
    """

    # ============================================================
    # Economics
    # ============================================================

    env.price = 100.0

    env.holding_cost_cdc = (
        100.0
        * 0.20
        / 365.0
    )

    env.holding_cost_rdc = (
        100.0
        * 0.30
        / 365.0
    )

    env.penalty_lost_sales = (
        150.0
    )

    # ============================================================
    # Capacity
    # ============================================================

    env.cdc_capacity = (
        3000.0
    )

    env.rdc_capacity = (
        300.0
    )

    # ============================================================
    # Moderate (s,S)
    # ============================================================

    env.s_cdc_base = (
        600.0
    )

    env.S_cdc_base = (
        1500.0
    )

    env.s_rdc_base = (
        30.0
    )

    env.S_rdc_base = (
        75.0
    )


def run_case(
    name,
    A_t,
    days=30,
    seed=42
):

    print(
        "\n"
        + "=" * 100
    )

    print(name)

    print(
        "=" * 100
    )

    env = SupplyChainEnv(
        seed=seed
    )

    configure_moderate_environment(
        env
    )

    # ============================================================
    # 固定supplier状态
    # Transport正常
    # ============================================================

    env.set_disruption_override(
        A_t=A_t,
        U_t=0
    )

    env.reset(
        seed=seed
    )

    # ============================================================
    # Moderate初始库存
    # ============================================================

    env.cdc_inventory[:] = (
        900.0
    )

    env.rdc_inventory[:] = (
        60.0
    )

    env.cdc_pipeline[:] = (
        0.0
    )

    total_demand = 0.0
    total_sales = 0.0
    total_lost = 0.0

    total_requested = 0.0
    total_fulfilled = 0.0
    total_unfulfilled = 0.0

    print(
        "Expected demand / product / day =",
        env.expected_demand_per_product
    )

    print(
        "Supplier capacity / product / day =",
        env.supplier_capacity_per_product[
            A_t
        ]
    )

    for day in range(
        1,
        days + 1
    ):

        (
            obs,
            reward,
            done,
            truncated,
            info
        ) = env.step(0)

        total_demand += (
            info[
                "demand_units"
            ]
        )

        total_sales += (
            info[
                "sales_units"
            ]
        )

        total_lost += (
            info[
                "lost_units"
            ]
        )

        total_requested += (
            info[
                "requested_upstream_units"
            ]
        )

        total_fulfilled += (
            info[
                "supplier_fulfilled_units"
            ]
        )

        total_unfulfilled += (
            info[
                "supplier_unfulfilled_units"
            ]
        )

        if (
            day <= 10
            or day % 5 == 0
        ):

            print(
                f"Day {day:2d}"
                f" | CDC="
                f"{np.sum(info['cdc_inventory']):8.1f}"
                f" | RDC="
                f"{np.sum(info['rdc_inventory']):8.1f}"
                f" | Pipeline="
                f"{np.sum(info['cdc_pipeline']):7.1f}"
                f" | Demand="
                f"{info['demand_units']:6.0f}"
                f" | Lost="
                f"{info['lost_units']:6.0f}"
                f" | Req="
                f"{info['requested_upstream_units']:8.1f}"
                f" | Supplied="
                f"{info['supplier_fulfilled_units']:6.1f}"
            )

    fill_rate = (
        total_sales
        / total_demand
        if total_demand > 0
        else 1.0
    )

    print(
        "\nSUMMARY"
    )

    print(
        "Total demand       =",
        round(
            total_demand,
            2
        )
    )

    print(
        "Total sales        =",
        round(
            total_sales,
            2
        )
    )

    print(
        "Total lost         =",
        round(
            total_lost,
            2
        )
    )

    print(
        "Fill rate          =",
        round(
            fill_rate,
            4
        )
    )

    print(
        "Requested supply   =",
        round(
            total_requested,
            2
        )
    )

    print(
        "Actual supply      =",
        round(
            total_fulfilled,
            2
        )
    )

    print(
        "Capacity shortfall =",
        round(
            total_unfulfilled,
            2
        )
    )

    return fill_rate


def main():

    normal_fill = run_case(
        name=(
            "CASE 1: "
            "Normal supplier capacity"
        ),
        A_t=0
    )

    mild_fill = run_case(
        name=(
            "CASE 2: "
            "Mild capacity shortage"
        ),
        A_t=1
    )

    severe_fill = run_case(
        name=(
            "CASE 3: "
            "Severe capacity shortage"
        ),
        A_t=2
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "SUPPLIER CAPACITY COMPARISON"
    )

    print(
        "=" * 100
    )

    print(
        "Normal fill =",
        round(
            normal_fill,
            4
        )
    )

    print(
        "Mild fill   =",
        round(
            mild_fill,
            4
        )
    )

    print(
        "Severe fill =",
        round(
            severe_fill,
            4
        )
    )

    # 不要求随机样本严格每级都下降，
    # 但Severe应该明显比Normal差。

    if (
        severe_fill
        >= normal_fill - 0.01
    ):

        print(
            "\n[WARNING] "
            "Severe仍未明显降低Fill Rate。"
        )

    else:

        print(
            "\nPASS: "
            "Severe disruption produces "
            "meaningful service degradation."
        )


if __name__ == "__main__":
    main()
