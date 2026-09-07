import numpy as np

from env import SupplyChainEnv


class ZeroDemandRNG:
    """
    Pipeline测试时让终端需求为0。
    """

    def poisson(
        self,
        lam=None,
        size=None
    ):

        return np.zeros(
            size,
            dtype=int
        )


def prepare_env(U_t):

    env = SupplyChainEnv(
        seed=42
    )

    # ============================================================
    # Pipeline测试只测试运输时间，
    # 不测试supplier capacity。
    # ============================================================

    env.supplier_capacity_per_product[:] = (
        1e9
    )

    # ============================================================
    # 禁止Markov变化
    # ============================================================

    env._update_markov_disruptions = (
        lambda: None
    )

    env.A_t = 0
    env.U_t = U_t

    # ============================================================
    # 只让产品0触发CDC补货
    # ============================================================

    env.cdc_inventory[:] = (
        12000.0
    )

    env.cdc_inventory[0] = (
        4000.0
    )

    env.cdc_pipeline[:] = 0.0

    # ============================================================
    # RDC库存够高，避免CDC发货
    # ============================================================

    env.rdc_inventory[:] = (
        1500.0
    )

    # ============================================================
    # 无需求
    # ============================================================

    env.rng = ZeroDemandRNG()

    return env


def print_state(
    env,
    label
):

    print(
        f"\n{label}"
    )

    print(
        "CDC product 0 inventory:",
        env.cdc_inventory[0]
    )

    print(
        "Pipeline product 0:",
        env.cdc_pipeline[:, 0]
    )


def run_case(
    title,
    U_t,
    action,
    steps
):

    print(
        "\n"
        + "=" * 70
    )

    print(title)

    print(
        "=" * 70
    )

    env = prepare_env(
        U_t
    )

    l, e, g = (
        env.decode_action(
            action
        )
    )

    print(
        f"Action = {action}, "
        f"decoded = "
        f"(l={l}, e={e}, g={g})"
    )

    print(
        "U_t =",
        U_t
    )

    print_state(
        env,
        "BEFORE STEP"
    )

    for day in range(
        steps
    ):

        (
            obs,
            reward,
            done,
            truncated,
            info
        ) = env.step(action)

        print_state(
            env,
            f"AFTER STEP {day + 1}"
        )


def main():

    # Normal, no expedite
    run_case(
        title=(
            "CASE 1: Normal transport, "
            "no expedite (expected LT=2)"
        ),
        U_t=0,
        action=0,
        steps=4
    )

    # Normal + expedite
    run_case(
        title=(
            "CASE 2: Normal transport + "
            "expedite (expected LT=1)"
        ),
        U_t=0,
        action=3,
        steps=3
    )

    # Delay
    run_case(
        title=(
            "CASE 3: Transport delay, "
            "no expedite (expected LT=3)"
        ),
        U_t=1,
        action=0,
        steps=5
    )

    # Delay + expedite
    run_case(
        title=(
            "CASE 4: Transport delay + "
            "expedite (expected LT=2)"
        ),
        U_t=1,
        action=3,
        steps=4
    )

    # Complete disruption
    run_case(
        title=(
            "CASE 5: Complete disruption "
            "(expected NO replenishment)"
        ),
        U_t=2,
        action=3,
        steps=3
    )


if __name__ == "__main__":
    main()
