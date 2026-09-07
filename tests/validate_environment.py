import numpy as np
from env import SupplyChainEnv


def main():
    print('=' * 80)
    print('GUIDE-ALIGNED ENVIRONMENT VALIDATION')
    print('=' * 80)

    # ------------------------------------------------------------------
    # 1. Observation / action spaces
    # ------------------------------------------------------------------
    env = SupplyChainEnv(seed=42, max_steps=20)
    obs, _ = env.reset(seed=42)

    print('\n[TEST 1] Observation and action spaces')
    print('observation_space =', env.observation_space)
    print('action_space      =', env.action_space)
    print('obs shape         =', obs.shape)
    print('obs min/max       =', float(obs.min()), float(obs.max()))

    assert obs.shape == (94,)
    assert np.all(np.isfinite(obs))
    assert obs.min() >= -1e-8
    assert obs.max() <= 1.0 + 1e-8
    assert env.action_space.n == 18
    print('PASS')

    # ------------------------------------------------------------------
    # 2. 18 unique actions + U3 mask
    # ------------------------------------------------------------------
    print('\n[TEST 2] Action decoding and U3 mask')
    decoded = [env.decode_action(a) for a in range(18)]
    for a, x in enumerate(decoded):
        print(f'action {a:2d} -> l={x[0]}, e={x[1]}, g={x[2]}')
    assert len(set(decoded)) == 18

    env.U_t = 3
    mask = env.action_masks()
    valid = np.where(mask)[0]
    invalid = np.where(~mask)[0]
    print('U3 valid   =', valid)
    print('U3 invalid =', invalid)
    assert len(valid) == 9
    assert len(invalid) == 9
    for a in invalid:
        assert env.decode_action(a)[1] == 1
    print('PASS')

    # ------------------------------------------------------------------
    # 3. l_t must match Guide Table 4 exactly
    # ------------------------------------------------------------------
    print('\n[TEST 3] Inventory-protection multipliers')
    expected_cw = [1.00, 1.10, 1.25]
    expected_fw = [1.00, 1.05, 1.10]

    for l in range(3):
        s_cw, S_cw, s_fw, S_fw = env._temporary_thresholds(l)
        assert np.allclose(s_cw, expected_cw[l] * env.s_cw_base)
        assert np.allclose(S_cw, expected_cw[l] * env.S_cw_base)
        assert np.allclose(
            s_fw,
            np.minimum(expected_fw[l] * env.s_fw_base, env.S_fw_base - 1e-8),
        )
        # Critical source requirement: FW S does NOT change with l_t.
        assert np.allclose(S_fw, env.S_fw_base)
        print(
            f'l={l}: CW factor={expected_cw[l]:.2f}, '
            f'FW-s factor={expected_fw[l]:.2f}, FW-S unchanged'
        )
    print('PASS')

    # ------------------------------------------------------------------
    # 4. g=1 and g=2 must be genuinely different rules
    # ------------------------------------------------------------------
    print('\n[TEST 4] Lowest-DOS vs Value/Penalty allocation')
    test_env = SupplyChainEnv(seed=1, max_steps=5)

    k = 0
    test_env.cw_inventory[:] = 0.0
    test_env.cw_inventory[k] = 50.0
    test_env.fw_inventory[:] = 0.0

    # Construct two FWs deliberately so the rankings disagree:
    # FW0: lambda=10, I=1  -> DOS=0.10, priority=9*b
    # FW1: lambda=20, I=5  -> DOS=0.25, priority=15*b
    # Therefore g=1 prefers FW0, while g=2 prefers FW1.
    test_env.lambda_matrix[0, k] = 10.0
    test_env.lambda_matrix[1, k] = 20.0
    test_env.fw_inventory[0, k] = 1.0
    test_env.fw_inventory[1, k] = 5.0

    requests = np.zeros((test_env.num_fw, test_env.num_products))
    requests[0, k] = 100.0
    requests[1, k] = 100.0

    alloc_g1 = test_env._allocate_cw_to_fw(requests, g_t=1)
    alloc_g2 = test_env._allocate_cw_to_fw(requests, g_t=2)

    print('g=1 allocation FW0/FW1 =', alloc_g1[0, k], alloc_g1[1, k])
    print('g=2 allocation FW0/FW1 =', alloc_g2[0, k], alloc_g2[1, k])

    assert alloc_g1[0, k] > alloc_g1[1, k]
    assert alloc_g2[1, k] > alloc_g2[0, k]
    assert not np.allclose(alloc_g1, alloc_g2)
    print('PASS')

    # ------------------------------------------------------------------
    # 5. Markov matrices
    # ------------------------------------------------------------------
    print('\n[TEST 5] 4-state Markov matrices')
    print('P_A row sums =', env.P_A.sum(axis=1))
    print('P_U row sums =', env.P_U.sum(axis=1))
    assert env.P_A.shape == (4, 4)
    assert env.P_U.shape == (4, 4)
    assert np.allclose(env.P_A.sum(axis=1), 1.0)
    assert np.allclose(env.P_U.sum(axis=1), 1.0)
    assert np.allclose(env.supplier_availability, [1.0, 0.8, 0.4, 0.0])
    print('PASS')

    # ------------------------------------------------------------------
    # 6. Base lead time must be DiscreteUniform{1,2,3}
    # ------------------------------------------------------------------
    print('\n[TEST 6] Base lead time support')
    samples = np.array([env._sample_base_lead_time() for _ in range(1000)])
    print('observed values =', np.unique(samples))
    assert set(np.unique(samples)).issubset({1, 2, 3})
    assert set(np.unique(samples)) == {1, 2, 3}
    print('PASS')

    # ------------------------------------------------------------------
    # 7. U3 must create backlog, not delete orders
    # ------------------------------------------------------------------
    print('\n[TEST 7] U3 transport backlog')
    block_env = SupplyChainEnv(seed=7, max_steps=5)
    block_env.set_disruption_override(A_t=0, U_t=3)
    block_env.reset(seed=7)

    # Force a CW replenishment need.
    block_env.cw_inventory[:] = 0.0
    block_env.cw_pipeline[:] = 0.0
    block_env.capacity_backlog[:] = 0.0
    block_env.transport_backlog[:] = 0.0

    _, _, _, _, info = block_env.step(0)
    print('new supplier order =', info['new_supplier_order_units'])
    print('transport backlog  =', info['transport_backlog_units'])
    print('pipeline units     =', info['pipeline_units'])

    assert info['new_supplier_order_units'] > 0
    assert info['transport_backlog_units'] > 0
    assert info['pipeline_units'] == 0
    print('PASS')

    # ------------------------------------------------------------------
    # 8. Accounting + long random rollout
    # ------------------------------------------------------------------
    print('\n[TEST 8] Random rollout / accounting / capacities')
    roll_env = SupplyChainEnv(seed=123, max_steps=30)
    obs, _ = roll_env.reset(seed=123)

    for t in range(300):
        action = roll_env.action_space.sample()
        obs, reward, done, truncated, info = roll_env.step(action)

        assert obs.shape == (94,)
        assert np.all(np.isfinite(obs))
        assert obs.min() >= -1e-8
        assert obs.max() <= 1.0 + 1e-8

        assert np.all(roll_env.cw_inventory >= -1e-8)
        assert np.all(roll_env.fw_inventory >= -1e-8)
        assert np.all(roll_env.cw_pipeline >= -1e-8)
        assert np.all(roll_env.supplier_backlog >= -1e-8)

        assert roll_env.cw_inventory.sum() <= roll_env.cw_capacity_total + 1e-6
        assert np.all(
            roll_env.fw_inventory.sum(axis=1)
            <= roll_env.fw_capacity_total + 1e-6
        )

        assert np.isclose(
            info['sales_units'] + info['lost_units'],
            info['demand_units'],
            atol=1e-6,
        )
        expected_profit = (
            info['revenue']
            - info['lost_sales_penalty']
            - info['holding_cost']
            - info['expedite_cost']
        )
        assert np.isclose(info['profit'], expected_profit, atol=1e-6)
        assert np.isclose(reward, info['profit'] / 10000.0, atol=1e-8)

        if done or truncated:
            obs, _ = roll_env.reset(seed=123 + t + 1)

    print('PASS: 300 random steps')

    print('\n' + '=' * 80)
    print('ALL GUIDE-ALIGNED TESTS PASSED')
    print('=' * 80)


if __name__ == '__main__':
    main()
    
