import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SupplyChainEnv(gym.Env):
    """
    CW–FW instant-retail supply-chain environment aligned with
    `MDP Action Design Guide CW–FW Instant Retail Network`.

    Decision epoch: evening.

    State (94 dims):
        CW inventory                       5
        supplier->CW pipeline buckets      5 x 5 = 25
        supplier backlog                   5
        FW inventory                       10 x 5 = 50
        supplier disruption A_t one-hot    4
        transport disruption U_t one-hot   4
        normalized time                    1
        Total                              94

    Action (18 discrete combinations):
        a_t = (l_t, e_t, g_t)
        l_t in {0,1,2}: inventory protection level
        e_t in {0,1}:   upstream expedite
        g_t in {0,1,2}: CW allocation rule
            0 proportional
            1 lowest-DOS priority
            2 value/penalty priority

    Important distinction:
    - Parameters explicitly defined by the guide are implemented as written.
    - Numerical values not fixed by the guide (demand levels, price vector,
      base (s,S), initial inventories and absolute capacities) are marked below
      as calibration assumptions.
    """

    metadata = {"render_modes": []}

    def __init__(self, seed=None, max_steps=90):
        super().__init__()

        # ============================================================
        # 1. Network size -- source-defined
        # ============================================================
        self.num_fw = 10
        self.num_rdc = self.num_fw  # compatibility alias
        self.num_products = 5
        self.max_steps = int(max_steps)
        self.time_step = 0

        # ============================================================
        # 2. Demand lambda_{i,k}
        # Source: Poisson(lambda_{i,k}), where lambda is built from a
        # product base demand and an FW demand-intensity multiplier.
        # Exact numerical values are NOT specified by the guide.
        #
        # [CALIBRATION ASSUMPTION]
        # Multipliers average exactly 1.0, so total network demand scale
        # remains controlled while FWs are heterogeneous.
        # ============================================================
        self.product_base_lambda = np.array(
            [12.0, 14.0, 15.0, 16.0, 18.0], dtype=np.float64
        )
        self.fw_demand_multiplier = np.array(
            [0.70, 0.80, 0.90, 0.90, 1.00,
             1.00, 1.10, 1.10, 1.20, 1.30],
            dtype=np.float64,
        )
        self.lambda_matrix = (
            self.fw_demand_multiplier[:, None]
            * self.product_base_lambda[None, :]
        )
        self.network_lambda = self.lambda_matrix.sum(axis=0)

        # ============================================================
        # 3. Product economics
        # Source:
        # - different product prices p_k
        # - lost-sales penalty b_k
        # - FW unit holding cost = 3 * CW unit holding cost
        # - expedite unit cost may use c_exp,k = 0.05 p_k
        # Exact p_k, b_k and h_k values are NOT fixed by the guide.
        #
        # [CALIBRATION ASSUMPTION]
        # ============================================================
        self.price = np.array(
            [80.0, 90.0, 100.0, 110.0, 120.0], dtype=np.float64
        )
        self.lost_sales_penalty = 1.5 * self.price

        annual_cw_holding_rate = 0.20
        self.holding_cost_cw = (
            self.price * annual_cw_holding_rate / 365.0
        )
        self.holding_cost_fw = 3.0 * self.holding_cost_cw
        self.expedite_cost_per_unit = 0.05 * self.price

        # Compatibility aliases used by some older scripts.
        self.holding_cost_cdc = self.holding_cost_cw
        self.holding_cost_rdc = self.holding_cost_fw
        self.penalty_lost_sales = self.lost_sales_penalty

        # ============================================================
        # 4. Total capacity constraints
        # Source: C_C = 20 * C_F for a single FW.
        # Absolute C_F is not fixed.
        #
        # [CALIBRATION ASSUMPTION]
        # ============================================================
        self.fw_capacity_total = 1500.0
        self.cw_capacity_total = 20.0 * self.fw_capacity_total

        # Compatibility aliases. Note these are TOTAL warehouse capacities,
        # not per-SKU capacities.
        self.rdc_capacity = self.fw_capacity_total
        self.cdc_capacity = self.cw_capacity_total

        # ============================================================
        # 5. Baseline (s,S)
        # Source defines the structure, not the numerical thresholds.
        #
        # [CALIBRATION ASSUMPTION]
        # FW: reorder at ~2 days, order-up-to ~5 days.
        # CW: reorder at ~4 network-demand days, order-up-to ~10 days.
        # ============================================================
        self.s_fw_base = 2.0 * self.lambda_matrix
        self.S_fw_base = 5.0 * self.lambda_matrix

        self.s_cw_base = 4.0 * self.network_lambda
        self.S_cw_base = 10.0 * self.network_lambda

        # Older-name compatibility aliases.
        self.s_rdc_base = self.s_fw_base
        self.S_rdc_base = self.S_fw_base
        self.s_cdc_base = self.s_cw_base
        self.S_cdc_base = self.S_cw_base

        # ============================================================
        # 6. l_t multipliers -- source-defined (Guide Table 4)
        # CW s : [1.00, 1.10, 1.25]
        # CW S : [1.00, 1.10, 1.25]
        # FW s : [1.00, 1.05, 1.10]
        # FW S : NEVER raised by l_t
        # ============================================================
        self.eta_cw_s = np.array([1.00, 1.10, 1.25], dtype=np.float64)
        self.eta_cw_S = np.array([1.00, 1.10, 1.25], dtype=np.float64)
        self.eta_fw_s = np.array([1.00, 1.05, 1.10], dtype=np.float64)

        # ============================================================
        # 7. Supplier availability A_t -- source-defined
        # A0 normal, A1 mild, A2 severe, A3 complete shutdown.
        # alpha(A) = [1.00, 0.80, 0.40, 0.00]
        # ============================================================
        self.supplier_availability = np.array(
            [1.00, 0.80, 0.40, 0.00], dtype=np.float64
        )
        self.P_A = np.array(
            [
                [0.995, 0.004, 0.0008, 0.0002],
                [0.300, 0.600, 0.0800, 0.0200],
                [0.080, 0.250, 0.5500, 0.1200],
                [0.020, 0.080, 0.2500, 0.6500],
            ],
            dtype=np.float64,
        )

        # ============================================================
        # 8. Upstream transportation U_t -- source-defined
        # U0 normal, U1 +1 day, U2 +2 days, U3 cannot ship today.
        # ============================================================
        self.transport_extra_delay = np.array(
            [0, 1, 2, 0], dtype=np.int64
        )
        self.P_U = np.array(
            [
                [0.992, 0.006, 0.0015, 0.0005],
                [0.400, 0.500, 0.0800, 0.0200],
                [0.150, 0.300, 0.4500, 0.1000],
                [0.050, 0.150, 0.3000, 0.5000],
            ],
            dtype=np.float64,
        )

        # ============================================================
        # 9. Lead time -- source-defined
        # Base L0 ~ DiscreteUniform{1,2,3}.
        # U2 can add 2 days, therefore L_max = 5 before expediting.
        # ============================================================
        self.base_lead_time_min = 1
        self.base_lead_time_max = 3
        self.max_lead_time = 5

        # ============================================================
        # 10. Action / observation spaces
        # ============================================================
        self.action_space = spaces.Discrete(18)

        # State = CW(5) + pipeline(5x5) + backlog(5) + FW(10x5)
        #       + one-hot A(4) + one-hot U(4) + time(1) = 94
        self.obs_dim = (
            self.num_products
            + self.max_lead_time * self.num_products
            + self.num_products
            + self.num_fw * self.num_products
            + 4 + 4 + 1
        )
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        # ============================================================
        # 11. Fixed-scenario overrides for controlled experiments
        # ============================================================
        self.forced_A_t = None
        self.forced_U_t = None

        # RNGs are split so demand paths remain comparable across policies
        # under common random numbers even if actions trigger different
        # numbers of lead-time draws.
        self._seed_all(seed)
        self.reset(seed=seed)

    # ----------------------------------------------------------------
    # RNG
    # ----------------------------------------------------------------
    def _seed_all(self, seed=None):
        ss = np.random.SeedSequence(seed)
        demand_ss, disruption_ss, lead_ss = ss.spawn(3)
        self.demand_rng = np.random.default_rng(demand_ss)
        self.disruption_rng = np.random.default_rng(disruption_ss)
        self.leadtime_rng = np.random.default_rng(lead_ss)
        if seed is not None:
            self.action_space.seed(seed)

    # ----------------------------------------------------------------
    # Action encoding
    # ----------------------------------------------------------------
    def decode_action(self, action_idx):
        action_idx = int(action_idx)
        if not 0 <= action_idx < 18:
            raise ValueError(f"action_idx must be in [0, 17], got {action_idx}")
        l_t = action_idx // 6
        rem = action_idx % 6
        e_t = rem // 3
        g_t = rem % 3
        return l_t, e_t, g_t

    def action_masks(self):
        """U3 complete transport interruption makes e_t=1 infeasible."""
        mask = np.ones(18, dtype=bool)
        if self.U_t == 3:
            for a in range(18):
                _, e_t, _ = self.decode_action(a)
                if e_t == 1:
                    mask[a] = False
        return mask

    # ----------------------------------------------------------------
    # Scenario control
    # ----------------------------------------------------------------
    def set_disruption_override(self, A_t=None, U_t=None):
        if A_t is not None and A_t not in (0, 1, 2, 3):
            raise ValueError("A_t must be 0,1,2,3 or None")
        if U_t is not None and U_t not in (0, 1, 2, 3):
            raise ValueError("U_t must be 0,1,2,3 or None")
        self.forced_A_t = A_t
        self.forced_U_t = U_t
        if A_t is not None:
            self.A_t = int(A_t)
        if U_t is not None:
            self.U_t = int(U_t)

    def clear_disruption_override(self):
        self.forced_A_t = None
        self.forced_U_t = None

    # ----------------------------------------------------------------
    # Reset / daily clock
    # ----------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._seed_all(seed)

        self.time_step = 0
        self.A_t = 0 if self.forced_A_t is None else int(self.forced_A_t)
        self.U_t = 0 if self.forced_U_t is None else int(self.forced_U_t)

        # [CALIBRATION ASSUMPTION]
        # Initial CW ~6 days of network demand; FW ~4 days locally.
        self.cw_inventory = 6.0 * self.network_lambda
        self.fw_inventory = 4.0 * self.lambda_matrix

        # pipeline[r, k]: quantity arriving after r+1 mornings.
        self.cw_pipeline = np.zeros(
            (self.max_lead_time, self.num_products), dtype=np.float64
        )

        # Internally split backlog to avoid double-applying supplier
        # availability to transport-confirmed quantities.
        self.capacity_backlog = np.zeros(self.num_products, dtype=np.float64)
        self.transport_backlog = np.zeros(self.num_products, dtype=np.float64)

        # CW->FW shipments accepted in the evening and arriving next morning.
        self.fw_overnight_shipments = np.zeros(
            (self.num_fw, self.num_products), dtype=np.float64
        )

        # Compatibility aliases.
        self.cdc_inventory = self.cw_inventory
        self.rdc_inventory = self.fw_inventory
        self.cdc_pipeline = self.cw_pipeline

        # Prepare Day 1 up to the evening decision epoch:
        # morning receipts -> daytime demand -> evening observation.
        self._start_day()
        return self._get_obs(), {}

    def _start_day(self):
        """Morning arrivals followed by daytime customer demand."""
        # Supplier -> CW arrivals.
        arrivals_cw = self.cw_pipeline[0, :].copy()
        self.cw_pipeline = np.roll(self.cw_pipeline, -1, axis=0)
        self.cw_pipeline[-1, :] = 0.0
        self.cw_inventory += arrivals_cw

        # Previous evening's CW -> FW accepted shipments arrive now.
        arrivals_fw = self.fw_overnight_shipments.copy()
        self.fw_inventory += arrivals_fw
        self.fw_overnight_shipments[:] = 0.0

        # Capacity sanity checks.
        if self.cw_inventory.sum() > self.cw_capacity_total + 1e-8:
            raise RuntimeError(
                f"CW capacity exceeded: {self.cw_inventory.sum():.3f} > "
                f"{self.cw_capacity_total:.3f}"
            )
        if np.any(self.fw_inventory.sum(axis=1) > self.fw_capacity_total + 1e-8):
            raise RuntimeError("FW capacity exceeded after morning arrivals")

        # Daytime demand.
        demand = self.demand_rng.poisson(self.lambda_matrix).astype(np.float64)
        sales = np.minimum(self.fw_inventory, demand)
        lost = demand - sales
        self.fw_inventory -= sales

        self.current_demand_matrix = demand
        self.current_sales_matrix = sales
        self.current_lost_matrix = lost
        self.current_revenue = float(np.sum(sales * self.price[None, :]))
        self.current_lost_penalty = float(
            np.sum(lost * self.lost_sales_penalty[None, :])
        )
        self.current_cw_arrivals = arrivals_cw
        self.current_fw_arrivals = arrivals_fw

    # ----------------------------------------------------------------
    # Markov disruptions
    # ----------------------------------------------------------------
    def _update_markov_disruptions(self):
        if self.forced_A_t is None:
            self.A_t = int(
                self.disruption_rng.choice(4, p=self.P_A[self.A_t])
            )
        else:
            self.A_t = int(self.forced_A_t)

        if self.forced_U_t is None:
            self.U_t = int(
                self.disruption_rng.choice(4, p=self.P_U[self.U_t])
            )
        else:
            self.U_t = int(self.forced_U_t)

    # ----------------------------------------------------------------
    # State
    # ----------------------------------------------------------------
    @property
    def supplier_backlog(self):
        return self.capacity_backlog + self.transport_backlog

    def _positive_squash(self, x, scale):
        """Map nonnegative values smoothly into [0,1)."""
        x = np.asarray(x, dtype=np.float64)
        scale = np.asarray(scale, dtype=np.float64)
        return x / (x + scale + 1e-8)

    def _get_obs(self):
        cw_norm = np.clip(
            self.cw_inventory / self.cw_capacity_total, 0.0, 1.0
        )
        fw_norm = np.clip(
            self.fw_inventory / self.fw_capacity_total, 0.0, 1.0
        ).flatten()

        pipe_scale = np.broadcast_to(
            self.S_cw_base[None, :], self.cw_pipeline.shape
        )
        pipe_norm = self._positive_squash(
            self.cw_pipeline, pipe_scale
        ).flatten()
        backlog_norm = self._positive_squash(
            self.supplier_backlog, self.S_cw_base
        )

        a_onehot = np.zeros(4, dtype=np.float64)
        u_onehot = np.zeros(4, dtype=np.float64)
        a_onehot[self.A_t] = 1.0
        u_onehot[self.U_t] = 1.0

        time_norm = np.array(
            [min(self.time_step / max(self.max_steps, 1), 1.0)], dtype=np.float64
        )

        obs = np.concatenate(
            [
                cw_norm,
                pipe_norm,
                backlog_norm,
                fw_norm,
                a_onehot,
                u_onehot,
                time_norm,
            ]
        ).astype(np.float32)

        if obs.shape != (self.obs_dim,):
            raise RuntimeError(f"Observation shape error: {obs.shape}")
        return obs

    # ----------------------------------------------------------------
    # Action 1: thresholds -- source-defined multipliers
    # ----------------------------------------------------------------
    def _temporary_thresholds(self, l_t):
        s_cw = self.eta_cw_s[l_t] * self.s_cw_base
        S_cw = self.eta_cw_S[l_t] * self.S_cw_base

        # Source Eq. (45): FW only raises s; S remains unchanged.
        s_fw = np.minimum(
            self.eta_fw_s[l_t] * self.s_fw_base,
            self.S_fw_base - 1e-8,
        )
        S_fw = self.S_fw_base.copy()
        return s_cw, S_cw, s_fw, S_fw

    # ----------------------------------------------------------------
    # Action 3: CW allocation
    # ----------------------------------------------------------------
    def _allocate_cw_to_fw(self, fw_requests, g_t):
        """
        Allocate current CW inventory to FW replenishment requests.

        g=0 proportional
        g=1 lowest DOS = I_F / (lambda + eps)
        g=2 highest b_k * max(lambda_{i,k} - I_F, 0)
        """
        fw_requests = np.asarray(fw_requests, dtype=np.float64)
        allocated = np.zeros_like(fw_requests)
        eps = 1e-8

        for k in range(self.num_products):
            req = fw_requests[:, k]
            total_req = float(req.sum())
            avail = float(self.cw_inventory[k])

            if total_req <= eps or avail <= eps:
                continue

            if avail >= total_req:
                allocated[:, k] = req
                continue

            if g_t == 0:
                allocated[:, k] = req * (avail / total_req)

            elif g_t == 1:
                dos = self.fw_inventory[:, k] / (
                    self.lambda_matrix[:, k] + eps
                )
                # Low DOS first. Stable tie-break by FW index.
                rank = np.argsort(dos, kind="stable")
                remaining = avail
                for i in rank:
                    if req[i] <= 0:
                        continue
                    q = min(remaining, req[i])
                    allocated[i, k] = q
                    remaining -= q
                    if remaining <= eps:
                        break

            elif g_t == 2:
                priority = self.lost_sales_penalty[k] * np.maximum(
                    self.lambda_matrix[:, k] - self.fw_inventory[:, k],
                    0.0,
                )
                dos = self.fw_inventory[:, k] / (
                    self.lambda_matrix[:, k] + eps
                )
                # Primary: higher value/penalty priority.
                # Tie-break: lower DOS.
                rank = np.lexsort((dos, -priority))
                remaining = avail
                for i in rank:
                    if req[i] <= 0:
                        continue
                    q = min(remaining, req[i])
                    allocated[i, k] = q
                    remaining -= q
                    if remaining <= eps:
                        break
            else:
                raise ValueError(f"Invalid g_t={g_t}")

        # Enforce TOTAL FW capacity. If a proposed shipment would overflow
        # an FW, scale that FW's accepted shipment vector proportionally.
        for i in range(self.num_fw):
            space = max(
                0.0,
                self.fw_capacity_total - float(self.fw_inventory[i].sum()),
            )
            incoming = float(allocated[i].sum())
            if incoming > space + eps and incoming > 0:
                allocated[i] *= space / incoming

        # Final physical feasibility against CW inventory.
        shipped_by_product = allocated.sum(axis=0)
        if np.any(shipped_by_product > self.cw_inventory + 1e-6):
            raise RuntimeError("Allocation exceeds CW inventory")

        return allocated

    # ----------------------------------------------------------------
    # Supplier / pipeline helpers
    # ----------------------------------------------------------------
    def _sample_base_lead_time(self):
        return int(
            self.leadtime_rng.integers(
                self.base_lead_time_min,
                self.base_lead_time_max + 1,
            )
        )

    def _send_to_pipeline(self, quantity, k, U_t, expedite=False):
        quantity = float(quantity)
        if quantity <= 0:
            return None
        if U_t == 3:
            raise RuntimeError("Cannot send to pipeline under U3")

        L0 = self._sample_base_lead_time()
        lead_time = L0 + int(self.transport_extra_delay[U_t])
        if expedite:
            lead_time = max(1, lead_time - 1)

        if not 1 <= lead_time <= self.max_lead_time:
            raise RuntimeError(f"Lead time out of range: {lead_time}")

        self.cw_pipeline[lead_time - 1, k] += quantity
        return lead_time

    def _process_supplier_orders(self, new_orders, e_t, A_t, U_t):
        """
        Process old capacity backlog, old transport backlog and today's new
        CW supplier orders.

        Internal implementation separates capacity and transport backlog,
        while the MDP state exposes their sum as B_sup.
        """
        alpha = float(self.supplier_availability[A_t])

        stats = {
            "new_order_units": float(np.sum(new_orders)),
            "capacity_confirmed_units": 0.0,
            "capacity_unmet_units": 0.0,
            "transport_blocked_units": 0.0,
            "backlog_released_units": 0.0,
            "pipeline_shipped_units": 0.0,
            "expedite_cost": 0.0,
            "expedited_units": 0.0,
        }

        # Existing transport backlog has already passed supplier capacity.
        if U_t != 3:
            for k in range(self.num_products):
                q = float(self.transport_backlog[k])
                if q > 0:
                    self._send_to_pipeline(q, k, U_t, expedite=False)
                    stats["backlog_released_units"] += q
                    stats["pipeline_shipped_units"] += q
            self.transport_backlog[:] = 0.0

        # Existing capacity backlog is re-attempted under today's A_t.
        old_capacity = self.capacity_backlog.copy()
        released_old = alpha * old_capacity
        self.capacity_backlog = old_capacity - released_old

        for k in range(self.num_products):
            q = float(released_old[k])
            if q <= 0:
                continue
            if U_t == 3:
                self.transport_backlog[k] += q
                stats["transport_blocked_units"] += q
            else:
                self._send_to_pipeline(q, k, U_t, expedite=False)
                stats["backlog_released_units"] += q
                stats["pipeline_shipped_units"] += q

        # Today's new order Q^C.
        confirmed_new = alpha * new_orders
        unmet_new = new_orders - confirmed_new
        self.capacity_backlog += unmet_new
        stats["capacity_confirmed_units"] += float(confirmed_new.sum())
        stats["capacity_unmet_units"] += float(unmet_new.sum())

        # Guide Eq. (48)-(49): C_exp,t = e_t * sum_k c_exp,k Q^C_k,t,
        # with c_exp,k = 0.05 p_k. Therefore the cost is charged on
        # today's CW supplier order Q^C, while the lead-time reduction
        # applies only to quantities that are actually shipped today.
        if e_t == 1 and U_t != 3:
            stats["expedite_cost"] = float(
                np.sum(new_orders * self.expedite_cost_per_unit)
            )

        for k in range(self.num_products):
            q = float(confirmed_new[k])
            if q <= 0:
                continue

            if U_t == 3:
                self.transport_backlog[k] += q
                stats["transport_blocked_units"] += q
                continue

            expedite = bool(e_t == 1)
            self._send_to_pipeline(q, k, U_t, expedite=expedite)
            stats["pipeline_shipped_units"] += q

            if expedite:
                stats["expedited_units"] += q

        return stats

    # ----------------------------------------------------------------
    # Main MDP transition: evening action -> next evening state
    # ----------------------------------------------------------------
    def step(self, action):
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        current_A_t = int(self.A_t)
        current_U_t = int(self.U_t)
        l_t, requested_e_t, g_t = self.decode_action(action)

        # U3: expediting is physically infeasible.
        e_t = requested_e_t
        expedite_forced_off = False
        if current_U_t == 3 and e_t == 1:
            e_t = 0
            expedite_forced_off = True

        s_cw, S_cw, s_fw, S_fw = self._temporary_thresholds(l_t)

        # ============================================================
        # Evening step 1: FWs generate requests after today's demand.
        # ============================================================
        fw_requests = np.where(
            self.fw_inventory <= s_fw,
            np.maximum(S_fw - self.fw_inventory, 0.0),
            0.0,
        )

        # ============================================================
        # Evening step 2: CW allocates inventory using g_t.
        # Accepted shipments arrive at FWs NEXT morning.
        # ============================================================
        allocated = self._allocate_cw_to_fw(fw_requests, g_t)
        shipped_by_product = allocated.sum(axis=0)
        self.cw_inventory -= shipped_by_product
        self.cw_inventory = np.maximum(self.cw_inventory, 0.0)
        self.fw_overnight_shipments = allocated.copy()

        # ============================================================
        # Evening step 3: CW supplier orders using inventory position
        # IP^C = on-hand CW + supplier pipeline (Guide Eq. 5).
        # ============================================================
        # Guide requirement:
        # CW inventory position must include ALL quantities already committed
        # but not yet received:
        #   on-hand + supplier capacity backlog + transport backlog + pipeline.
        #
        # Omitting backlog causes repeated ordering during disruptions and can
        # create an artificial post-disruption arrival surge that exceeds CW
        # capacity.
        inventory_position = (
            self.cw_inventory
            + self.capacity_backlog
            + self.transport_backlog
            + self.cw_pipeline.sum(axis=0)
        )

        new_orders = np.where(
            inventory_position <= s_cw,
            np.maximum(S_cw - inventory_position, 0.0),
            0.0,
        )

        # ============================================================
        # Evening step 4: A_t confirms supply; U_t/e_t determine shipping.
        # Unmet / unshippable quantities enter backlog rather than vanish.
        # ============================================================
        supplier_stats = self._process_supplier_orders(
            new_orders=new_orders,
            e_t=e_t,
            A_t=current_A_t,
            U_t=current_U_t,
        )

        # ============================================================
        # Evening step 5: reward for the current day.
        # ============================================================
        holding_cw = float(np.sum(self.cw_inventory * self.holding_cost_cw))
        holding_fw = float(
            np.sum(self.fw_inventory * self.holding_cost_fw[None, :])
        )
        holding_total = holding_cw + holding_fw
        expedite_cost = float(supplier_stats["expedite_cost"])

        profit = (
            self.current_revenue
            - self.current_lost_penalty
            - holding_total
            - expedite_cost
        )
        reward = profit / 10000.0

        total_demand = float(self.current_demand_matrix.sum())
        total_sales = float(self.current_sales_matrix.sum())
        total_lost = float(self.current_lost_matrix.sum())
        fill_rate = total_sales / total_demand if total_demand > 0 else 1.0

        # Save current-day information BEFORE advancing the day.
        info = {
            "profit": float(profit),
            "revenue": float(self.current_revenue),
            "lost_sales_penalty": float(self.current_lost_penalty),
            "holding_cost": float(holding_total),
            "holding_cost_cw": float(holding_cw),
            "holding_cost_fw": float(holding_fw),
            "expedite_cost": float(expedite_cost),
            "demand_units": total_demand,
            "sales_units": total_sales,
            "lost_units": total_lost,
            "fill_rate": float(fill_rate),
            "A_t": current_A_t,
            "U_t": current_U_t,
            "action": action,
            "l_t": int(l_t),
            "requested_e_t": int(requested_e_t),
            "executed_e_t": int(e_t),
            "g_t": int(g_t),
            "expedite_forced_off": bool(expedite_forced_off),
            "fw_request_units": float(fw_requests.sum()),
            "fw_shipment_units": float(allocated.sum()),
            "new_supplier_order_units": float(new_orders.sum()),
            "supplier_backlog_units": float(self.supplier_backlog.sum()),
            "capacity_backlog_units": float(self.capacity_backlog.sum()),
            "transport_backlog_units": float(self.transport_backlog.sum()),
            "pipeline_units": float(self.cw_pipeline.sum()),
            "cw_inventory_total": float(self.cw_inventory.sum()),
            "fw_inventory_total": float(self.fw_inventory.sum()),
            "demand_matrix": self.current_demand_matrix.copy(),
            "sales_matrix": self.current_sales_matrix.copy(),
            "lost_matrix": self.current_lost_matrix.copy(),
            "fw_requests": fw_requests.copy(),
            "allocated_shipments": allocated.copy(),
            "cw_inventory": self.cw_inventory.copy(),
            "fw_inventory": self.fw_inventory.copy(),
            "cw_pipeline": self.cw_pipeline.copy(),
            "supplier_backlog": self.supplier_backlog.copy(),
            "s_cw": s_cw.copy(),
            "S_cw": S_cw.copy(),
            "s_fw": s_fw.copy(),
            "S_fw": S_fw.copy(),
            **supplier_stats,
        }

        # ============================================================
        # Evening step 6: next disruption state.
        # ============================================================
        self._update_markov_disruptions()
        info["next_A_t"] = int(self.A_t)
        info["next_U_t"] = int(self.U_t)

        # One simulated day has completed.
        self.time_step += 1
        terminated = self.time_step >= self.max_steps
        truncated = False

        # Prepare next day's morning and daytime BEFORE returning the next
        # evening state. Do not generate an unused day after termination.
        if not terminated:
            self._start_day()

        # Refresh aliases in case arrays were rolled/rebound.
        self.cdc_inventory = self.cw_inventory
        self.rdc_inventory = self.fw_inventory
        self.cdc_pipeline = self.cw_pipeline

        return self._get_obs(), float(reward), bool(terminated), truncated, info