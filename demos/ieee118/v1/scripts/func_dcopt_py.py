"""
Pure Python port of func_dcopt.m, add_branchCapacity.m, and load2disp.

No MATLAB/Octave dependency — uses scipy.optimize.linprog for DC-OPF solving
with the dual-simplex method (matching the patched MATPOWER qps_ot.m solver).
Compatible with MATPOWER .m case files via the included parser.

Usage:
    from func_dcopt_py import load_case, add_branch_capacity, func_dcopt

    ppc = load_case('case14')  # or load_case('/path/to/case_ACTIVSg2000.m')
    ppc = add_branch_capacity(ppc, alpha=2.0)

    system_state = np.zeros(nb + nl)
    system_state[0] = 1  # fail bus 1
    blackout_size, flag = func_dcopt(system_state, ppc)
"""

import numpy as np
import re
from copy import deepcopy
from pathlib import Path
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

from pypower.rundcpf import rundcpf
from pypower.ppoption import ppoption
from pypower.idx_bus import BUS_I, BUS_TYPE, PD, QD, GS, BS, VM, BASE_KV, REF
from pypower.idx_gen import GEN_BUS, PG, QG, QMAX, QMIN, VG, MBASE, GEN_STATUS, PMAX, PMIN, APF
from pypower.idx_brch import F_BUS, T_BUS, BR_X, TAP, RATE_A, BR_STATUS, PF as BR_PF, PT as BR_PT
from pypower.idx_cost import MODEL, STARTUP, SHUTDOWN, NCOST, COST, POLYNOMIAL


# ---------------------------------------------------------------------------
# Case loading
# ---------------------------------------------------------------------------

def _parse_matpower_matrix(text, field_name):
    """Extract a numeric matrix from MATPOWER .m file text."""
    pattern = rf'mpc\.{field_name}\s*=\s*\[(.*?)\];'
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None
    block = match.group(1)
    rows = []
    for line in block.strip().split('\n'):
        line = line.split('%')[0].strip().rstrip(';')
        if not line:
            continue
        vals = line.split()
        if vals:
            rows.append([float(v) for v in vals])
    return np.array(rows, dtype=float) if rows else None


def load_case(case_name_or_path):
    """
    Load a MATPOWER case, returning a PYPOWER-compatible dict.

    Args:
        case_name_or_path: Either a built-in PYPOWER case name (e.g., 'case14')
            or a path to a MATPOWER .m file.

    Returns:
        ppc dict with keys: version, baseMVA, bus, gen, branch, gencost
    """
    path = Path(case_name_or_path)
    if path.exists() and path.suffix == '.m':
        # Parse MATPOWER .m file
        with open(path, 'r') as f:
            text = f.read()
        bus = _parse_matpower_matrix(text, 'bus')
        gen = _parse_matpower_matrix(text, 'gen')
        branch = _parse_matpower_matrix(text, 'branch')
        gencost = _parse_matpower_matrix(text, 'gencost')
        m = re.search(r'mpc\.baseMVA\s*=\s*(\d+\.?\d*)', text)
        baseMVA = float(m.group(1)) if m else 100.0

        ppc = {
            'version': '2',
            'baseMVA': baseMVA,
            'bus': bus,
            'gen': gen,
            'branch': branch,
        }
        if gencost is not None:
            ppc['gencost'] = gencost
        return ppc
    else:
        # Try built-in PYPOWER case
        try:
            mod = __import__(f'pypower.{case_name_or_path}',
                             fromlist=[case_name_or_path])
            return getattr(mod, case_name_or_path)()
        except (ImportError, AttributeError):
            raise FileNotFoundError(
                f"Case '{case_name_or_path}' not found as file or PYPOWER built-in"
            )


# ---------------------------------------------------------------------------
# Branch capacity (port of add_branchCapacity.m)
# ---------------------------------------------------------------------------

def add_branch_capacity(ppc, alpha=2.0):
    """
    Set branch thermal limits based on DC power flow results.

    If all RATE_A values are zero, runs a DC power flow and sets
    RATE_A = alpha * 0.5 * |PF - PT| for each branch.

    Args:
        ppc: PYPOWER case dict
        alpha: capacity scaling factor (default 2.0)

    Returns:
        ppc with updated branch RATE_A values
    """
    ppc = deepcopy(ppc)
    if np.all(ppc['branch'][:, RATE_A] == 0):
        ppopt = ppoption(VERBOSE=0, OUT_ALL=0)
        results = rundcpf(ppc, ppopt)
        res_branch = results[0]['branch']
        ppc['branch'][:, RATE_A] = (
            alpha * 0.5 * np.abs(res_branch[:, BR_PF] - res_branch[:, BR_PT])
        )
    # PYPOWER treats RATE_A=0 as zero capacity (unlike MATPOWER where 0=unlimited)
    # Replace any remaining zeros with a large value
    zero_rate = ppc['branch'][:, RATE_A] == 0
    if np.any(zero_rate):
        ppc['branch'][zero_rate, RATE_A] = 9999.0
    return ppc


# ---------------------------------------------------------------------------
# load2disp (port of MATPOWER's load2disp.m)
# ---------------------------------------------------------------------------

def load2disp(ppc):
    """
    Convert fixed loads to negative dispatchable generators.

    For each bus with PD > 0, creates a dispatchable "generator" with
    negative Pg representing the load, allowing the OPF to shed load
    at a high cost (value of lost load).

    Returns:
        ppc with modified bus, gen, gencost tables
    """
    ppc = deepcopy(ppc)
    bus = ppc['bus']
    gen = ppc['gen']
    gencost = ppc.get('gencost')

    # Ensure gen has at least 21 columns (PYPOWER standard)
    if gen.shape[1] < 21:
        gen = np.hstack([gen, np.zeros((gen.shape[0], 21 - gen.shape[1]))])

    # Find load buses
    load_idx = np.where(bus[:, PD] != 0)[0]
    nld = len(load_idx)

    if nld == 0:
        ppc['gen'] = gen
        return ppc

    ng = gen.shape[0]

    # Create dispatchable load generators (negative generation)
    new_gen = np.zeros((nld, gen.shape[1]))
    new_gen[:, GEN_BUS] = bus[load_idx, BUS_I]
    new_gen[:, PG] = -bus[load_idx, PD]
    new_gen[:, QG] = -bus[load_idx, QD]
    new_gen[:, QMAX] = 0
    new_gen[:, QMIN] = -bus[load_idx, QD]
    new_gen[:, VG] = bus[load_idx, VM]
    new_gen[:, MBASE] = bus[load_idx, BASE_KV] if np.any(bus[load_idx, BASE_KV]) else 100.0
    new_gen[:, GEN_STATUS] = 1
    new_gen[:, PMAX] = 0
    new_gen[:, PMIN] = -bus[load_idx, PD]

    # Append to gen table
    ppc['gen'] = np.vstack([gen, new_gen])

    # Zero out original loads
    ppc['bus'][load_idx, PD] = 0
    ppc['bus'][load_idx, QD] = 0

    # Create gencost for dispatchable loads (high cost = value of lost load)
    # Cost = VOLL * Pg. Since Pg is negative for loads, the OPF is incentivized
    # to serve load (large negative cost) rather than shed it (cost = 0).
    voll = 5000.0
    new_gencost = np.zeros((nld, 7))
    new_gencost[:, MODEL] = POLYNOMIAL
    new_gencost[:, STARTUP] = 0
    new_gencost[:, SHUTDOWN] = 0
    new_gencost[:, NCOST] = 2  # linear cost: c1*Pg + c0
    new_gencost[:, COST] = voll  # linear coefficient
    new_gencost[:, COST + 1] = 0  # constant term

    if gencost is not None:
        # Pad gencost to same number of columns if needed
        nc = max(gencost.shape[1], new_gencost.shape[1])
        if gencost.shape[1] < nc:
            gencost = np.hstack([gencost, np.zeros((gencost.shape[0], nc - gencost.shape[1]))])
        if new_gencost.shape[1] < nc:
            new_gencost = np.hstack([new_gencost, np.zeros((nld, nc - new_gencost.shape[1]))])
        ppc['gencost'] = np.vstack([gencost, new_gencost])
    else:
        ppc['gencost'] = new_gencost

    return ppc


# ---------------------------------------------------------------------------
# DC-OPF solver using scipy.optimize.linprog (dual-simplex)
# ---------------------------------------------------------------------------

def _solve_dcopf(mpc):
    """
    Solve DC optimal power flow using scipy.optimize.linprog.

    Formulation (all in p.u. on baseMVA):
        Decision variables: x = [Pg (ng); Va (nb)]

        Minimize: c_pg' * Pg

        Subject to:
            - Power balance at each bus: Cg * Pg - Bbus * Va = 0
            - Branch flow limits: -RATE_A <= Bf * Va <= RATE_A
            - Reference bus angle: Va_ref = 0
            - Generator limits: Pmin <= Pg <= Pmax
            - Va unbounded

    This formulation handles disconnected islands naturally since each
    island has its own independent power balance constraints.

    Returns:
        dict with 'success', 'gen' (updated generator table with Pg)
    """
    bus = mpc['bus']
    gen = mpc['gen']
    branch = mpc['branch']
    gencost = mpc['gencost']
    baseMVA = mpc.get('baseMVA', 100.0)

    nb = bus.shape[0]
    ng_total = gen.shape[0]
    nl = branch.shape[0]

    if nb == 0 or ng_total == 0 or nl == 0:
        return {'success': False}

    # Bus numbering: external -> internal (0-indexed)
    bus_ids = bus[:, BUS_I].astype(int)
    bus_map = {int(bid): i for i, bid in enumerate(bus_ids)}

    # Branch susceptance
    x_br = branch[:, BR_X].copy()
    x_br[x_br == 0] = 1e-6
    tap = branch[:, TAP].copy()
    tap[tap == 0] = 1.0
    b_branch = 1.0 / (x_br * tap)

    f_bus = np.array([bus_map[int(b)] for b in branch[:, F_BUS]])
    t_bus = np.array([bus_map[int(b)] for b in branch[:, T_BUS]])

    # Bf: branch-bus susceptance matrix (nl x nb)
    Bf = np.zeros((nl, nb))
    for l in range(nl):
        Bf[l, f_bus[l]] = b_branch[l]
        Bf[l, t_bus[l]] = -b_branch[l]

    # Bbus: bus susceptance matrix (nb x nb)
    Bbus = np.zeros((nb, nb))
    for l in range(nl):
        f, t = f_bus[l], t_bus[l]
        b = b_branch[l]
        Bbus[f, f] += b
        Bbus[f, t] -= b
        Bbus[t, f] -= b
        Bbus[t, t] += b

    # Cg: generator-bus connection matrix (nb x ng_total)
    Cg = np.zeros((nb, ng_total))
    for g in range(ng_total):
        g_bus_idx = bus_map.get(int(gen[g, GEN_BUS]))
        if g_bus_idx is not None:
            Cg[g_bus_idx, g] = 1.0

    # Reference bus (one per island)
    ref_buses = _find_ref_buses(bus, branch, bus_map)

    # --- Decision variables: x = [Pg (ng_total), Va (nb)] ---
    n_vars = ng_total + nb

    # Objective: min c_pg' * Pg + 0 * Va
    c_obj = np.zeros(n_vars)
    for g in range(ng_total):
        if gencost[g, NCOST] >= 2:
            c_obj[g] = gencost[g, COST]

    # Bounds
    pg_min = gen[:, PMIN] / baseMVA
    pg_max = gen[:, PMAX] / baseMVA
    bounds = (
        [(pg_min[g], pg_max[g]) for g in range(ng_total)] +
        [(-np.inf, np.inf) if i not in ref_buses else (0, 0) for i in range(nb)]
    )

    # Equality: Cg * Pg - Bbus * Va = 0  (power balance at each bus)
    A_eq = np.hstack([Cg, -Bbus])
    b_eq = np.zeros(nb)

    # Inequality: -RATE_A <= Bf * Va <= RATE_A
    rate_a = branch[:, RATE_A] / baseMVA
    rate_a[rate_a == 0] = 1e10  # unlimited

    # [0 | Bf] * x <= rate_a  and  [0 | -Bf] * x <= rate_a
    zeros_ng = np.zeros((nl, ng_total))
    A_ub = np.vstack([
        np.hstack([zeros_ng, Bf]),
        np.hstack([zeros_ng, -Bf]),
    ])
    b_ub = np.concatenate([rate_a, rate_a])

    # --- Solve with dual-simplex (matching patched qps_ot.m) ---
    try:
        result = linprog(
            c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
            bounds=bounds, method='highs-ds',
            options={'disp': False, 'presolve': True},
        )
    except Exception:
        result = type('R', (), {'success': False})()

    if not result.success:
        # Fallback to interior-point
        try:
            result = linprog(
                c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                bounds=bounds, method='highs-ipm',
                options={'disp': False, 'presolve': True},
            )
        except Exception:
            return {'success': False}

    if not result.success:
        return {'success': False}

    # Extract Pg results (scale back to MW)
    pg_result = result.x[:ng_total] * baseMVA

    gen_result = gen.copy()
    gen_result[:, PG] = pg_result

    return {'success': True, 'gen': gen_result}


def _find_ref_buses(bus, branch, bus_map):
    """Find one reference bus per connected island."""
    nb = bus.shape[0]
    bus_ids = bus[:, BUS_I].astype(int)

    # Build adjacency
    adj = [[] for _ in range(nb)]
    for row in branch:
        f = bus_map.get(int(row[F_BUS]))
        t = bus_map.get(int(row[T_BUS]))
        if f is not None and t is not None:
            adj[f].append(t)
            adj[t].append(f)

    # BFS to find islands, pick ref bus for each
    visited = [False] * nb
    ref_buses = set()
    for start in range(nb):
        if visited[start]:
            continue
        # Find all buses in this island
        island = []
        queue = [start]
        visited[start] = True
        while queue:
            node = queue.pop(0)
            island.append(node)
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        # Pick existing REF bus, or first bus in island
        ref_found = False
        for i in island:
            if bus[i, BUS_TYPE] == REF:
                ref_buses.add(i)
                ref_found = True
                break
        if not ref_found:
            ref_buses.add(island[0])

    return ref_buses


# ---------------------------------------------------------------------------
# func_dcopt (port of func_dcopt.m)
# ---------------------------------------------------------------------------

def func_dcopt(system_state, ppc0):
    """
    Compute the relative blackout size (%) given a system state vector.

    Port of func_dcopt.m — uses scipy.optimize.linprog (dual-simplex)
    for the DC-OPF, matching the patched qps_ot.m solver.

    Args:
        system_state: 1D array of length (nb + nl).
            - system_state[0:nb]: bus states (1 = failed, 0 = operational,
              fractional values for partial generator degradation)
            - system_state[nb:nb+nl]: branch states (1 = failed, 0 = operational)
        ppc0: base PYPOWER case dict (with branch capacities already set)

    Returns:
        blackout_size: float, percentage of total load not served
        flag: int, 1 if OPF converged, 0 otherwise
    """
    # Basic network info
    bus_dic = ppc0['bus'][:, BUS_I].copy()
    gen_dic = ppc0['gen'][:, GEN_BUS].copy()
    branch_dic = ppc0['branch'][:, [F_BUS, T_BUS]].copy()

    nb = len(bus_dic)
    ng = len(gen_dic)
    nl = len(branch_dic)

    # Parse system state
    bus_state = system_state[:nb].copy()
    branch_state = system_state[nb:nb + nl].copy()

    # Generator state derived from bus state
    gen_idx = np.array([np.where(bus_dic == g)[0][0] for g in gen_dic])
    gen_state = bus_state[gen_idx]

    # Convert fixed loads to dispatchable loads
    mpc = load2disp(ppc0)

    # No cost for original generators
    mpc['gencost'][:ng, :] = 0
    mpc['gencost'][:ng, MODEL] = POLYNOMIAL
    mpc['gencost'][:ng, NCOST] = 2

    # Zero out shunt conductance and undispatchable loads
    mpc['bus'][:, PD] = 0
    mpc['bus'][:, GS] = 0
    mpc['gen'][:ng, PMIN] = 0
    mpc['gen'][:ng, GEN_STATUS] = 1  # all generators in service

    # Total real power demand (sum of negative Pg from dispatchable loads)
    totpf = -np.sum(mpc['gen'][ng:, PG])

    # --- Remove failed buses ---
    removed_bus_idx = np.where(bus_state == 1)[0]
    removed_bus_no = bus_dic[removed_bus_idx]

    # Delete bus rows
    mpc['bus'] = np.delete(mpc['bus'], removed_bus_idx, axis=0)

    # --- Update generator capacity for partial degradation ---
    mpc['gen'][:ng, QMAX] *= (1 - gen_state)
    mpc['gen'][:ng, QMIN] *= (1 - gen_state)
    mpc['gen'][:ng, PMAX] *= (1 - gen_state)
    mpc['gen'][:ng, PMIN] *= (1 - gen_state)

    # Remove generators/loads at failed buses
    gen_at_removed = np.isin(mpc['gen'][:, GEN_BUS], removed_bus_no)
    mpc['gen'] = mpc['gen'][~gen_at_removed]
    mpc['gencost'] = mpc['gencost'][~gen_at_removed]

    # --- Remove failed branches ---
    removed_branch_idx = np.where(branch_state == 1)[0]
    br_from_removed = np.isin(branch_dic[:, 0], removed_bus_no)
    br_to_removed = np.isin(branch_dic[:, 1], removed_bus_no)
    all_removed_br = np.zeros(nl, dtype=bool)
    all_removed_br[removed_branch_idx] = True
    all_removed_br |= br_from_removed | br_to_removed

    mpc['branch'] = mpc['branch'][~all_removed_br]

    # --- Run DC-OPF ---
    results = _solve_dcopf(mpc)

    if not results['success']:
        return 100.0, 0

    # --- Compute blackout size ---
    n_surviving_gen = int(np.sum(gen_state != 1))

    surviving_gen_status = mpc['gen'][:n_surviving_gen, GEN_STATUS]
    surviving_gen_pg = results['gen'][:n_surviving_gen, PG]
    realpf = np.dot(surviving_gen_status, surviving_gen_pg)

    blackout_size = 100.0 * round(abs(realpf - totpf) / totpf * 1e8) / 1e8

    return blackout_size, 1


# ---------------------------------------------------------------------------
# Main: test against known Octave results
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import time

    # Load case14 from MATPOWER .m file for consistency
    matpower_case14 = '/mnt/c/Projects/matpower8.1/data/case14.m'
    ppc0 = load_case(matpower_case14)
    ppc0 = add_branch_capacity(ppc0, alpha=2.0)

    nb = ppc0['bus'].shape[0]
    nl = ppc0['branch'].shape[0]
    print(f"Loaded case14: {nb} buses, {ppc0['gen'].shape[0]} generators, {nl} branches\n")

    tests = [
        ("All operational", {}),
        ("Branch 3 failed", {nb + 2: 1}),
        ("Branches 1,2 failed", {nb + 0: 1, nb + 1: 1}),
        ("Bus 1 failed", {0: 1}),
        ("Bus 1 + branch 3", {0: 1, nb + 2: 1}),
        ("Branches 3,8,10", {nb + 2: 1, nb + 7: 1, nb + 9: 1}),
        ("5 branches failed", {nb + 0: 1, nb + 1: 1, nb + 2: 1, nb + 3: 1, nb + 4: 1}),
        ("Buses 1,2 failed", {0: 1, 1: 1}),
        ("Buses 1,2,3 failed", {0: 1, 1: 1, 2: 1}),
        ("6 branches (8-13)", {nb + 7: 1, nb + 8: 1, nb + 9: 1, nb + 10: 1, nb + 11: 1, nb + 12: 1}),
        ("Gens at 40% cap", {0: 0.6, 1: 0.6}),
    ]

    for name, failures in tests:
        ss = np.zeros(nb + nl)
        for idx, val in failures.items():
            ss[idx] = val
        t0 = time.time()
        bs, flag = func_dcopt(ss, ppc0)
        dt = time.time() - t0
        print(f"  {name:30s}  blackout={bs:7.4f}%  flag={flag}  ({dt:.3f}s)")
