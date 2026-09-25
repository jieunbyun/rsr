"""
TSUM system function (sfun) that wraps the pure Python DC-OPF blackout model.

No MATLAB/Octave dependency — uses scipy.optimize.linprog for DC-OPF.

Usage:
    from sfun_dcopt import make_dcopt_sfun

    sfun = make_dcopt_sfun(
        case_path='/mnt/c/Projects/matpower8.1/data/case14.m',
        blackout_threshold=54.8,
        alpha=2.0,
    )

    # comps_st: dict mapping component IDs to state indices
    # e.g., {'br1': 1, 'br2': 0, 'vbus1': 3, ...}
    fval, sys_st, min_comps_st = sfun(comps_st)
"""

import numpy as np
from typing import Dict, Tuple, Optional

from func_dcopt_py import load_case, add_branch_capacity, func_dcopt


def make_dcopt_sfun(
    case_path: str = '/mnt/c/Projects/matpower8.1/data/case14.m',
    blackout_threshold: float = 54.8,
    alpha: float = 2.0,
):
    """
    Create a TSUM-compatible system function that evaluates blackout size
    via pure Python DC-OPF (scipy linprog with dual-simplex).

    Args:
        case_path: Path to MATPOWER .m case file, or PYPOWER case name
        blackout_threshold: Blackout size (%) above which system is failed
        alpha: Branch capacity scaling factor for add_branch_capacity

    Returns:
        sfun: callable (comps_st: Dict[str, int]) -> (fval, sys_st, min_comps_st)
    """
    # Load and prepare the case once
    ppc0 = load_case(case_path)
    ppc0 = add_branch_capacity(ppc0, alpha)

    # Extract network info
    bus_ids = ppc0['bus'][:, 0].astype(int).tolist()
    gen_bus_ids = set(ppc0['gen'][:, 0].astype(int).tolist())
    nb = len(bus_ids)
    ng = len(gen_bus_ids)
    nl = ppc0['branch'].shape[0]

    print(f"DC-OPF sfun ready: {nb} buses, {ng} generators, {nl} branches")

    # State mappings: TSUM state index -> MATPOWER systemState value
    gen_state_map = {0: 1.0, 1: 0.6, 2: 0.2, 3: 0.0}
    bus_state_map = {0: 1.0, 1: 0.0}
    branch_state_map = {0: 1.0, 1: 0.0}

    def sfun(comps_st: Dict[str, int]) -> Tuple[float, int, Optional[Dict]]:
        """
        Evaluate system state given component states.

        Args:
            comps_st: Dict mapping component IDs to state indices.
                - 'vbus{id}': bus component (state 0=failed ... 3=operational for generators)
                - 'br{i}': branch component (state 0=failed, 1=operational)

        Returns:
            (blackout_size, sys_st, None)
        """
        system_state = np.zeros(nb + nl)

        # Map bus components
        for i, bus_id in enumerate(bus_ids):
            key = f"vbus{bus_id}"
            if key in comps_st:
                tsum_state = comps_st[key]
                if bus_id in gen_bus_ids:
                    system_state[i] = gen_state_map.get(tsum_state, 0.0)
                else:
                    system_state[i] = bus_state_map.get(tsum_state, 0.0)

        # Map branch components
        for j in range(nl):
            key = f"br{j+1}"
            if key in comps_st:
                tsum_state = comps_st[key]
                system_state[nb + j] = branch_state_map.get(tsum_state, 0.0)

        # Compute blackout
        blackout_size, flag = func_dcopt(system_state, ppc0)

        if flag == 0:
            sys_st = 0
            blackout_size = 100.0
        else:
            sys_st = 1 if blackout_size < blackout_threshold else 0

        return blackout_size, sys_st, None

    return sfun


if __name__ == '__main__':
    import time

    sfun = make_dcopt_sfun(
        case_path='/mnt/c/Projects/matpower8.1/data/case14.m',
        blackout_threshold=54.8,
    )

    # Test 1: All operational (branches only)
    comps_st_ok = {f'br{i}': 1 for i in range(1, 21)}
    t0 = time.time()
    fval, sys_st, _ = sfun(comps_st_ok)
    t1 = time.time()
    print(f"All operational:    blackout={fval:7.4f}%, sys_st={sys_st}  ({t1-t0:.3f}s)")

    # Test 2: Branch 3 failed
    comps_st_br3 = {f'br{i}': 1 for i in range(1, 21)}
    comps_st_br3['br3'] = 0
    t0 = time.time()
    fval, sys_st, _ = sfun(comps_st_br3)
    t1 = time.time()
    print(f"Branch 3 failed:    blackout={fval:7.4f}%, sys_st={sys_st}  ({t1-t0:.3f}s)")

    # Test 3: Buses 1,2,3 failed (with bus components)
    comps_st_bus = {f'br{i}': 1 for i in range(1, 21)}
    for bid in range(1, 15):
        comps_st_bus[f'vbus{bid}'] = 1
    comps_st_bus['vbus1'] = 0
    comps_st_bus['vbus2'] = 0
    comps_st_bus['vbus3'] = 0
    t0 = time.time()
    fval, sys_st, _ = sfun(comps_st_bus)
    t1 = time.time()
    print(f"Buses 1,2,3 failed: blackout={fval:7.4f}%, sys_st={sys_st}  ({t1-t0:.3f}s)")

    # Test 4: Speed test
    t0 = time.time()
    n_calls = 1000
    for _ in range(n_calls):
        sfun(comps_st_ok)
    elapsed = time.time() - t0
    print(f"\n{n_calls} calls: {elapsed:.2f}s total, {elapsed/n_calls*1000:.1f}ms per call")
