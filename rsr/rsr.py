import torch
import itertools
import operator
from itertools import product, combinations
from math import prod
import math
from decimal import Decimal
import numpy as np
import os, json, time
from typing import Callable, Dict, Any, List, Optional, Tuple, Sequence, Iterable, Union
from torch import Tensor
import psutil

import random
import multiprocessing as mp
from collections import deque

import rsr

# ---- Shared state for parallel worker processes (inherited via fork) ----
_MP_SFUN = None
_MP_SYS_UPPER_ST = None
_MP_N_STATE = None


def _minimize_one_unknown(args):
    """
    Worker function for parallel minimization of unknown samples.
    Accesses module-level shared state set before the pool is created.
    """
    comps_st_test, fval = args
    sfun = _MP_SFUN
    sys_upper_st = _MP_SYS_UPPER_ST
    n_state = _MP_N_STATE

    fval, sys_st, min_comps_st0 = sfun(comps_st_test)
    if min_comps_st0 is None:
        min_comps_st0 = comps_st_test.copy()
    elif isinstance(next(iter(min_comps_st0.values())), tuple):
        min_comps_st0 = {k: v[1] for k, v in min_comps_st0.items()}

    if sys_st >= sys_upper_st:
        min_comps_st, info = minimise_upper_states_random(
            min_comps_st0, sfun, sys_upper_st=sys_upper_st, fval=fval)
        fval = info.get('final_sys_state', fval)
    else:
        min_comps_st, info = minimise_lower_states_random(
            min_comps_st0, sfun, max_state=n_state - 1,
            sys_lower_st=sys_upper_st - 1, fval=fval)
        fval = info.get('final_sys_state', fval)

    return min_comps_st, sys_st, fval

# For use in mixted sorting 
try:
    import numpy as np
    _NUMPY_NUM = (np.integer, np.floating)
except Exception:
    _NUMPY_NUM = tuple()


def get_min_lower_comps_st(comps_st, max_st, sys_lower_st):
    """
    Get the minimal lower reference component states from a given state,
    by recording components in comps_st != max_st

    Args:
        comps_st (dict): {comp_name: state (int)}
        max_st (int): the highest state
        sys_lower_st (int): the system lower reference state

    Returns:
        (dict): {comp_name: ('comparison_operator', state (int))}

    """
    min_comps_st = {k: ('<=', v) for k, v in comps_st.items() if v < max_st}
    min_comps_st['sys'] = ('<=', sys_lower_st)
    return min_comps_st


def get_min_upper_comps_st(comps_st, sys_upper_st):
    """
    Get the minimal upper reference component states from a given state,
    by recording components in comps_st != max_st

    Args:
        comps_st (dict): {comp_name: state (int)}
        sys_upper_st (int): the system upper reference state

    Returns:
        (dict): {comp_name: ('comparison_operator', state (int))}

    """
    min_comps_st = {k: ('>=', v) for k, v in comps_st.items() if v > 0}
    min_comps_st['sys'] = ('>=', sys_upper_st)
    return min_comps_st


def minimise_upper_states_random(
    comps_st: Dict[str, int],
    sfun: Callable[[Dict[Any, int]], Tuple[Any, Tuple[str, int], Dict[Any, int]]],
    sys_upper_st: int,
    *,
    fval: Optional[Any] = None,
    min_state: int = 0,
    step: int = 1,
    seed: Optional[int] = None,
    exclude_keys: Iterable[str] = ("sys",)
) -> Tuple[Dict[str, int], Dict[str, Any]]:
    """
    Random greedy reduction of component states.

    Algorithm (given a random permutation of components):
      - Try lowering each component by `step` (e.g., 1).
      - Call sfun(modified_state).
        Expect sfun to return a tuple where the 2nd element is int that represents a system state.
      - If status >= sys_upper_st: keep the lowered value and continue cycling.
        If the component reaches `min_state`, remove it (can't lower further).
      - If status < sys_lower_st: revert the change and remove that component (no further attempts).

    Stops when all components have been removed from the candidate pool.

    Returns:
      final_state, info
        - final_state: dict of the minimized states.
        - info: {
            'permutation': [...],
            'removed_on_lower': [comp,...],
            'hit_min_state': [comp,...],
            'attempts': int,
          }
    """
    rng = random.Random(seed)

    # Work on a (shallow) copy; do NOT mutate caller's dict (value int is immutable)
    state = dict(comps_st)

    # Build candidate component key deque from a random permutation
    candidates = [k for k, v in state.items()
                  if k not in set(exclude_keys) and isinstance(v, int) and v > min_state]
    rng.shuffle(candidates)
    dq = deque(candidates)

    removed_on_lower = []
    hit_min_state = []
    attempts = 0

    while dq:
        comp = dq[0]

        # If already at/below min_state, remove and continue
        #if state.get(comp, min_state) <= min_state: # state[comp] always works
        if state[comp] <= min_state: # state[comp] always works
            dq.popleft()
            hit_min_state.append(comp)
            continue

        prev = state[comp]
        fval_prev = fval
        state[comp] = prev - step
        attempts += 1

        # Expect sfun to return (value, 's'/'f', info) or similar
        try:
            fval, status, _ = sfun(state)
        except Exception as e:
            # If your sfun has a different signature, surface the error clearly
            state[comp] = prev  # revert
            fval = fval_prev
            dq.popleft()
            removed_on_lower.append(comp)
            continue

        if status >= sys_upper_st:
            # Keep lowered value
            if state[comp] <= min_state:
                dq.popleft()
                hit_min_state.append(comp)
            else:
                dq.rotate(-1)  # move to back; try again later
        else:
            # Revert and remove from further consideration
            state[comp] = prev
            fval = fval_prev
            dq.popleft()
            removed_on_lower.append(comp)

    info = {
        'permutation': candidates,
        'removed_on_lower': removed_on_lower,
        'hit_min_state': hit_min_state,
        'attempts': attempts,
        'final_state': state,
        'final_sys_state': fval
    }

    min_ref = get_min_upper_comps_st(state, sys_upper_st)

    return min_ref, info


def minimise_lower_states_random(
    comps_st: Dict[str, int],
    sfun,
    sys_lower_st: int,
    max_state: int,
    *,
    fval: Optional[Any] = None,
    step: int = 1,
    seed: Optional[int] = None,
    exclude_keys: Iterable[str] = ("sys",)
) -> Tuple[Dict[str, int], Dict[str, Any]]:
    """
    Random greedy reduction of component states.

    Algorithm (given a random permutation of components):
      - Try increasing each component by `step` (e.g., 1).
      - Call sfun(modified_state).
        Expect sfun to return a tuple where the 2nd element is an int representing a system state.
      - If status <= sys_lower_st: keep the increased value and continue cycling.
        If the component reaches `max_state`, remove it (can't increase further).
      - If status > sys_lower_st: revert the change and remove that component (no further attempts).

    Stops when all components have been removed from the candidate pool.

    Returns:
      final_state, info
        - final_state: dict of the minimized states.
        - info: {
            'permutation': [...],
            'removed_on_lower': [comp,...],
            'hit_min_state': [comp,...],
            'attempts': int,
            'final_state': {comp: state,...}
          }
    """
    rng = random.Random(seed)

    # Work on a copy; do NOT mutate caller's dict
    state = dict(comps_st)

    # Build candidate deque from a random permutation
    candidates = [k for k, v in state.items()
                  if k not in set(exclude_keys) and isinstance(v, int) and v < max_state]
    rng.shuffle(candidates)
    dq = deque(candidates)

    removed_on_upper = []
    hit_min_state = []
    attempts = 0

    while dq:
        comp = dq[0]

        # If already at/below min_state, remove and continue
        if state.get(comp, max_state) >= max_state:
            dq.popleft()
            hit_min_state.append(comp)
            continue

        prev = state[comp]
        fval_prev = fval
        state[comp] = prev + step
        attempts += 1

        # Expect sfun to return (value, 's'/'f', info) or similar
        try:
            fval, status, _ = sfun(state)
        except Exception as e:
            # If your sfun has a different signature, surface the error clearly
            state[comp] = prev  # revert
            fval = fval_prev
            dq.popleft()
            removed_on_upper.append(comp)
            continue

        if status <= sys_lower_st:
            # Keep increased value
            if state[comp] >= max_state:
                dq.popleft()
                hit_min_state.append(comp)
            else:
                dq.rotate(-1)  # move to back; try again later
        else:
            # Revert and remove from further consideration
            state[comp] = prev
            fval = fval_prev
            dq.popleft()
            removed_on_upper.append(comp)

    info = {
        'permutation': candidates,
        'removed_on_upper': removed_on_upper,
        'hit_min_state': hit_min_state,
        'attempts': attempts,
        'final_state': state,
        'final_sys_state': fval
    }

    min_ref = get_min_lower_comps_st(state, max_state, sys_lower_st)

    return min_ref, info


def from_ref_dict_to_mat(ref_dict, row_names, max_st):
    """
    Convert a ref dictionary to a matrix representation.

    Args:
        ref_dict (dict): {name: ('comparison_operator', state (int))}
        row_names (list): list of component names associated with each row in order
        max_st (int): the highest state

    Returns:
        mat (list): binary matrix with shape (n_comp, max_st)

    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mat = torch.zeros((len(row_names), max_st), dtype=torch.int32, device=device)

    for row, name in enumerate(row_names):  
        if name in ref_dict:
            op, state = ref_dict[name]
            if op == '<=':
                mat[row, :state + 1] = 1
            elif op == '<':
                mat[row, :state] = 1
            elif op == '>=':
                mat[row, state:] = 1
            elif op == '>':
                mat[row, state + 1:] = 1
            elif op == '==':
                mat[row, state] = 1
        else:
            mat[row, :] = 1

    return mat

def from_Bbound_to_comps_st(Bbound, row_names):
    """
    Extracts the index of the first non-zero state for each component (ignoring the system row).

    Args:
        Bbound (Tensor): shape (n_var, n_state)
        row_names (list): list of variable names including system

    Returns:
        comps_st (dict): {component_name: state_index}
    """
    n_var, n_state = Bbound.shape

    comps_st = {}
    for i in range(n_var):
        row = Bbound[i]
        nz = torch.nonzero(row, as_tuple=False)
        if len(nz) > 0:
            comps_st[row_names[i]] = int(nz[0])
        else:
            comps_st[row_names[i]] = None  # or raise an error

    return comps_st

def get_branches_cap_branches(B1, B2, batch_size=64):
    """
    Memory-efficient intersection of branches with batching over the larger tensor (B1 or B2).
    Inputs:
        B1: (n_br1, n_var, n_state)
        B2: (n_br2, n_var, n_state)
    Returns:
        Bnew: (n_valid, n_var, n_state)
    """
    device = B1.device
    n_br1, n_var, n_state = B1.shape
    n_br2 = B2.shape[0]
    results = []

    if n_br1 >= n_br2:
        # Batch over B1
        for start in range(0, n_br1, batch_size):
            end = min(start + batch_size, n_br1)
            B1_batch = B1[start:end]                    # (batch_size, n_var, n_state)

            B1_exp = B1_batch.unsqueeze(1)              # (batch_size, 1, n_var, n_state)
            B2_exp = B2.unsqueeze(0)                    # (1, n_br2, n_var, n_state)
            Bnew = B1_exp & B2_exp                      # (batch_size, n_br2, n_var, n_state)
            Bnew = Bnew.view(-1, n_var, n_state)

            # Filter invalid
            invalid_mask = (Bnew == 0).all(dim=2)
            keep_mask = ~invalid_mask.any(dim=1)
            Bnew = Bnew[keep_mask]

            results.append(Bnew)
    else:
        # Batch over B2
        for start in range(0, n_br2, batch_size):
            end = min(start + batch_size, n_br2)
            B2_batch = B2[start:end]

            B1_exp = B1.unsqueeze(1)                    # (n_br1, 1, n_var, n_state)
            B2_exp = B2_batch.unsqueeze(0)              # (1, batch_size, n_var, n_state)
            Bnew = B1_exp & B2_exp                      # (n_br1, batch_size, n_var, n_state)
            Bnew = Bnew.view(-1, n_var, n_state)

            invalid_mask = (Bnew == 0).all(dim=2)
            keep_mask = ~invalid_mask.any(dim=1)
            Bnew = Bnew[keep_mask]

            results.append(Bnew)

    if results:
        return torch.cat(results, dim=0)
    else:
        return torch.empty((0, n_var, n_state), dtype=B1.dtype, device=device)

def get_complementary_events(mat):
    """
    Given a (n_vars, n_state) matrix with the last row as the system event,
    generate a set of complementary logical events (one per component).

    Returns:
        Bnew: (n_comps_kept, n_vars, n_state)
    """
    n_vars, n_state = mat.shape

    # Prepare output tensor
    B = torch.ones((n_vars, n_vars, n_state), dtype=mat.dtype, device=mat.device)

    # Broadcast mat for all i
    mat_exp = mat.unsqueeze(0).expand(n_vars, n_vars, n_state)

    # Create lower-triangular mask to copy rows before i
    mask = torch.arange(n_vars, device=mat.device).unsqueeze(0) < torch.arange(n_vars, device=mat.device).unsqueeze(1)  # (n_vars, n_vars)
    mask = mask.unsqueeze(-1).expand(-1, -1, n_state)  # (n_vars, n_vars, n_state)
    B[mask] = mat_exp[mask]  # copy rows before i

    # Flip row i in each batch
    flip_mask = torch.eye(n_vars, dtype=torch.bool, device=mat.device).unsqueeze(-1).expand(-1, -1, n_state)  # (n_vars, n_vars, n_state)
    B[:n_vars, :n_vars][flip_mask] = 1 - mat_exp[:n_vars, :n_vars][flip_mask]

    # Remove combinations where any row (excluding system) is all-zero across states
    invalid_mask = (B[:, :-1, :] == 0).all(dim=2)  # shape: (n_vars, n_vars)
    keep_mask = ~invalid_mask.any(dim=1)          # shape: (n_vars,)
    Bnew = B[keep_mask]

    return Bnew

def get_branch_probs(tensor, prob):
    """
    Computes the probability of each branch given a binary event tensor and state probabilities.

    Args:
        tensor: (n_br, n_var, n_state) - binary indicator of active states per variable per branch
        prob:   (n_var, n_state)   - probability per state for each component variable

    Returns:
        Bprob: (n_br,) - probability per branch
    """
    n_br, n_var, n_state = tensor.shape
    device = tensor.device

    # Expand to match tensor: (n_br, n_comps, n_state)
    prob_exp = prob.unsqueeze(0).expand(n_br, -1, -1)

    # Element-wise multiplication and summing across states
    prob_selected = tensor * prob_exp  # (n_br, n_comps, n_state)
    prob_per_var = prob_selected.sum(dim=2)  # (n_br, n_comps)
    Bprob = prob_per_var.prod(dim=1)  # (n_br,)

    return Bprob

import torch

def get_boundary_branches(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute boundary branches for each input branch.

    Input:
        tensor: (n_vars, n_state)  OR  (n_br, n_vars, n_state)
                int/bool tensor with 0/1 entries.

                - n_vars includes the system row as the LAST row.
                - For each component row, the active state(s) are 1s along n_state.
                  We pick:
                    * 'lower' boundary: first active state (min index)
                    * 'upper' boundary: last  active state (max index)

    Output:
        (2, n_vars, n_state)             if input was (n_vars, n_state)
        (2*n_br, n_vars, n_state)        if input was (n_br, n_vars, n_state)

        The last row (system row) is set to all 1s in both upper and lower outputs.
    """
    assert tensor.ndim in (2, 3), "Input must be 2D (n_vars,n_state) or 3D (n_br,n_vars,n_state)"

    # Normalize to 3D (batch of branches)
    squeeze_back = (tensor.ndim == 2)
    if squeeze_back:
        x = tensor.unsqueeze(0)  # (1, n_vars, n_state)
    else:
        x = tensor               # (n_br, n_vars, n_state)

    n_br, n_vars, n_state = x.shape
    # n_comps = n_vars - 1 # OBSOLETE: system row is now excluded from input
    n_comps = n_vars

    # Work only on component rows (exclude final system row)
    comp = x[:, :n_comps, :]             # (n_br, n_comps, n_state)
    mask = comp.bool()

    # First active state index per component
    first_hit = (mask.float().cumsum(dim=-1) == 1).float()
    first_idx = first_hit.argmax(dim=-1)  # (n_br, n_comps)

    # Last active state index per component
    rev = torch.flip(mask, dims=[-1])
    last_hit = (rev.float().cumsum(dim=-1) == 1).float()
    last_idx = last_hit.argmax(dim=-1)
    last_idx = (n_state - 1) - last_idx   # (n_br, n_comps)

    # (Optional) If a row has no 1s at all, both argmax above return 0.
    # If you want to *suppress* placing a 1 in such rows, detect and skip:
    # has_one = mask.any(dim=-1)                              # (n_br, n_comps)
    # first_idx = torch.where(has_one, first_idx, -1)         # -1 will be ignored by scatter_
    # last_idx  = torch.where(has_one,  last_idx,  -1)

    # Build lower/upper with one-hot at first/last indices
    lower = torch.zeros_like(comp)
    upper = torch.zeros_like(comp)

    lower.scatter_(-1, first_idx.unsqueeze(-1), 1)
    upper.scatter_(-1, last_idx.unsqueeze(-1), 1)

    # Stack branches: [upper; lower] along branch dimension
    out = torch.cat([upper, lower], dim=0)   # (2*n_br, n_vars, n_state)

    # Squeeze back if original was 2D: return shape (2, n_vars, n_state)
    return out if not squeeze_back else out.view(2, n_vars, n_state)


# FIXME: unused
def get_boundary_refs(tensor):
    n_br, n_vars, n_state = tensor.shape
    #n_comps = n_vars - 1 # exclude system event (last row) <- OUTDATED: system row is now excluded from input
    n_comps = n_vars

    comp_tensor = tensor[:, :n_comps, :]  # (n_br, n_comps, n_state)

    # Create boolean mask of active entries
    mask = comp_tensor.bool()  # (n_br, n_comps, n_state)

    # Get first and last nonzero indices
    first_idx = mask.float().cumsum(dim=2)
    first_idx = (first_idx == 1).float()
    first_idx = first_idx.argmax(dim=2)  # (n_br, n_comps)

    # Reverse to find last
    reversed_mask = torch.flip(mask, dims=[2])
    last_idx = reversed_mask.float().cumsum(dim=2)
    last_idx = (last_idx == 1).float()
    last_idx = last_idx.argmax(dim=2)
    last_idx = n_state - 1 - last_idx  # reverse indices

    # Build upper and lower tensors
    ###### only this part is different from get_boundary_branches #####
    state_idx = torch.arange(n_state, device=last_idx.device).view(1, 1, -1).expand(n_br, n_comps, -1)
    upper = (state_idx >= last_idx.unsqueeze(-1)).to(tensor.dtype)
    lower = (state_idx <= first_idx.unsqueeze(-1)).to(tensor.dtype)
    ####################################################################

    # Append system row of all 1s 
    #system = torch.ones((n_br, 1, n_state), dtype=tensor.dtype, device=tensor.device)

    #B_upper = torch.cat([upper, system], dim=1)
    B_upper = upper
    #B_lower = torch.cat([lower, system], dim=1)
    B_lower = lower

    return torch.cat([B_upper, B_lower], dim=0)  # shape: (2*n_br, n_vars, n_state)

# FIXME: ununsed
def is_intersect(events1, events2):
    """
    Determine whether each event in events1 intersects with any event in events2.

    Args:
        events1: (n_event1, n_vars, n_state)
        events2: (n_event2, n_vars, n_state)

    Returns:
        labels: (n_event1,) boolean tensor
    """
    n_event1, n_vars, n_state = events1.shape
    n_event2, _, _ = events2.shape

    # Expand for broadcasting
    events1_exp = events1.unsqueeze(1).expand(-1, n_event2, -1, -1)
    events2_exp = events2.unsqueeze(0).expand(n_event1, -1, -1, -1)

    # Compute intersection and check if any is non-zero per pair
    intersect = events1_exp & events2_exp  # logical AND
    is_empty = (intersect == 0).all(dim=3).any(dim=2)  # shape: (n_event1, n_event2)
    labels = ~is_empty.all(dim=1)  # if any intersected, mark True

    return labels

def is_subset(mat, tensor):
    """
    Checks if:
      1. `mat` is a subset of any of the events in `tensor`, and
      2. Any of the events in `tensor` is a subset of `mat`.

    Args:
        mat: Tensor of shape (n_var, n_state)
        tensor: Tensor of shape (n_event, n_var, n_state)

    Returns:
        is_mat_subset: bool
        is_tensor_subset: BoolTensor of shape (n_event,)
    """
    n_event, n_var, n_state = tensor.shape
    mat_e = mat.unsqueeze(0).expand(n_event, -1, -1)  # (n_event, n_var, n_state)

    intersect = mat_e & tensor  # (n_event, n_var, n_state)

    is_mat_subset = torch.any(torch.all(mat_e == intersect, dim=(1, 2))).item()
    is_tensor_subset = torch.all(tensor == intersect, dim=(1, 2))  # shape (n_event,)

    return bool(is_mat_subset), is_tensor_subset

import torch
from math import prod

@torch.no_grad()
def find_first_nonempty_combination(Rcs, batch_size=65536, verbose=False):
    """
    Rcs: list[Tensor] with shapes (n_i, n_vars, n_state), same (n_vars, n_state)
    Order: increasing sum of tuple indices, then lexicographic within that sum.
    Returns: (selected_mat: (n_vars, n_state), idx_tuple) or (None, None)
    """
    assert len(Rcs) > 0
    device = Rcs[0].device
    n_vars, n_state = Rcs[0].shape[1:]
    ns = torch.tensor([r.shape[0] for r in Rcs], device=device, dtype=torch.long)
    k = len(ns)
    assert all((r.device == device and r.shape[1]==n_vars and r.shape[2]==n_state) for r in Rcs)

    # total combinations and linear-index strides (mixed radix, right-to-left)
    n_combs = int(torch.prod(ns).item())
    if n_combs == 0:
        return None

    strides = torch.ones_like(ns)
    if k > 1:
        strides[:-1] = torch.cumprod(ns.flip(0)[:-1], dim=0).flip(0)  # lex rank weights too

    # maximum possible sum level
    max_sum = int((ns - 1).sum().item())

    # scan sum shells s = 0..max_sum
    for s in range(max_sum + 1):
        if verbose:
            print(f"[sum={s}] scanning...")
        start = 0

        best_lex_rank = None
        best_sel_mat = None
        best_tuple = None
        best_global_idx = None

        while start < n_combs:
            end = min(start + batch_size, n_combs)

            # linear indices (GPU)
            lin = torch.arange(start, end, device=device, dtype=torch.long)

            # decode to tuples (batch, k)
            idx = (lin[:, None] // strides) % ns

            # filter rows at the current sum level
            sum_mask = (idx.sum(dim=1) == s)
            if sum_mask.any():
                idx_s = idx[sum_mask]
                # gather the needed rows only (saves compute)
                mats = [r[idx_s[:, i]] for i, r in enumerate(Rcs)]
                mat = torch.stack(mats, dim=0).prod(dim=0)  # (batch_s, n_vars, n_state)

                # non-empty check (your original ref)
                is_empty = (mat == 0).all(dim=2).any(dim=1)
                valid = ~is_empty

                if valid.any():
                    # lexicographic rank within this sum shell
                    lex_rank = (idx_s * strides).sum(dim=1)  # dot with strides
                    # among valid, choose min lex
                    lex_rank_valid = lex_rank.clone()
                    # mask out invalid by setting to +inf
                    lex_rank_valid[~valid] = torch.iinfo(torch.int64).max

                    # candidate in this batch
                    batch_min_lex, batch_pos = torch.min(lex_rank_valid, dim=0)
                    if batch_min_lex != torch.iinfo(torch.int64).max:
                        # global best within this sum s (merge across batches)
                        if (best_lex_rank is None) or (batch_min_lex < best_lex_rank):
                            best_lex_rank = batch_min_lex
                            best_sel_mat = mat[batch_pos]              # (n_vars, n_state)
                            best_tuple = tuple(int(v) for v in idx_s[batch_pos].tolist())
                            best_global_idx = int((idx_s[batch_pos] * strides).sum().item())

            start = end

        if best_sel_mat is not None:
            if verbose:
                print(f"Selected index: {best_tuple} (sum={s}, lex_rank={int(best_lex_rank)}, lin={best_global_idx})")
            return best_sel_mat

    return None


# FIXME: unused
def sum_sorted_tuples_limited(max_vals):
    """
    Generate all tuples of non-negative integers with len=max_vals,
    where each element i ≤ max_vals[i],
    ordered by increasing sum, then lexicographically.
    
    Args:
        max_vals (list or tuple): list of maximum values per position.
    
    Yields:
        tuple of ints
    """
    n = len(max_vals)
    sum_level = 0
    while True:
        found = False
        for t in itertools.product(*(range(v+1) for v in max_vals)):
            if sum(t) == sum_level:
                yield t
                found = True
        if not found:
            break  # no more combinations possible
        sum_level += 1

# FIXME: unused
def merge_branches(B):
    "Use hashing for computational efficiency"

    is_merge = True

    while is_merge:
        B_com = bit_compress(B)
        groups_by_col = groups_by_column_remhash_dict(B_com)
        merges = plan_merges(groups_by_col, B.shape[0])
        B, _ = apply_merges(B, merges)

        is_merge = any(len(g) > 1 for g in groups_by_col)

    return B

# FIXME: unused
def merge_branches_old(B, batch_size=100_000):
    device = B.device
    dtype = B.dtype

    B = B.clone()
    changed = True

    while changed:
        changed = False
        n_br, n_comp, n_state = B.shape
        keep_mask = torch.ones(n_br, dtype=torch.bool, device=device)
        new_branches = []

        # Generate all i < j combinations
        all_pairs = list(combinations(range(n_br), 2))
        total_pairs = len(all_pairs)

        used = torch.zeros(n_br, dtype=torch.bool, device=device)

        for start in range(0, total_pairs, batch_size):
            end = min(start + batch_size, total_pairs)
            idx_i, idx_j = zip(*all_pairs[start:end])
            idx_i = torch.tensor(idx_i, device=device)
            idx_j = torch.tensor(idx_j, device=device)

            bi = B[idx_i]  # (n_pair, n_comp, n_state)
            bj = B[idx_j]

            # Step 1: Compare along components to count differing rows
            diffs = (bi != bj).any(dim=2)  # (n_pair, n_comp)
            num_diff_rows = diffs.sum(dim=1)  # (n_pair,)
            one_diff_mask = num_diff_rows == 1

            if one_diff_mask.sum() == 0:
                continue  # no valid pairs in this batch

            valid_idx_i = idx_i[one_diff_mask]
            valid_idx_j = idx_j[one_diff_mask]
            valid_diffs = diffs[one_diff_mask]
            valid_bi = bi[one_diff_mask]
            valid_bj = bj[one_diff_mask]

            diff_row_idx = valid_diffs.float().argmax(dim=1)  # (n_valid_pairs,)

            # Extract differing rows
            ri = torch.stack([valid_bi[k, diff_row_idx[k]] for k in range(len(diff_row_idx))])
            rj = torch.stack([valid_bj[k, diff_row_idx[k]] for k in range(len(diff_row_idx))])

            disjoint_mask = (ri & rj).sum(dim=1) == 0
            if disjoint_mask.sum() == 0:
                continue

            final_merge_indices = []
            for k in range(disjoint_mask.size(0)):
                if not disjoint_mask[k]:
                    continue
                i = valid_idx_i[k].item()
                j = valid_idx_j[k].item()
                if used[i] or used[j]:
                    continue
                used[i] = True
                used[j] = True
                final_merge_indices.append(k)

            if not final_merge_indices:
                continue

            changed = True
            final_merge_indices = torch.tensor(final_merge_indices, device=device)

            disjoint_i = valid_idx_i[final_merge_indices]
            disjoint_j = valid_idx_j[final_merge_indices]
            disjoint_bi = B[disjoint_i]
            disjoint_bj = B[disjoint_j]
            disjoint_diff_idx = diff_row_idx[disjoint_mask][final_merge_indices]

            for i in range(disjoint_i.size(0)):
                merged = disjoint_bi[i].clone()
                merged[disjoint_diff_idx[i]] = disjoint_bi[i][disjoint_diff_idx[i]] | disjoint_bj[i][disjoint_diff_idx[i]]
                new_branches.append(merged)

        keep_mask[used] = False
        if new_branches:
            B = torch.cat([B[keep_mask], torch.stack(new_branches)], dim=0)

    return B

def get_complementary_events_nondisjoint(mat: torch.Tensor) -> torch.Tensor:
    """
    Given a (n_vars, n_state) matrix with the last row as the system event,
    generate a set of complementary logical events by flipping each row.
    NOTE: The resulted events are not disjoint.

    Returns:
        Bnew: (n_events_kept, n_vars, n_state)
    """
    n_vars, n_state = mat.shape

    # Prepare output tensor
    B = torch.ones((n_vars, n_vars, n_state), dtype=mat.dtype, device=mat.device)

    # Flip row i in batch i
    idx = torch.arange(n_vars, device=mat.device)
    if mat.dtype == torch.bool:
        B[idx, idx, :] = ~mat[idx, :]
    else:
        # assumes binary in {0,1}; works for float or int tensors
        B[idx, idx, :] = 1 - mat[idx, :]

    # Remove combinations where any row (excluding system) is all-zero across states
    invalid_mask = (B == 0).all(dim=2)  # shape: (n_vars, n_vars)
    keep_mask = ~invalid_mask.any(dim=1)          # shape: (n_vars,)
    Bnew = B[keep_mask]

    return Bnew

def bit_compress(B: torch.Tensor) -> torch.Tensor:
    """
    Convert a tensor B of shape (n, m, k) of bits {0,1}
    into an integer tensor of shape (n, m),
    where each element is sum_k 2^k * B[i,j,k].
    """
    n, m, k = B.shape
    # weights = [1, 2, 4, ..., 2^(k-1)]
    weights = (2 ** torch.arange(k, device=B.device, dtype=torch.int32))
    return (B.to(torch.int32) * weights).sum(dim=2)


def groups_by_column_remhash_dict(X: torch.Tensor):
    """
    For each column j, return groups of row indices that are identical on all
    other columns but differ on column j. Uses removable hashes + CPU dict
    (expected O(m n)) and includes a collision-guard verification.
    """
    X = X.to(torch.long)
    device = X.device
    n, m = X.shape
    out = [[] for _ in range(m)]
    if n == 0 or m == 0:
        return out

    # Two primes + per-column coefficients
    p1 = 2_147_483_629
    p2 = 2_147_483_647
    a1 = torch.arange(1, m + 1, device=device, dtype=torch.long)
    a2 = (a1 * 1315423911) % p2

    # Precompute full-row hashes
    H1 = (X * a1).sum(dim=1) % p1
    H2 = (X * a2).sum(dim=1) % p2

    for j in range(m):
        H1_wo = (H1 - X[:, j] * a1[j]) % p1
        H2_wo = (H2 - X[:, j] * a2[j]) % p2

        # skinny keys to CPU dict
        keys = torch.stack((H1_wo, H2_wo), dim=1).cpu().tolist()
        buckets = {}
        for i, k in enumerate(keys):
            buckets.setdefault((int(k[0]), int(k[1])), []).append(i)

        for rows in buckets.values():
            if len(rows) < 2:
                continue
            rows_t = torch.tensor(rows, device=device)
            vals_j = X[rows_t, j]
            # must truly differ at column j
            if torch.unique(vals_j).numel() < 2:
                continue

            # --- collision guard: check equality on all other columns ---
            Xg = X[rows_t]  # (s, m)
            same_cols = (Xg == Xg[0]).all(dim=0)  # (m,) True if all rows equal in that column
            # require all columns except j to be identical
            if bool(same_cols[torch.arange(m, device=device) != j].all()):
                out[j].append(rows_t)

    return out


def plan_merges(groups_per_col, n_rows):
    """
    groups_per_col: list where groups_per_col[j] is a list of 1D LongTensors of row indices (same device)
    n_rows: total number of rows
    returns: list of (i, k, j) merges, greedy, non-overlapping across all columns
    """
    # Track rows already used in a merge
    # Keep this on CPU bool for simplicity; adjust to CUDA if you prefer
    used = torch.zeros(n_rows, dtype=torch.bool)
    merges = []

    for j, groups in enumerate(groups_per_col):
        for g in groups:
            # Greedily pair left-to-right inside this group, skipping used rows
            # Note: keep device of g, but we only read its indices here
            # Collect unused indices in order
            unused = [int(idx) for idx in g.tolist() if not used[int(idx)].item()]
            # Pair consecutive unused
            for t in range(0, len(unused) - 1, 2):
                i, k = unused[t], unused[t+1]
                if used[i] or used[k]:
                    continue
                merges.append((i, k, j))
                used[i] = True
                used[k] = True
            # If odd count, last one is left unmatched (as you wanted)

    return merges

def apply_merges(B, merges, reducer="or"):
    """
    B: (n, m, k) tensor (CPU or CUDA)
    merges: list of (i, k, j)
    reducer: "or" (clip sum to {0,1}), "sum" (raw sum), or "max"
    Returns: (B_merged, kept_indices)
      - B_merged: tensor with merged rows; second rows in pairs are removed
      - kept_indices: 1D LongTensor mapping new rows back to old indices
    """
    device = B.device
    n, m, k = B.shape
    keep = torch.ones(n, dtype=torch.bool, device=device)

    for (i, k_idx, j) in merges:
        i = int(i); k_idx = int(k_idx); j = int(j)
        if reducer == "or":
            B[i, j] = torch.clamp(B[i, j] + B[k_idx, j], min=0, max=1)
        elif reducer == "sum":
            B[i, j] = B[i, j] + B[k_idx, j]
        elif reducer == "max":
            B[i, j] = torch.maximum(B[i, j], B[k_idx, j])
        else:
            raise ValueError("reducer must be 'or', 'sum', or 'max'")
        keep[k_idx] = False  # drop the second row of the pair

    kept_indices = torch.nonzero(keep, as_tuple=False).flatten()
    B_new = B[keep]
    return B_new, kept_indices

def sample_new_comp_st_to_test(probs, refs_mat, B=1_024, max_iters=1_000):

    device = probs.device
    n_comp, n_state = probs.shape
    #n_var = n_comp + 1  # including system event <- OUTDATED: system row is now excluded from input
    n_var = n_comp

    if len(refs_mat) == 0:
        all_samples = torch.ones((1, n_var, n_state), dtype=torch.int32, device=device)
        return all_samples[0], all_samples

    all_samples = torch.empty((0, n_var, n_state), dtype=torch.int32, device=device)

    for iter in range(max_iters):

        # Start with all-ones batch
        samples_b = torch.ones((B, n_var, n_state), dtype=torch.int32, device=device)

        # Strategy 1: The same permutation applies within a batch
        refs_ord = np.random.permutation(len(refs_mat))
        # Strategy 2: Sort the refs by their probs
        #refs_probs = get_branch_probs(refs_mat, probs)
        #refs_ord = torch.argsort(refs_probs, descending=True)

        # Sampling starts.
        for r_idx in refs_ord:

            r_mat = refs_mat[r_idx]
            r_mat_c = get_complementary_events_nondisjoint(r_mat)

            # Decide whether to sample: skip samples that already contradicts r_mat (to obtain minimal refs)
            is_sampled = torch.ones((B,), dtype=torch.bool, device=device)
            for rc1 in r_mat_c:
                flag1, flag2 = is_subset(rc1, samples_b)

                is_sampled[flag2] = False

            # Select a r_mat_c
            r_mat_c_probs = get_branch_probs(r_mat_c, probs)
            r_mat_c_probs = r_mat_c_probs / r_mat_c_probs.sum()
            idx = torch.multinomial(r_mat_c_probs, num_samples=B, replacement=True)

            # Update samples if is_sampled == True
            samples_b[is_sampled] = samples_b[is_sampled] * r_mat_c[idx[is_sampled]].squeeze(0)

        # Check if there are events with positive prob
        real_prs = get_branch_probs(samples_b, probs)

        all_samples = torch.cat((all_samples, samples_b), dim=0)

        if (real_prs > 0).any():

            x = torch.randint(0, 2, (1,)).item() # which strategy to select?
            # Strategy 1: pick the ref with the highest probability
            if x == 0:
                s_idx = torch.argmax(real_prs)
            # Strategy 2: pick the lowest probability ref
            else:
                # Replace non-positives with +inf so they don't get picked
                masked = torch.where(real_prs > 0, real_prs, torch.inf)  # (B,1)
                s_idx = torch.argmin(masked)  # scalar index into the flattened tensor
            
            sample = samples_b[s_idx] 
            
            bound_br = get_boundary_branches(sample.unsqueeze(0))
            ## decide whether to check upper or lower bound first
            x = torch.randint(0, 2, (1,)).item()
            #x = 1 # check the upper bound first
            is_b_subset, _ = is_subset(bound_br[x], refs_mat) 
            if not is_b_subset:
                return bound_br[x], all_samples
            else:
                is_a_subset, _ = is_subset(bound_br[1-x], refs_mat) 
                if not is_a_subset:
                    return bound_br[1-x], all_samples
                else:
                    Warning("Both boundary branches are subsets of the existing refs. Something's wrong.")
            
            # Strategy 2: pick the branch with the highest probability
            """samples_b = samples_b[real_prs > 0]
            samples_br = get_boundary_refs(samples_b)
            br_prs = get_branch_probs(samples_br, probs)
            br_idx = torch.argsort(br_prs, descending=True)
            for b_idx in br_idx:
                bound_br = samples_br[b_idx]
                # decide whether to check upper or lower bound first
                is_b_subset, _ = is_subset(bound_br, refs_mat) 
                if not is_b_subset:
                    return bound_br, all_samples"""

        elif iter == max_iters - 1:
            print("Max iterations reached without finding a valid sample.")
            return None, all_samples


def _check_any_subset(samples_flat, not_refs_flat, sample_chunk=10000):
    """
    Check which samples are subsets of at least one ref using matmul.

    sample ⊆ ref iff (sample & ~ref) has no 1s, i.e. sample_flat @ not_rule_flat.T == 0.

    Args:
        samples_flat: (B, D) float tensor (flattened binary samples)
        not_refs_flat: (N_rules, D) float tensor (flattened ~refs)
        sample_chunk: process this many samples at a time to bound memory

    Returns:
        (B,) bool tensor — True if sample is subset of any ref
    """
    B = samples_flat.shape[0]
    device = samples_flat.device
    result = torch.zeros(B, dtype=torch.bool, device=device)

    for start in range(0, B, sample_chunk):
        end = min(start + sample_chunk, B)
        # (chunk, D) @ (D, N_rules) → (chunk, N_rules): count of violations
        violations = samples_flat[start:end] @ not_refs_flat.T
        result[start:end] = (violations == 0).any(dim=1)

    return result


def _ensure_refs_tensor(refs, device):
    """Convert refs to a 3D tensor if given as a list."""
    if isinstance(refs, torch.Tensor):
        return refs.to(device)
    if len(refs) == 0:
        return torch.zeros((0,), device=device)
    return torch.stack([r.to(device) for r in refs])


def classify_samples(samples, upper_refs, lower_refs):
    """
    Classify samples as upper, lower, or unknown using subset checks.

    Uses batched matmul instead of per-ref loop for O(1) GPU ops regardless
    of ref count.

    Args:
        samples: (n_sample, n_var, n_state) sample tensor (binary)
        upper_refs: (n_surv, n_var, n_state) ref tensor or list
        lower_refs: (n_fail, n_var, n_state) ref tensor or list

    Returns:
        counts: dict with keys 'upper', 'lower', 'unknown'
    """
    device = samples.device
    n_sample = samples.shape[0]
    upper_refs = _ensure_refs_tensor(upper_refs, device)
    lower_refs = _ensure_refs_tensor(lower_refs, device)

    samples_flat = samples.reshape(n_sample, -1).to(dtype=torch.float16)

    # Survival check
    upper_mask = torch.zeros(n_sample, dtype=torch.bool, device=device)
    if upper_refs.ndim == 3 and upper_refs.shape[0] > 0:
        not_surv = (~upper_refs.bool()).reshape(upper_refs.shape[0], -1).to(dtype=torch.float16)
        upper_mask = _check_any_subset(samples_flat, not_surv)

    # Lower check (only on non-upper samples)
    lower_mask = torch.zeros(n_sample, dtype=torch.bool, device=device)
    remaining = ~upper_mask
    if lower_refs.ndim == 3 and lower_refs.shape[0] > 0 and remaining.any():
        not_fail = (~lower_refs.bool()).reshape(lower_refs.shape[0], -1).to(dtype=torch.float16)
        fail_sub = _check_any_subset(samples_flat[remaining], not_fail)
        lower_mask[remaining] = fail_sub

    counts = {
        'upper': int(upper_mask.sum().item()),
        'lower': int(lower_mask.sum().item()),
        'unknown': int((~upper_mask & ~lower_mask).sum().item())
    }
    return counts

def _sample_and_classify_on_device(args):
    """
    Sample + classify on a single GPU device. Used by multi-GPU sampling.
    Runs in a thread — GPU ops release the GIL during kernel execution.
    """
    probs_dev, n_sample, refs_upper_dev, refs_lower_dev, with_indices = args
    samples = sample_categorical(probs_dev, n_sample)
    if with_indices:
        res = classify_samples_with_indices(samples, refs_upper_dev, refs_lower_dev, return_masks=True)
    else:
        res = classify_samples(samples, refs_upper_dev, refs_lower_dev)
    return samples, res


def sample_categorical(probs, n_sample):
    """
    Sample binary event tensors from categorical distributions.

    Args:
        probs: (n_var, n_state) - probabilities per state per variable.
        n_sample: Number of samples to draw.

    Returns:
        samples: (n_sample, n_var, n_state) - one-hot encoded state selection.
    """

    device = probs.device

    n_var, n_state = probs.shape

    # Step 1: Cumulative probability
    cum_probs = torch.cumsum(probs, dim=1)  # shape (n_var, n_state)

    # Step 2: Uniform random values for each variable
    rand_vals = torch.rand(n_sample, n_var, device=device)  # shape (n_sample, n_var)

    # Step 3: Use searchsorted to get index of selected state
    # cum_probs: (n_var, n_state) → expand to (n_sample, n_var, n_state)
    cum_probs_exp = cum_probs.unsqueeze(0).expand(n_sample, -1, -1)  # (n_sample, n_var, n_state)
    rand_vals_exp = rand_vals.unsqueeze(2)  # (n_sample, n_var, 1)

    # state_indices: (n_sample, n_var)
    state_indices = torch.sum(rand_vals_exp > cum_probs_exp, dim=2)

    # Step 4: One-hot encode
    samples = torch.nn.functional.one_hot(state_indices, num_classes=n_state).int()  # (n_sample, n_var, n_state)

    return samples

def mask_from_first_one(
    x: torch.Tensor,
    mode: str = "after"
) -> torch.Tensor:
    """
    Create masks relative to the first 1 in each row.

    Args:
        x: (n_row, n_col) or (batch, n_row, n_col) int/bool tensor with 0/1 entries
        mode:
            - "after"  → ones from first 1 (inclusive) to end
            - "before" → ones from start up to first 1 (inclusive)
    Returns:
        Tensor of same shape as x, dtype=int32, device preserved.
    """
    assert x.ndim in (2, 3), "x must be 2D or 3D"
    device = x.device

    # Normalize to 3D: (B, N, M)
    squeeze_back = (x.ndim == 2)
    if squeeze_back:
        x3 = x.unsqueeze(0)
    else:
        x3 = x

    B, N, M = x3.shape

    # Column indices for broadcasting comparisons
    cols = torch.arange(M, device=device).view(1, 1, M).expand(B, N, M)

    # First index of "1" per row
    x_bool = (x3 == 1) if x3.dtype != torch.bool else x3
    has_one = x_bool.any(dim=2)                 # (B, N)
    first_idx = x_bool.int().argmax(dim=2)      # (B, N); 0 if none
    first_idx = torch.where(has_one, first_idx, torch.full_like(first_idx, M))

    if mode == "after":
        mask = cols >= first_idx.unsqueeze(-1)  # (B, N, M)
    elif mode == "before":
        mask = cols <= first_idx.unsqueeze(-1)  # (B, N, M)
    else:
        raise ValueError("mode must be 'after' or 'before'")

    mask = mask.to(torch.int32)

    return mask.squeeze(0) if squeeze_back else mask

def update_refs(min_comps_st, refs_dict, refs_mat, row_names, verbose=False):
    _, _, n_state = refs_mat.shape
    Rnew = from_ref_dict_to_mat(min_comps_st, row_names, n_state)
    is_Rnew_subset, are_Rset_subset = is_subset(Rnew, refs_mat)

    if is_Rnew_subset:
        if verbose:
            print("WARNING: New ref is a subset of existing refs. No update made.")
        return refs_dict, refs_mat

    refs_mat = refs_mat[~are_Rset_subset,:,:]
    refs_dict = [r for r, keep in zip(refs_dict, ~are_Rset_subset) if keep]

    refs_dict.append(min_comps_st)
    refs_mat = torch.cat((refs_mat, Rnew.unsqueeze(0)), dim=0)
    if verbose:
        print("No. of existing refs removed: ", int(sum(are_Rset_subset)))

    return refs_dict, refs_mat


def update_refs_batch(new_refs_dicts, refs_dict, refs_mat, row_names, verbose=False):
    """
    Batch version of update_refs: process multiple new refs at once.

    Instead of calling is_subset N times (each against a growing refs_mat),
    this does:
      1. Convert all new refs to matrices in one pass
      2. One batched dominance check: new vs existing
      3. One batched dominance check: new vs new (inter-batch)
      4. Filter and append all upper refs at once

    Returns:
        (refs_dict, refs_mat, n_added, n_removed)
    """
    if not new_refs_dicts:
        return refs_dict, refs_mat, 0, 0

    n_existing, n_var, n_state = refs_mat.shape
    device = refs_mat.device

    # Step 1: convert all new refs to matrices
    new_mats = []
    for rd in new_refs_dicts:
        new_mats.append(from_ref_dict_to_mat(rd, row_names, n_state))
    new_batch = torch.stack(new_mats, dim=0)  # (N_new, n_var, n_state)
    n_new = new_batch.shape[0]

    # Step 2: check new vs existing
    # For each new ref, is it dominated by any existing ref?
    # For each existing ref, is it dominated by any new ref?
    # new_batch: (N_new, n_var, n_state), refs_mat: (N_ex, n_var, n_state)
    new_dominated = torch.zeros(n_new, dtype=torch.bool, device=device)
    existing_dominated = torch.zeros(n_existing, dtype=torch.bool, device=device)

    if n_existing > 0 and n_new > 0:
        # Chunk over existing refs to bound memory: (N_new, chunk, n_var, n_state)
        # With N_new=96, chunk=8000, n_var=120, n_state=2: ~180MB per chunk
        chunk_size = max(1, 500_000_000 // (n_new * n_var * n_state * 4))  # ~500MB limit
        for c_start in range(0, n_existing, chunk_size):
            c_end = min(c_start + chunk_size, n_existing)
            ex_chunk = refs_mat[c_start:c_end]  # (chunk, n_var, n_state)
            new_exp = new_batch.unsqueeze(1)      # (N_new, 1, n_var, n_state)
            ex_exp = ex_chunk.unsqueeze(0)        # (1, chunk, n_var, n_state)
            intersect = new_exp & ex_exp          # (N_new, chunk, n_var, n_state)

            # new[i] dominated by existing[j]?
            new_eq = (new_exp == intersect).all(dim=(2, 3))  # (N_new, chunk)
            new_dominated |= new_eq.any(dim=1)

            # existing[j] dominated by new[i]?
            ex_eq = (ex_exp == intersect).all(dim=(2, 3))    # (N_new, chunk)
            existing_dominated[c_start:c_end] |= ex_eq.any(dim=0)

    # Step 3: among upper new refs, check inter-dominance
    upper_new_idx = torch.where(~new_dominated)[0]
    if len(upper_new_idx) > 1:
        upper_batch = new_batch[upper_new_idx]  # (M, n_var, n_state)
        M = upper_batch.shape[0]
        s_exp_i = upper_batch.unsqueeze(1)  # (M, 1, n_var, n_state)
        s_exp_j = upper_batch.unsqueeze(0)  # (1, M, n_var, n_state)
        s_inter = s_exp_i & s_exp_j        # (M, M, n_var, n_state)
        # i is subset of j: s_exp_i == s_inter
        i_sub_j = (s_exp_i == s_inter).all(dim=(2, 3))  # (M, M)
        j_sub_i = (s_exp_j == s_inter).all(dim=(2, 3))  # (M, M)
        # Mask diagonal
        i_sub_j.fill_diagonal_(False)
        j_sub_i.fill_diagonal_(False)
        # Strict dominance: j dominates i (i⊂j but j⊄i)
        strict = i_sub_j & ~j_sub_i
        # Equal refs (i⊂j AND j⊂i): tiebreak by index — only j<i can dominate i
        equal = i_sub_j & j_sub_i
        lower_mask = torch.tril(torch.ones(M, M, dtype=torch.bool, device=device), diagonal=-1)
        # Ref i is dominated if strictly dominated by any j, or equal to some j<i
        inter_dominated = strict.any(dim=1) | (equal & lower_mask).any(dim=1)  # (M,)
        # Map back: mark dominated ones
        dominated_in_upper = upper_new_idx[inter_dominated]
        new_dominated[dominated_in_upper] = True

    # Step 4: filter existing, append upper new
    keep_existing = ~existing_dominated
    keep_new = ~new_dominated

    n_removed = int(existing_dominated.sum().item())
    n_added = int(keep_new.sum().item())

    refs_mat = torch.cat([
        refs_mat[keep_existing],
        new_batch[keep_new],
    ], dim=0)

    refs_dict = [r for r, k in zip(refs_dict, keep_existing.tolist()) if k]
    for i, rd in enumerate(new_refs_dicts):
        if keep_new[i]:
            refs_dict.append(rd)

    if verbose:
        print(f"Batch update: {n_added} refs added, {n_removed} existing refs removed "
              f"({n_new - n_added} new refs dominated)")

    return refs_dict, refs_mat, n_added, n_removed

def mixed_sort_key(x):
    if x is None:
        return (2, 0, 0.0, "")
    is_numeric = (
        isinstance(x, (int, float, Decimal)) and not isinstance(x, bool)
    ) or isinstance(x, _NUMPY_NUM)
    if is_numeric:
        v = float(x)
        if math.isnan(v):
            return (0, 1, 0.0, "")
        return (0, 0, v, "")
    if isinstance(x, str):
        return (1, 0, 0.0, x.lower())
    return (1, 0, 0.0, str(x).lower())

def classify_samples_with_indices(
    samples: torch.Tensor,
    upper_refs: List[torch.Tensor],
    lower_refs: List[torch.Tensor],
    *,
    return_masks: bool = False
) -> Dict[str, Any]:
    """
    Classify samples as upper, lower, or unknown using subset checks,
    and return indices for each class.

    Args:
        samples: (n_sample, n_var, n_state) binary tensor
        upper_refs: list of ref tensors, each (n_var, n_state) or (n_var+1, n_state)
        lower_refs: list of ref tensors, each (n_var, n_state) or (n_var+1, n_state)
        return_masks: if True, also return boolean masks per class

    Returns:
        {
          'upper': int,
          'lower' : int,
          'unknown' : int,
          'idx_upper': LongTensor[ns],
          'idx_lower' : LongTensor[nf],
          'idx_unknown' : LongTensor[nu],
          # optionally:
          'mask_upper': BoolTensor[n_sample],
          'mask_lower' : BoolTensor[n_sample],
          'mask_unknown' : BoolTensor[n_sample],
        }
    """
    device = samples.device
    n_sample = samples.shape[0]
    upper_refs = _ensure_refs_tensor(upper_refs, device)
    lower_refs = _ensure_refs_tensor(lower_refs, device)

    samples_flat = samples.reshape(n_sample, -1).to(dtype=torch.float16)

    # Survival check
    upper_mask = torch.zeros(n_sample, dtype=torch.bool, device=device)
    if upper_refs.ndim == 3 and upper_refs.shape[0] > 0:
        not_surv = (~upper_refs.bool()).reshape(
            upper_refs.shape[0], -1).to(dtype=torch.float16)
        upper_mask = _check_any_subset(samples_flat, not_surv)

    # Lower check (only on non-upper samples)
    lower_mask = torch.zeros(n_sample, dtype=torch.bool, device=device)
    remaining = ~upper_mask
    if lower_refs.ndim == 3 and lower_refs.shape[0] > 0 and remaining.any():
        not_fail = (~lower_refs.bool()).reshape(
            lower_refs.shape[0], -1).to(dtype=torch.float16)
        fail_sub = _check_any_subset(samples_flat[remaining], not_fail)
        lower_mask[remaining] = fail_sub

    unknown_mask = ~upper_mask & ~lower_mask

    # Indices
    idx_upper = torch.where(upper_mask)[0]
    idx_lower  = torch.where(lower_mask)[0]
    idx_unknown  = torch.where(unknown_mask)[0]

    result: Dict[str, Any] = {
        'upper': int(upper_mask.sum().item()),
        'lower' : int(lower_mask.sum().item()),
        'unknown' : int(unknown_mask.sum().item()),
        'idx_upper': idx_upper,
        'idx_lower' : idx_lower,
        'idx_unknown' : idx_unknown,
    }

    if return_masks:
        result['mask_upper'] = upper_mask
        result['mask_lower']  = lower_mask
        result['mask_unknown']  = unknown_mask

    return result

def get_comp_cond_sys_prob(
    refs_mat_upper: Tensor,
    refs_mat_lower: Tensor,
    probs: Tensor,
    comps_st_cond: Dict[str, int],
    row_names: Sequence[str],
    s_fun: callable = None,                          # Callable[[Dict[str,int]], tuple]
    sys_upper_st: int = 1,        # system state value indicating upper reference
    n_sample: int = 1_000_000,
    n_batch:  int = 1_000_000
) -> Dict[str, float]:
    """
    P(system state | given component states).

    - 'probs' is (n_var, n_state) categorical; we condition rows listed in comps_st_cond to one-hot.
    - We classify samples using refs; for unknowns we call s_fun(comps_dict) to resolve.
    - Returns probabilities over {'upper','lower'} that sum ~ 1.0.

    """
    # --- clone probs and apply conditioning ---
    if torch.is_tensor(probs):
        probs_cond = probs.clone()
        n_comps, n_states = probs_cond.shape
    else:
        raise TypeError("Expected 'probs' to be a torch.Tensor of shape (n_var, n_state).")

    if len(row_names) != n_comps:
        raise ValueError(f"row_names length ({len(row_names)}) must match probs rows ({n_comps}).")

    for x, s in comps_st_cond.items():
        try:
            row_idx = row_names.index(x)
        except ValueError:
            raise ValueError(f"Component {x} not found in row_names.")
        if not (0 <= int(s) < n_states):
            raise ValueError(f"State {s} for component {x} is out of bounds [0,{n_states-1}].")
        probs_cond[row_idx].zero_()
        probs_cond[row_idx, int(s)] = 1.0

    # --- sampling loop (exactly n_sample draws) ---
    batch_size = max(1, min(int(n_batch), int(n_sample)))
    remaining = int(n_sample)

    counts = {"upper": 0, "lower": 0, "unknown": 0}

    while remaining > 0:
        b = min(batch_size, remaining)
        # IMPORTANT: sample from the *conditioned* probs
        samples = sample_categorical(probs_cond, b)  # (b, n_var, n_state) one-hot

        res = classify_samples_with_indices(
            samples, refs_mat_upper, refs_mat_lower, return_masks=True
        )

        counts["upper"] += int(res["upper"])
        counts["lower"]  += int(res["lower"])

        idx_unknown = res["idx_unknown"]
        if idx_unknown.numel() > 0:
            if s_fun is not None:
                # Resolve unknowns with s_fun
                for j in idx_unknown.tolist():
                    sample_j = samples[j]  # (n_var, n_state)
                    # convert one-hot row -> state index per var
                    states = torch.argmax(sample_j, dim=1).tolist()

                    # build comps dict for s_fun
                    comps = {row_names[k]: int(states[k]) for k in range(n_comps)}

                    _, sys_st, _ = s_fun(comps)

                    if sys_st >= sys_upper_st:
                        counts["upper"] += 1
                    else:
                        counts["lower"] += 1

            else:
                counts["unknown"] += int(idx_unknown.shape[0])

        remaining -= b

    # --- normalize to probabilities (denominator = requested n_sample) ---
    total = float(n_sample)
    cond_probs = {k: counts[k] / total for k in counts}
    return cond_probs

def get_comp_cond_sys_prob_multi(
    refs_dict_upper: Dict[int, Tensor],
    refs_dict_lower: Dict[int, Tensor],
    probs: Tensor,
    comps_st_cond: Dict[str, int],
    row_names: Sequence[str],
    s_fun: callable = None,                          # Callable[[Dict[str,int]], tuple]
    n_sample: int = 1_000_000,
    n_batch:  int = 1_000_000
) -> Dict[str, float]:
    """
    Estimate P(system state = s | given component states) for multi-state systems by Monte Carlo.

    Args:
        refs_dict_upper: dict of system upper reference state tensors {state: Tensor(n_var, n_state)}.
        refs_dict_lower: dict of system lower reference state tensors {state: Tensor(n_var, n_state)}.
        probs: (n_var, n_state) categorical probability tensor.
        comps_st_cond: dict of known component states {name: state_index}.
        row_names: list of variable (component) names matching probs rows.
        s_fun: function(comps_dict) -> tuple(_, sys_state, _).
        n_sample, n_batch: number of samples total and per batch.

    Returns:
        Dictionary {state: probability}, summing to 1.0.
    """
    # --- clone probs and apply conditioning ---
    if torch.is_tensor(probs):
        probs_cond = probs.clone()
        n_comps, n_states = probs_cond.shape
    else:
        raise TypeError("Expected 'probs' to be a torch.Tensor of shape (n_var, n_state).")

    if len(row_names) != n_comps:
        raise ValueError(f"row_names length ({len(row_names)}) must match probs rows ({n_comps}).")

    # Applying conditioning
    for x, s in comps_st_cond.items():
        try:
            row_idx = row_names.index(x)
        except ValueError:
            raise ValueError(f"Component {x} not found in row_names.")
        if not (0 <= int(s) < n_states):
            raise ValueError(f"State {s} for component {x} is out of bounds [0,{n_states-1}].")
        probs_cond[row_idx].zero_()
        probs_cond[row_idx, int(s)] = 1.0

    # Validate ref keys
    keys_surv = set(refs_dict_upper.keys())
    keys_fail = set(refs_dict_lower.keys())
    if keys_surv != keys_fail:
        raise ValueError("Upper and lower ref dictionaries must have identical keys.")
    sys_st_list = sorted(keys_surv)
    max_st = max(sys_st_list)
    if sys_st_list != list(range(1, max_st + 1)):
        raise ValueError("Ref dictionary keys must be consecutive integers starting at 1.")

    # --- sampling loop (exactly n_sample draws) ---
    batch_size = max(1, min(int(n_batch), int(n_sample)))
    remaining = int(n_sample)
    counts = {s: 0 for s in [0] + sys_st_list}
    device = probs.device

    while remaining > 0:
        b = min(batch_size, remaining)
        samples = sample_categorical(probs_cond, b)  # (b, n_var, n_state) one-hot
        active = torch.ones(b, dtype=torch.bool, device=device)

        upper_prev = torch.ones(b, dtype=torch.bool, device=device) # upper indices in the previous rounds
        for s in range(1, max_st + 1):

            _res = classify_samples_with_indices(
                samples[active], refs_dict_upper[s], refs_dict_lower[s], return_masks=True
            )

            # back to original indices
            active_idx = torch.where(active)[0]  # positions in the original batch
            # subset masks from the classifier (length == active.sum())
            mask_surv_sub = _res["mask_upper"]
            mask_fail_sub = _res["mask_lower"]
            mask_unk_sub  = _res["mask_unknown"]

            # create full-size masks (length == b) and place subset masks at active positions
            mask_upper_full = torch.zeros(b, dtype=torch.bool, device=device)
            mask_fail_full = torch.zeros(b, dtype=torch.bool, device=device)
            mask_unk_full  = torch.zeros(b, dtype=torch.bool, device=device)

            mask_upper_full[active_idx] = mask_surv_sub
            mask_fail_full[active_idx] = mask_fail_sub
            mask_unk_full[active_idx]  = mask_unk_sub

            # Samples for sys = s-1
            _samp_s_1 = mask_fail_full & upper_prev
            counts[s-1] += int(_samp_s_1.sum().item())

            # update trackers
            active   = active & ~_samp_s_1  # remove finalized ones
            upper_prev = mask_upper_full # upper matches roll to next level
        # Last state
        counts[s] += int(upper_prev.sum().item())
        active = active & ~upper_prev
        active_idx = torch.where(active)[0]  # positions in the original batch

        # Resolve unknowns with s_fun
        if active_idx.numel() > 0:

            for j in active_idx.tolist():
                sample_j = samples[j]  # (n_var, n_state)
                # convert one-hot row -> state index per var
                states = torch.argmax(sample_j, dim=1).tolist()

                # build comps dict for s_fun
                comps = {row_names[k]: int(states[k]) for k in range(n_comps)}

                _, sys_st, _ = s_fun(comps)
                counts[sys_st] += 1

        remaining -= b

    # --- normalize to probabilities (denominator = requested n_sample) ---
    total = float(n_sample)
    cond_probs = {k: counts[k] / total for k in counts}
    return cond_probs

def run_ref_extraction_by_mcs(
    *,
    sfun,
    probs: torch.Tensor,
    row_names: List[str],
    n_state: int,
    sys_upper_st: int,
    refs_upper: Optional[List[Dict[str, Any]]] = None,
    refs_lower: Optional[List[Dict[str, Any]]] = None,
    refs_mat_upper: Optional[torch.Tensor] = None,
    refs_mat_lower: Optional[torch.Tensor] = None,
    # Termination / threshold settings
    unk_prob_thres: float = 1e-2,
    unk_prob_opt: str = "rel", # "abs" or "rel"
    max_rounds: int = 10000,     # hard cap on rounds to prevent infinite loops
    # Frequencies / sampling settings
    prob_update_every: int = 500,
    save_every: int = 10,
    n_sample: int = 10_000_000,
    sample_batch_size: int = 100_000,
    max_search_loops: int = 0,  # max batches per round for searching unknowns (0 = use n_sample // sample_batch_size)
    min_ref_search: bool = True,
    ref_update_verbose: bool = True,
    # Parallelism
    n_workers: int = 1,  # number of CPU workers for parallel sfun + minimization
    devices: Optional[List[str]] = None,  # list of GPU devices for multi-GPU sampling, e.g. ["cuda:0", "cuda:1"]
    # Output control
    output_dir: str = "rsr_res",
    upper_json_name: str = None,
    lower_json_name: str = None,
    upper_pt_name: str = None,
    lower_pt_name: str = None,
    metrics_path: str = "metrics.json",
) -> Dict[str, Any]:

    os.makedirs(output_dir, exist_ok=True)

    if upper_json_name is None:
        upper_json_name = f"refs_up_{sys_upper_st}.json"
    if lower_json_name is None:
        lower_json_name = f"refs_low_{sys_upper_st-1}.json"
    if upper_pt_name is None:
        upper_pt_name = f"refs_up_{sys_upper_st}.pt"
    if lower_pt_name is None:
        lower_pt_name = f"refs_low_{sys_upper_st-1}.pt"

    # ---- helpers ----
    def _avg_rule_len(ref_store: Any) -> float:
        try:
            if not ref_store:
                return 0.0
            return (sum(len(r) - 1 for r in ref_store)) / len(ref_store)
        except Exception:
            return 0.0

    def _save_json(obj, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=4)

    def _save_pt(t: torch.Tensor, path: str) -> None:
        torch.save(t.detach().cpu(), path)

    # ---- initial state ----
    if refs_upper is None: refs_upper = []
    if refs_lower is None: refs_lower = []

    device = probs.device

    unk_prob = 1.0
    n_round = 0
    metrics_log: List[Dict[str, Any]] = []

    n_vars = len(row_names)
    if refs_mat_upper is None:
        refs_mat_upper = torch.empty((0, n_vars, n_state), dtype=torch.int32, device=device)
    if refs_mat_lower is None:
        refs_mat_lower = torch.empty((0, n_vars, n_state), dtype=torch.int32, device=device)

    sys_val_list: List[Any] = []

    metrics_path = os.path.join(output_dir, metrics_path)
    refs_upper_path = os.path.join(output_dir, upper_json_name)
    refs_lower_path = os.path.join(output_dir, lower_json_name)
    refs_upper_pt_path = os.path.join(output_dir, upper_pt_name)
    refs_lower_pt_path = os.path.join(output_dir, lower_pt_name)

    is_new_cand = True
    last_probs = {"upper": 0.0, "lower": 0.0, "unknown": 1.0}

    # ---- parallel worker pool (fork-based, inherits sfun via global) ----
    global _MP_SFUN, _MP_SYS_UPPER_ST, _MP_N_STATE
    _pool = None
    if n_workers > 1:
        _MP_SFUN = sfun
        _MP_SYS_UPPER_ST = sys_upper_st
        _MP_N_STATE = n_state
        _ctx = mp.get_context('fork')
        _pool = _ctx.Pool(n_workers)
        print(f"Parallel mode: {n_workers} CPU workers for sfun + minimization")

    # ---- multi-GPU setup ----
    _use_multi_gpu = False
    _gpu_devices = []
    _gpu_probs = []       # probs replicated to each device
    _gpu_thread_pool = None
    if devices is not None and len(devices) > 1:
        from concurrent.futures import ThreadPoolExecutor
        _gpu_devices = [torch.device(d) for d in devices]
        _gpu_probs = [probs.to(d) for d in _gpu_devices]
        _gpu_thread_pool = ThreadPoolExecutor(max_workers=len(_gpu_devices))
        _use_multi_gpu = True
        print(f"Multi-GPU mode: sampling across {devices}")

    total_loops = max(n_sample // sample_batch_size, 1)
    # Search loops: capped for finding unknowns; full total_loops used only for probability estimation
    search_loops = min(max_search_loops, total_loops) if max_search_loops > 0 else total_loops

    # ---- main loop ----
    while is_new_cand and (unk_prob > unk_prob_thres if unk_prob_opt == "abs" else unk_prob / (min([last_probs["lower"]+1e-12, last_probs["upper"]+1e-12])) > unk_prob_thres):
        n_round += 1
        t0 = time.perf_counter()

        print("---")
        print(f"Round: {n_round}, Unk. prob.: {unk_prob:.3e}")
        if last_probs['upper'] is not None and last_probs['lower'] is not None:
            print(f"Upper probs: {last_probs['upper']:.3e}, Lower probs: {last_probs['lower']:.3e}")
        print(f"No. of non-dominant refs: {len(refs_mat_upper)+len(refs_mat_lower)}, "
              f"Survival refs: {len(refs_mat_upper)}, Failure refs: {len(refs_mat_lower)}")

        is_new_cand = False
        counts = {"upper": 0, "lower": 0, "unknown": 0}
        res = None
        samples = None
        i = -1
        _t_search = 0.0
        _t_minimize = 0.0
        _t_refs = 0.0
        _t_probs = 0.0

        _ts = time.perf_counter()
        for i in range(search_loops):
            if _use_multi_gpu:
                # Split batch across GPUs, sample + classify in parallel threads
                n_gpus = len(_gpu_devices)
                per_gpu = sample_batch_size // n_gpus
                remainder = sample_batch_size % n_gpus
                tasks = []
                for gi in range(n_gpus):
                    n_gi = per_gpu + (1 if gi < remainder else 0)
                    refs_s_gi = refs_mat_upper.to(_gpu_devices[gi])
                    refs_f_gi = refs_mat_lower.to(_gpu_devices[gi])
                    tasks.append((_gpu_probs[gi], n_gi, refs_s_gi, refs_f_gi, True))

                futures = list(_gpu_thread_pool.map(_sample_and_classify_on_device, tasks))

                # Merge results back to primary device
                all_samples = []
                for samples_gi, res_gi in futures:
                    all_samples.append(samples_gi.to(device))
                    counts["upper"] += int(res_gi["upper"])
                    counts["lower"]  += int(res_gi["lower"])
                    counts["unknown"]  += int(res_gi["unknown"])
                samples = torch.cat(all_samples, dim=0)

                # Re-classify merged batch on primary device for correct indices
                res = classify_samples_with_indices(samples, refs_mat_upper, refs_mat_lower, return_masks=True)
            else:
                samples = sample_categorical(probs, sample_batch_size)  # (B, n_var, n_state)
                res = classify_samples_with_indices(samples, refs_mat_upper, refs_mat_lower, return_masks=True)

                counts["upper"] += int(res["upper"])
                counts["lower"]  += int(res["lower"])
                counts["unknown"]  += int(res["unknown"])

            if res['idx_unknown'].numel() > 0:
                is_new_cand = True
                break

        _t_search = time.perf_counter() - _ts

        # denominator = number of samples actually processed
        n_sample_actual = sample_batch_size * (i + 1)
        samp_probs = {k: v / n_sample_actual for k, v in counts.items()}
        unk_prob = samp_probs["unknown"]
        last_probs.update(samp_probs)

        # If no unknowns found, skip candidate creation and continue to periodic update / exit
        if not is_new_cand:
            probs_updated = False
            # When search is capped, the unk_prob estimate from search_loops is rough;
            # force a full probability update to get an accurate termination check.
            needs_full_estimate = (search_loops < total_loops) or (n_round % prob_update_every) == 0
            if needs_full_estimate:
                # refresh with a full estimate
                loops = max(n_sample // sample_batch_size, 1)
                c2 = {"upper": 0, "lower": 0, "unknown": 0}
                for _ in range(loops):
                    if _use_multi_gpu:
                        n_gpus = len(_gpu_devices)
                        per_gpu = sample_batch_size // n_gpus
                        remainder = sample_batch_size % n_gpus
                        tasks = []
                        for gi in range(n_gpus):
                            n_gi = per_gpu + (1 if gi < remainder else 0)
                            refs_s_gi = refs_mat_upper.to(_gpu_devices[gi])
                            refs_f_gi = refs_mat_lower.to(_gpu_devices[gi])
                            tasks.append((_gpu_probs[gi], n_gi, refs_s_gi, refs_f_gi, False))
                        for _, ci in _gpu_thread_pool.map(_sample_and_classify_on_device, tasks):
                            for k in c2:
                                c2[k] += ci[k]
                    else:
                        s = sample_categorical(probs, sample_batch_size)
                        ci = classify_samples(s, refs_mat_upper, refs_mat_lower)
                        for k in c2:
                            c2[k] += ci[k]
                sp2 = {k: v / (sample_batch_size * loops) for k, v in c2.items()}
                print("---")
                print(f"Probs: 'upper': {sp2['upper']: .3e}, 'lower': {sp2['lower']: .3e}, 'unkn': {sp2['unknown']: .3e}")
                unk_prob = sp2["unknown"]
                last_probs.update(sp2)
                n_sample_actual = sample_batch_size * loops
                probs_updated = True

            # metrics, persist, then break condition handled by while guard
            dt = time.perf_counter() - t0
            rss_gb = psutil.Process().memory_info().rss / (1024**3)
            metrics_log.append({
                "round": n_round,
                "time_sec": dt,
                "t_search": round(_t_search, 3),
                "t_minimize": 0.0,
                "t_refs": 0.0,
                "t_probs": round(dt - _t_search, 3),
                "n_refs_upper": int(len(refs_mat_upper)),
                "n_refs_lower": int(len(refs_mat_lower)),
                "probs_updated": probs_updated,
                "p_upper": last_probs["upper"],
                "p_lower": last_probs["lower"],
                "p_unknown": last_probs["unknown"],
                "n_sample_actual": n_sample_actual,
                "avg_len_upper": _avg_rule_len(refs_upper),
                "avg_len_lower": _avg_rule_len(refs_lower),
                "rss_gb": rss_gb,
            })

            if (n_round % save_every) == 0:
                with open(metrics_path, "a", encoding="utf-8") as mf:
                    for e in metrics_log[-save_every:]:
                        mf.write(json.dumps(e) + "\n")
                _save_json(refs_upper, refs_upper_path)
                _save_json(refs_lower, refs_lower_path)
                _save_pt(refs_mat_upper, refs_upper_pt_path)
                _save_pt(refs_mat_lower, refs_lower_pt_path)

            continue  # go to next while-check (likely exit if unk_prob <= thresh)

        # --- We have unknowns: extract unknown(s) and build ref(s) ---
        idx_unknown = res['idx_unknown']

        _ts = time.perf_counter()
        if _pool is not None and min_ref_search:
            # ---- Parallel: pick up to n_workers unknowns and minimize concurrently ----
            n_pick = min(n_workers, len(idx_unknown))
            perm = torch.randperm(len(idx_unknown))[:n_pick]
            picked_indices = idx_unknown[perm]

            tasks = []
            for idx_i in picked_indices:
                s0 = samples[idx_i.item()]
                sts = torch.argmax(s0, dim=1).tolist()
                cst = {row_names[k]: int(sts[k]) for k in range(n_vars)}
                tasks.append((cst, None))

            results = _pool.map(_minimize_one_unknown, tasks)
            _t_minimize = time.perf_counter() - _ts

            _ts = time.perf_counter()
            # Separate results into upper and lower batches
            new_surv_dicts = []
            new_fail_dicts = []
            for min_comps_st, sys_st, fval in results:
                if sys_st >= sys_upper_st:
                    new_surv_dicts.append(min_comps_st)
                else:
                    new_fail_dicts.append(min_comps_st)

                if isinstance(fval, float):
                    fval = int(round(fval * 1000)) / 1000.0
                if fval not in sys_val_list:
                    sys_val_list.append(fval)
                    sys_val_list.sort(key=mixed_sort_key)

            # Batch update: one dominance check per type instead of N sequential ones
            if new_surv_dicts:
                refs_upper, refs_mat_upper, n_add, n_rem = update_refs_batch(
                    new_surv_dicts, refs_upper, refs_mat_upper, row_names, verbose=ref_update_verbose)
                print(f"Survival: {n_add} refs added, {n_rem} removed (from {len(new_surv_dicts)} candidates)")
            if new_fail_dicts:
                refs_lower, refs_mat_lower, n_add, n_rem = update_refs_batch(
                    new_fail_dicts, refs_lower, refs_mat_lower, row_names, verbose=ref_update_verbose)
                print(f"Failure: {n_add} refs added, {n_rem} removed (from {len(new_fail_dicts)} candidates)")

            if sys_val_list:
                sys_val_list.sort(key=mixed_sort_key)
                print(f"Updated sys_vals: {sys_val_list}")

        else:
            # ---- Serial (original): pick one unknown ----
            rand_idx = idx_unknown[torch.randint(len(idx_unknown), (1,))].item()
            sample0 = samples[rand_idx]  # (n_var, n_state)

            states = torch.argmax(sample0, dim=1).tolist()
            comps_st_test = {row_names[k]: int(states[k]) for k in range(n_vars)}

            fval, sys_st, min_comps_st0 = sfun(comps_st_test)
            if min_comps_st0 is None:
                min_comps_st0 = comps_st_test.copy()
            elif isinstance(next(iter(min_comps_st0.values())), tuple):
                min_comps_st0 = {k: v[1] for k, v in min_comps_st0.items()}

            if sys_st >= sys_upper_st:
                if min_ref_search:
                    min_comps_st, info = minimise_upper_states_random(min_comps_st0, sfun, sys_upper_st=sys_upper_st, fval=fval)
                    fval = info.get('final_sys_state', fval)
                else:
                    min_comps_st = get_min_upper_comps_st(min_comps_st0, sys_upper_st=sys_upper_st)
            else:
                if min_ref_search:
                    min_comps_st, info = minimise_lower_states_random(min_comps_st0, sfun, max_state=n_state-1, sys_lower_st=sys_upper_st-1, fval=fval)
                    fval = info.get('final_sys_state', fval)
                else:
                    min_comps_st = get_min_lower_comps_st(min_comps_st0, max_st=n_state-1, sys_lower_st=sys_upper_st-1)
            _t_minimize = time.perf_counter() - _ts

            _ts = time.perf_counter()
            if sys_st >= sys_upper_st:
                print("Survival sample found from sampling.")
                refs_upper, refs_mat_upper = update_refs(min_comps_st, refs_upper, refs_mat_upper, row_names, verbose=ref_update_verbose)
            else:
                print("Failure sample found from sampling.")
                refs_lower, refs_mat_lower = update_refs(min_comps_st, refs_lower, refs_mat_lower, row_names, verbose=ref_update_verbose)

            print(f"New ref added. System state: {sys_st}, System value: {fval}. Total samples: {n_sample_actual}.")
            print(f"New ref (No. of conditions: {len(min_comps_st)-1}): {min_comps_st}")

            if isinstance(fval, float):
                fval = int(round(fval * 1000)) / 1000.0
            if fval not in sys_val_list:
                sys_val_list.append(fval)
                sys_val_list.sort(key=mixed_sort_key)
                print(f"Updated sys_vals: {sys_val_list}")

        # ---- Periodic probability (bound) test via sampling ----
        if _t_refs == 0.0:
            _t_refs = time.perf_counter() - _ts
        probs_updated = False
        _ts = time.perf_counter()
        if (n_round % prob_update_every) == 0:
            loops = max(n_sample // sample_batch_size, 1)
            c2 = {"upper": 0, "lower": 0, "unknown": 0}
            for _ in range(loops):
                if _use_multi_gpu:
                    n_gpus = len(_gpu_devices)
                    per_gpu = sample_batch_size // n_gpus
                    remainder = sample_batch_size % n_gpus
                    tasks = []
                    for gi in range(n_gpus):
                        n_gi = per_gpu + (1 if gi < remainder else 0)
                        refs_s_gi = refs_mat_upper.to(_gpu_devices[gi])
                        refs_f_gi = refs_mat_lower.to(_gpu_devices[gi])
                        tasks.append((_gpu_probs[gi], n_gi, refs_s_gi, refs_f_gi, False))
                    for _, ci in _gpu_thread_pool.map(_sample_and_classify_on_device, tasks):
                        for k in c2:
                            c2[k] += ci[k]
                else:
                    s = sample_categorical(probs, sample_batch_size)
                    ci = classify_samples(s, refs_mat_upper, refs_mat_lower)
                    for k in c2:
                        c2[k] += ci[k]
            sp2 = {k: v / (sample_batch_size * loops) for k, v in c2.items()}
            print("---")
            print(f"Probs: 'upper': {sp2['upper']: .3e}, 'lower': {sp2['lower']: .3e}, 'unkn': {sp2['unknown']: .3e}")
            unk_prob = sp2["unknown"]
            last_probs.update(sp2)
            n_sample_actual = sample_batch_size * loops
            probs_updated = True

        # ---- metrics for this round ----
        _t_probs = time.perf_counter() - _ts
        rss_gb = psutil.Process().memory_info().rss / (1024**3)
        dt = time.perf_counter() - t0
        metrics_log.append({
            "round": n_round,
            "time_sec": dt,
            "t_search": round(_t_search, 3),
            "t_minimize": round(_t_minimize, 3),
            "t_refs": round(_t_refs, 3),
            "t_probs": round(_t_probs, 3),
            "n_refs_upper": int(len(refs_mat_upper)),
            "n_refs_lower": int(len(refs_mat_lower)),
            "probs_updated": probs_updated,
            "p_upper": last_probs["upper"],
            "p_lower": last_probs["lower"],
            "p_unknown": last_probs["unknown"],
            "n_sample_actual": n_sample_actual,
            "avg_len_upper": _avg_rule_len(refs_upper),
            "avg_len_lower": _avg_rule_len(refs_lower),
            "rss_gb": rss_gb,
        })

        if (n_round % save_every) == 0:
            with open(metrics_path, "a", encoding="utf-8") as mf:
                for e in metrics_log[-save_every:]:
                    mf.write(json.dumps(e) + "\n")
            _save_json(refs_upper, refs_upper_path)
            _save_json(refs_lower, refs_lower_path)
            _save_pt(refs_mat_upper, refs_upper_pt_path)
            _save_pt(refs_mat_lower, refs_lower_pt_path)

        if n_round >= max_rounds:
            print(f"Reached maximum rounds ({max_rounds}). Terminating.")
            break

    # Final flush of any remaining metrics not yet written by save_every
    last_flushed_rounds = (n_round // save_every) * save_every
    if last_flushed_rounds < n_round and metrics_log:
        with open(metrics_path, "a", encoding="utf-8") as mf:
            for e in metrics_log[last_flushed_rounds:]:
                mf.write(json.dumps(e) + "\n")

    # Final snapshot of refs
    _save_json(refs_upper, refs_upper_path)
    _save_json(refs_lower, refs_lower_path)
    _save_pt(refs_mat_upper, refs_upper_pt_path)
    _save_pt(refs_mat_lower, refs_lower_pt_path)

    # Final probability check
    loops = max(n_sample // sample_batch_size, 1)
    c2 = {"upper": 0, "lower": 0, "unknown": 0}
    for _ in range(loops):
        if _use_multi_gpu:
            n_gpus = len(_gpu_devices)
            per_gpu = sample_batch_size // n_gpus
            remainder = sample_batch_size % n_gpus
            tasks = []
            for gi in range(n_gpus):
                n_gi = per_gpu + (1 if gi < remainder else 0)
                refs_s_gi = refs_mat_upper.to(_gpu_devices[gi])
                refs_f_gi = refs_mat_lower.to(_gpu_devices[gi])
                tasks.append((_gpu_probs[gi], n_gi, refs_s_gi, refs_f_gi, False))
            for _, ci in _gpu_thread_pool.map(_sample_and_classify_on_device, tasks):
                for k in c2:
                    c2[k] += ci[k]
        else:
            s = sample_categorical(probs, sample_batch_size)
            ci = classify_samples(s, refs_mat_upper, refs_mat_lower)
            for k in c2:
                c2[k] += ci[k]
    sp2 = {k: v / (sample_batch_size * loops) for k, v in c2.items()}
    print("---")
    print(f"[Final results] Probs: 'upper': {sp2['upper']: .3e}, 'lower': {sp2['lower']: .3e}, 'unkn': {sp2['unknown']: .3e}")

    # Final metrics entry
    rss_gb = psutil.Process().memory_info().rss / (1024**3)
    metrics_log.append({
        "round": n_round,
        "time_sec": 0.0,
        "n_refs_upper": int(len(refs_mat_upper)),
        "n_refs_lower": int(len(refs_mat_lower)),
        "probs_updated": True,
        "p_upper": sp2["upper"],
        "p_lower": sp2["lower"],
        "p_unknown": sp2["unknown"],
        "avg_len_upper": _avg_rule_len(refs_upper),
        "avg_len_lower": _avg_rule_len(refs_lower),
        "rss_gb": rss_gb,   
    })

    # ---- clean up worker pools ----
    if _pool is not None:
        _pool.close()
        _pool.join()
    if _gpu_thread_pool is not None:
        _gpu_thread_pool.shutdown(wait=False)

    return {
        "sys_vals": sorted(sys_val_list, key=mixed_sort_key),
        "metrics_path": metrics_path,
        "refs_upper_path": refs_upper_path,
        "refs_lower_path": refs_lower_path,
        "refs_upper_pt_path": refs_upper_pt_path,
        "refs_lower_pt_path": refs_lower_pt_path,
        "metrics_log": metrics_log,
    }

