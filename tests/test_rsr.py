from pathlib import Path
import pytest
import torch
import pdb


from rsr import rsr
#from rsr.utils import sys_fun_sum
from rsr import utils


HOME = Path(__file__).absolute().parent

def test_get_min_lower_comps_st1():

    comps_st = {'x1': 2, 'x2': 1, 'x3': 0, 'x4': 0} # Example state that leads to system lower reference state

    min_comps_st = rsr.get_min_lower_comps_st(comps_st, 2, 0)

    assert min_comps_st == {'x2': ('<=', 1), 'x3': ('<=', 0), 'x4': ('<=', 0), 'sys': ('<=', 0)}, f"Expected {{'x2': ('<=', 1), 'x3': ('<=', 0), 'x4': ('<=', 0), 'sys': ('<=', 0)}}, got {min_comps_st}"

def test_get_min_lower_comps_st2():

    comps_st = {'x1': 2, 'x2': 0, 'x3': 2, 'x4': 2} # Example state that leads to system lower reference state

    min_comps_st = rsr.get_min_lower_comps_st(comps_st, 2, 0)

    assert min_comps_st == {'x2': ('<=', 0), 'sys': ('<=', 0)}, f"Expected {{'x2': ('<=', 0), 'sys': ('<=', 0)}}, got {min_comps_st}"

def test_get_min_lower_comps_st3():

    comps_st = {'x1': 2, 'x2': 1, 'x3': 1, 'x4': 3} # Example state that leads to system lower reference state

    min_comps_st = rsr.get_min_lower_comps_st(comps_st, 3, 1)

    assert min_comps_st == {'x1': ('<=', 2), 'x2': ('<=', 1), 'x3': ('<=', 1), 'sys': ('<=', 1)}, f"Expected {{'x1': ('<=', 2), 'x2': ('<=', 1), 'x3': ('<=', 1), 'sys': ('<=', 1)}}, got {min_comps_st}"

def test_get_min_upper_comps_st3():

    comps_st = {'x1': 2, 'x2': 1, 'x3': 1, 'x4': 3} # Example state that leads to system upper reference state

    min_comps_st = rsr.get_min_upper_comps_st(comps_st, 1)

    assert min_comps_st == {'x1': ('>=', 2), 'x2': ('>=', 1), 'x3': ('>=', 1), 'x4': ('>=', 3),'sys': ('>=', 1)}, f"Expected {{'x1': ('>=', 2), 'x2': ('>=', 1), 'x3': ('>=', 1), 'x4': ('>=', 3),'sys': ('>=', 1)}}, got {min_comps_st}"

def test_get_min_upper_comps_st2():

    comps_st = {'x1': 2, 'x2': 0, 'x3': 2, 'x4': 2} # Example state that leads to system upper reference state

    min_comps_st = rsr.get_min_upper_comps_st(comps_st, 1)

    assert min_comps_st == {'x1': ('>=', 2),'x3': ('>=', 2), 'x4': ('>=', 2), 'sys': ('>=', 1)}, f"Expected {{'x1': ('>=', 2),'x3': ('>=', 2),'x4': ('>=', 2), 'sys': ('>=', 1)}}, got {min_comps_st}"


def test_from_rule_dict_to_mat1():
    rule = {'x1': ('>=', 2), 'x2': ('>=', 2), 'sys': ('>=', 1)}
    col_names = ['x1', 'x2', 'x3', 'x4']
    max_st = 3

    rule_mat = rsr.from_rule_dict_to_mat(rule, col_names, max_st)

    assert torch.equal(rule_mat, torch.tensor([[0, 0, 1],
                                               [0, 0, 1],
                                               [1, 1, 1],
                                               [1, 1, 1]], device=rule_mat.device))

def test_from_rule_dict_to_mat2():
    rule = {'x1': ('>=', 2), 'x2': ('>=', 2), 'x3': ('>=', 2), 'x4': ('>=', 1), 'sys': ('>=', 1)}
    col_names = ['x1', 'x2', 'x3', 'x4']
    max_st = 3

    rule_mat = rsr.from_rule_dict_to_mat(rule, col_names, max_st)

    assert torch.equal(rule_mat, torch.tensor([[0, 0, 1],
                                               [0, 0, 1],
                                               [0, 0, 1],
                                               [0, 1, 1]], device=rule_mat.device))

def test_from_rule_dict_to_mat3():
    rule = {'x2': ('<=', 1), 'x3': ('<', 1), 'x4': ('<=', 0), 'sys': ('<=', 0)}
    col_names = ['x1', 'x2', 'x3', 'x4']
    max_st = 4

    rule_mat = rsr.from_rule_dict_to_mat(rule, col_names, max_st)

    assert torch.equal(rule_mat, torch.tensor([[1, 1, 1, 1],
                                               [1, 1, 0, 0],
                                               [1, 0, 0, 0],
                                               [1, 0, 0, 0]], device=rule_mat.device))

def test_get_branches_cap_branches1():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    B1 = torch.tensor([
        [[1,1,1],
         [1,1,1],
         [1,1,1],
         [1,1,0]]
    ], dtype=torch.int32, device=device)
    utils.print_tensor(B1)

    B2 = torch.tensor([
        [[1,1,1],
         [1,1,1],
         [1,0,0],
         [1,1,0]],
        [[1,1,1],
         [1,1,1],
         [0,1,1],
         [1,0,0]]
    ], dtype=torch.int32, device=device)
    utils.print_tensor(B2)

    Bnew = rsr.get_branches_cap_branches(B1, B2)
    utils.print_tensor(Bnew)

    expected = torch.tensor([
        [[1,1,1],
         [1,1,1],
         [1,0,0],
         [1,1,0]],
        [[1,1,1],
         [1,1,1],
         [0,1,1],
         [1,0,0]]
    ], dtype=torch.int32, device=device)

    Bnew = rsr.get_branches_cap_branches(B1, B2)
    utils.print_tensor(Bnew)
    assert torch.equal(Bnew, expected)


def test_get_branches_cap_branches2():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    B1 = torch.tensor([
        [[1,1,1],
         [1,1,1],
         [1,0,0],
         [1,1,0]],
        [[1,1,1],
         [1,1,1],
         [0,1,1],
         [1,0,0]
    ]], dtype=torch.int32, device=device)
    utils.print_tensor(B1)

    B2 = torch.tensor([
        [[1,1,1],
         [1,0,0],
         [1,1,1],
         [0,1,1]],
        [[1,1,1],
         [0,1,1],
         [1,1,1],
         [1,0,0]]
    ], dtype=torch.int32, device=device)

    Bnew = rsr.get_branches_cap_branches(B1, B2)
    expected = torch.tensor([
        [[1,1,1],
         [1,0,0],
         [1,0,0],
         [0,1,0]],
        [[1,1,1],
         [0,1,1],
         [1,0,0],
         [1,0,0]],
        [[1,1,1],
         [0,1,1],
         [0,1,1],
         [1,0,0]]
    ], dtype=torch.int32, device=device)

    utils.print_tensor(Bnew)

    assert torch.equal(Bnew, expected)

def test_get_branches_cap_branches3():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    B1 = torch.tensor([
        [[1,1,1],
         [1,0,0],
         [1,0,0],
         [0,1,0]],
        [[1,1,1],
         [0,1,1],
         [1,0,0],
         [1,1,0]],
        [[1,1,1],
         [0,1,1],
         [0,1,1],
         [1,0,0]]
    ], dtype=torch.int32, device=device)

    B2 = torch.tensor([
        [[1,0,0],
         [1,1,1],
         [1,1,1],
         [0,1,1]],
        [[0,1,1],
         [1,1,1],
         [1,1,1],
         [1,0,0]]
    ], dtype=torch.int32, device=device)

    Bnew = rsr.get_branches_cap_branches(B1, B2)
    expected = torch.tensor([
        [[1,0,0],
         [1,0,0],
         [1,0,0],
         [0,1,0]],
        [[1,0,0],
         [0,1,1],
         [1,0,0],
         [0,1,0]],
        [[0,1,1],
         [0,1,1],
         [1,0,0],
         [1,0,0]],
        [[0,1,1],
         [0,1,1],
         [0,1,1],
         [1,0,0]]
    ], dtype=torch.int32, device=device)

    assert torch.equal(Bnew, expected)

def test_get_complementary_events1():
    R = torch.tensor([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [0, 0, 1],
    ], dtype=torch.int32)

    Bnew = rsr.get_complementary_events(R)

    expected = torch.tensor([
        [[1,1,1],
         [1,1,1],
         [1,1,1],
         [1,1,0]]
    ], dtype=torch.int32)

    utils.print_tensor(Bnew)

    assert torch.equal(Bnew, expected)

def test_get_complementary_events2():
    R = torch.tensor([
        [1,1,1],
        [1,1,1],
        [0,1,1],
        [0,1,1],
    ], dtype=torch.int32)

    Bnew = rsr.get_complementary_events(R)

    expected = torch.tensor([
        [[1,1,1],
         [1,1,1],
         [1,0,0],
         [1,1,1]],
        [[1,1,1],
         [1,1,1],
         [0,1,1],
         [1,0,0]],
    ], dtype=torch.int32)

    assert torch.equal(Bnew, expected)

def test_get_complementary_events3():
    R = torch.tensor([
        [1,1,1],
        [0,1,1],
        [1,1,1],
        [0,1,1],
    ], dtype=torch.int32)

    Bnew = rsr.get_complementary_events(R)

    expected = torch.tensor([
        [[1,1,1],
         [1,0,0],
         [1,1,1],
         [1,1,1]],
        [[1,1,1],
         [0,1,1],
         [1,1,1],
         [1,0,0]],
    ], dtype=torch.int32)

    assert torch.equal(Bnew, expected)

def test_get_branch_probs1():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B1 = torch.tensor([
        [[1,1,1],
         [1,0,0],
         [1,0,0],
         [0,1,0]],
        [[1,1,1],
         [0,1,1],
         [1,0,0],
         [1,1,0]],
        [[1,1,1],
         [0,1,1],
         [0,1,1],
         [1,0,0]]
    ], dtype=torch.int32, device=device)

    prob1 = torch.tensor([[0.3, 0.7, 0.0],
             [0.1, 0.2, 0.7],
             [0.2, 0.8, 0.0],
             [0.1, 0.3, 0.6]], dtype=torch.float32, device=device)

    Bprob = rsr.get_branch_probs(B1, prob1)
    print(Bprob)
    expected = torch.tensor([0.006, 0.072, 0.072], dtype=torch.float32, device=device)

    assert torch.allclose(Bprob, expected, atol=1e-5)

def test_get_branch_probs2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B1 = torch.tensor([
        [[1,1,1],
         [1,0,0],
         [1,0,0],
         [0,1,0]],
        [[1,1,1],
         [0,1,1],
         [1,0,0],
         [1,1,0]],
        [[1,1,1],
         [0,1,1],
         [0,1,1],
         [1,0,0]]
    ], dtype=torch.int32, device=device)

    prob1 = torch.tensor([[0.1, 0.9, 0.0],
             [0.3, 0.2, 0.5],
             [0.5, 0.5, 0.0],
             [0.5, 0.1, 0.4]], dtype=torch.float32, device=device)

    Bprob = rsr.get_branch_probs(B1, prob1)
    expected = torch.tensor([0.015, 0.21, 0.175], dtype=torch.float32, device=device)

    assert torch.allclose(Bprob, expected, atol=1e-5)

def test_get_branch_probs3():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B1 = torch.tensor([
        [[1,1,1],
         [1,1,1],
         [1,0,0],
         [1,1,0]],
        [[1,1,1],
         [1,1,1],
         [0,1,1],
         [1,0,0]]
    ], dtype=torch.int32, device=device)

    prob1 = torch.tensor([[0.3, 0.7, 0.0],
             [0.1, 0.2, 0.7],
             [0.2, 0.8, 0.0],
             [0.1, 0.3, 0.6]], dtype=torch.float32, device=device)

    Bprob = rsr.get_branch_probs(B1, prob1)
    expected = torch.tensor([0.08, 0.08], dtype=torch.float32, device=device)

    assert torch.allclose(Bprob, expected, atol=1e-5)

def test_get_boundary_branches1():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    B1 = torch.tensor([
        [[1,1,1],
         [1,0,0],
         [1,0,0],
         [0,1,0]],
        [[1,1,1],
         [0,1,1],
         [1,0,0],
         [1,1,0]],
        [[1,1,1],
         [0,1,1],
         [0,1,1],
         [1,0,0]]
    ], dtype=torch.int32, device=device)

    expected = torch.tensor([
        [[0,0,1],
         [1,0,0],
         [1,0,0],
         [0,1,0]],
        [[0,0,1],
         [0,0,1],
         [1,0,0],
         [0,1,0]],
        [[0,0,1],
         [0,0,1],
         [0,0,1],
         [1,0,0]],
        [[1,0,0],
         [1,0,0],
         [1,0,0],
         [0,1,0]],
        [[1,0,0],
         [0,1,0],
         [1,0,0],
         [1,0,0]],
        [[1,0,0],
         [0,1,0],
         [0,1,0],
         [1,0,0]]
    ], dtype=torch.int32, device=device)

    Bbound = rsr.get_boundary_branches(B1)
    assert torch.equal(Bbound, expected)

def test_get_boundary_branches2():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    B = torch.tensor([
        [[1,1,1],
         [1,1,1],
         [1,0,0],
         [1,1,0]],

        [[1,1,1],
         [1,1,1],
         [0,1,1],
         [1,0,0]]
    ], dtype=torch.int32, device=device)

    expected = torch.tensor([
        [[0,0,1],
         [0,0,1],
         [1,0,0],
         [0,1,0]],
        [[0,0,1],
         [0,0,1],
         [0,0,1],
         [1,0,0]],
        [[1,0,0],
         [1,0,0],
         [1,0,0],
         [1,0,0]],
        [[1,0,0],
         [1,0,0],
         [0,1,0],
         [1,0,0]]
    ], dtype=torch.int32, device=device)

    Bbound = rsr.get_boundary_branches(B)
    assert torch.equal(Bbound, expected), "Test 2 failed: Bbound does not match expected output"

def test_is_intersect1():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = torch.tensor([
        [[0,0,1],[1,0,0],[1,0,0],[0,1,0]],
        [[0,0,1],[0,0,1],[1,0,0],[0,1,0]],
        [[0,0,1],[0,0,1],[0,0,1],[1,0,0]],
        [[1,0,0],[1,0,0],[1,0,0],[0,1,0]],
        [[1,0,0],[0,1,0],[1,0,0],[1,0,0]],
        [[1,0,0],[0,1,0],[0,1,0],[0,1,0]]
    ], dtype=torch.int32, device=device)

    R = torch.tensor([
        [[1,1,1],[1,1,1],[1,1,1],[0,0,1]],
        [[1,1,1],[1,1,1],[0,1,1],[0,1,1]]
    ], dtype=torch.int32, device=device)

    expected = torch.tensor([False, False, False, False, False, True], device=device)
    result = rsr.is_intersect(B, R)
    assert torch.equal(result, expected)

def test_is_intersect2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = torch.tensor([
        [[0,0,1],[0,0,1],[1,0,0],[0,1,0]],
        [[0,0,1],[0,0,1],[0,0,1],[1,0,0]],
        [[1,0,0],[1,0,0],[1,0,0],[1,0,0]],
        [[1,0,0],[1,0,0],[0,1,0],[1,0,0]]
    ], dtype=torch.int32, device=device)

    R = torch.tensor([
        [[1,1,1],[1,1,1],[1,1,1],[1,0,1]],
        [[1,1,1],[1,1,1],[0,1,1],[0,1,1]]
    ], dtype=torch.int32, device=device)

    expected = torch.tensor([False, True, True, True], device=device)
    result = rsr.is_intersect(B, R)
    assert torch.equal(result, expected)

def test_from_Bbound_to_comps_st1():
    B = torch.tensor([
        [0, 0, 1],
        [1, 0, 0],
        [1, 0, 0],
        [0, 1, 0]
    ], dtype=torch.int32)

    row_names = ['x1', 'x2', 'x3', 'x4']
    expected = {'x1': 2, 'x2': 0, 'x3': 0, 'x4': 1}

    result = rsr.from_Bbound_to_comps_st(B, row_names)
    assert result == expected, f"Expected {expected}, but got {result}"

def test_is_subset1():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Rnew = torch.tensor(
        [[0,1,1],[0,1,1],[0,0,1],[1,1,1]],
        dtype=torch.int32, device=device
    )

    R = torch.tensor([
        [[1,1,1],[1,1,1],[1,1,1],[0,0,1]],
        [[1,1,1],[1,1,1],[0,1,1],[0,1,1]]
    ], dtype=torch.int32, device=device)

    is_mat_subset, is_tensor_subset = rsr.is_subset(Rnew, R)

    assert is_mat_subset == False
    assert torch.equal(is_tensor_subset, torch.tensor([False, False], device=device))

def test_is_subset2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Rnew = torch.tensor(
        [[0,1,1],[0,1,1],[0,0,1],[1,1,1]],
        dtype=torch.int32, device=device
    )

    R = torch.tensor([
        [[0,0,1],[0,0,1],[0,0,1],[0,1,1]],
        [[1,1,1],[1,1,1],[0,1,1],[1,1,1]]
    ], dtype=torch.int32, device=device)

    is_mat_subset, is_tensor_subset = rsr.is_subset(Rnew, R)

    assert is_mat_subset == True
    assert torch.equal(is_tensor_subset, torch.tensor([True, False], device=device))

def test_find_first_nonempty_combination1():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    R = torch.tensor([
        [[0,0,1],[0,0,1],[0,0,1],[0,1,1]],
        [[1,1,1],[1,1,1],[0,1,1],[1,1,1]]
    ], dtype=torch.int32, device=device)

    Rcs = []
    for i in range(R.shape[0]):
        Ri = R[i,:,:]
        Ri_c = rsr.get_complementary_events(Ri)
        Rcs.append(Ri_c)

    mat = rsr.find_first_nonempty_combination(Rcs, verbose=False)
    expected = torch.tensor([[1,1,0], [1,1,1], [1,0,0], [1,1,1]], dtype=torch.int32, device=device)
    assert torch.equal(mat, expected), f"Expected {expected}, but got {mat}"

def test_find_first_nonempty_combination2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    R = torch.tensor([
        [[1,1,1],[1,1,1],[1,1,1],[0,0,1],[0,1,1]],
        [[1,1,1],[1,1,1],[0,1,1],[0,1,1],[0,1,1]]
    ], dtype=torch.int32, device=device)

    Rcs = []
    for i in range(R.shape[0]):
        Ri = R[i,:,:]
        Ri_c = rsr.get_complementary_events(Ri)
        Rcs.append(Ri_c)

    mat = rsr.find_first_nonempty_combination(Rcs, verbose=False)
    expected = torch.tensor([[1,1,1], [1,1,1], [1,0,0], [1,1,0],[1,1,1]], dtype=torch.int32, device=device)
    assert torch.equal(mat, expected), f"Expected {expected}, but got {mat}"

def to_set_of_tuples(T: torch.Tensor):
    # T is (n, m, k), so each branch is (m, k)
    # To make the testing immune to the order of branches
    return {tuple(T[i].flatten().tolist()) for i in range(T.size(0))}

def test_merge_branches1():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = torch.tensor([
    [[1,0,0],[0,0,1],[1,0,0],[0,0,1]], # merged with 6th branch
    [[0,1,1],[0,1,0],[1,0,0],[0,0,1]], 
    [[0,1,1],[1,0,0],[0,1,1],[0,0,1]],
    [[1,0,0],[0,0,1],[0,1,1],[1,1,0]], # merged with 7th branch
    [[0,1,1],[0,1,0],[0,1,1],[1,1,0]], 
    [[0,1,1],[0,0,1],[1,0,0],[0,0,1]],
    [[0,1,1],[0,0,1],[0,1,1],[1,1,0]]
    ], dtype=torch.int32, device=device)


    expected = torch.tensor([
    [[0,1,1],[0,1,0],[1,0,0],[0,0,1]],
    [[0,1,1],[1,0,0],[0,1,1],[0,0,1]], 
    [[0,1,1],[0,1,0],[0,1,1],[1,1,0]],
    [[1,1,1],[0,0,1],[1,0,0],[0,0,1]],
    [[1,1,1],[0,0,1],[0,1,1],[1,1,0]]
    ], dtype=torch.int32, device=device)

    result = rsr.merge_branches(B)

    for i in range(result.shape[0]):
        print(f"Result branch {i}: {result[i].cpu().numpy()}")

    expected_set = to_set_of_tuples(expected.cpu())
    result_set = to_set_of_tuples(result.cpu())

    assert expected_set == result_set, f"Expected {expected_set}, but got {result_set}"

def test_merge_branches2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = torch.tensor([
    [[1,0,0],[1,0,0],[0,1,1],[0,1,1]], # merged with 3rd branch
    [[1,0,0],[0,1,1],[1,0,0],[0,1,1]], # merged with 4th branch
    [[0,1,1],[1,0,0],[0,1,1],[0,1,1]],
    [[0,1,1],[0,1,1],[1,0,0],[0,1,1]],
    [[1,1,1],[0,1,1],[0,1,1],[1,1,1]]
    ], dtype=torch.int32, device=device)


    expected = torch.tensor([
    [[1,1,1],[0,1,1],[0,1,1],[1,1,1]],
    [[1,1,1],[1,0,0],[0,1,1],[0,1,1]],
    [[1,1,1],[0,1,1],[1,0,0],[0,1,1]]
    ], dtype=torch.int32, device=device)


    result = rsr.merge_branches(B)

    expected_set = to_set_of_tuples(expected.cpu())
    result_set = to_set_of_tuples(result.cpu())

    assert expected_set == result_set, f"Expected {expected_set}, but got {result_set}"


def test_merge_branches3():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = torch.tensor([
    [[1,1,1],[1,1,1],[1,0,0],[0,0,1]],
    [[1,1,1],[0,1,1],[0,1,1],[0,1,1]], # merged with 3rd branch
    [[1,1,1],[1,0,0],[0,1,1],[0,1,1]]
    ], dtype=torch.int32, device=device)


    expected = torch.tensor([
    [[1,1,1],[1,1,1],[1,0,0],[0,0,1]],
    [[1,1,1],[1,1,1],[0,1,1],[0,1,1]]
    ], dtype=torch.int32, device=device)

    result = rsr.merge_branches(B)

    expected_set = to_set_of_tuples(expected.cpu())
    result_set = to_set_of_tuples(result.cpu())

    assert expected_set == result_set, f"Expected {expected_set}, but got {result_set}"

def test_get_complementary_events_nondisjoint1():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    R1 = torch.tensor([[1,1,1], [1,1,1], [1,1,1], [0,0,1]], dtype=torch.int32, device=device)

    expected = torch.tensor([[[1,1,1], [1,1,1], [1,1,1], [1,1,0]]], dtype=torch.int32, device=device)

    result = rsr.get_complementary_events_nondisjoint(R1)

    assert torch.equal(result, expected), f"Expected {expected}, but got {result}"

def test_get_complementary_events_nondisjoint2():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    R2 = torch.tensor([[1,1,1], [1,1,0], [1,0,0], [1,0,0]], dtype=torch.int32, device=device)

    expected = torch.tensor([[[1,1,1], [0,0,1], [1,1,1], [1,1,1]],
                             [[1,1,1], [1,1,1], [0,1,1], [1,1,1]],
                             [[1,1,1], [1,1,1], [1,1,1], [0,1,1]]], dtype=torch.int32, device=device)

    result = rsr.get_complementary_events_nondisjoint(R2)

    assert torch.equal(result, expected), f"Expected {expected}, but got {result}"

@pytest.fixture
def def_B1():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = torch.tensor([
        [[1,0,0],[1,0,0],[0,1,1],[0,1,1],[0,1,1]], # merged with 3rd branch
        [[1,0,0],[0,1,1],[1,0,0],[0,1,1],[0,1,1]], # merged with 4th branch
        [[0,1,1],[1,0,0],[0,1,1],[0,1,1],[0,1,1]],
        [[0,1,1],[0,1,1],[1,0,0],[0,1,1],[0,1,1]],
        [[1,1,1],[0,1,1],[0,1,1],[1,1,1],[0,1,1]]
        ], dtype=torch.int32, device=device)

    return B

@pytest.fixture
def def_B2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = torch.tensor([
    [[1,0,0],[0,0,1],[1,0,0],[0,0,1],[0,1,1]],
    [[0,1,1],[0,1,0],[1,0,0],[0,0,1],[0,1,1]],
    [[0,1,1],[1,0,0],[0,1,1],[0,0,1],[0,1,1]],
    [[1,0,0],[0,0,1],[0,1,1],[1,1,0],[0,1,1]],
    [[0,1,1],[0,1,0],[0,1,1],[1,1,0],[0,1,1]],
    [[0,1,1],[0,0,1],[1,0,0],[0,0,1],[0,1,1]],
    [[0,1,1],[0,0,1],[0,1,1],[1,1,0],[0,1,1]]
    ], dtype=torch.int32, device=device)
    return B

@pytest.fixture
def def_B3():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B = torch.tensor([
    [[1,1,1],[1,1,1],[1,0,0],[0,0,1],[0,1,1]],
    [[1,1,1],[0,1,1],[0,1,1],[0,1,1],[0,1,1]], # merged with 3rd branch
    [[1,1,1],[1,0,0],[0,1,1],[0,1,1],[0,1,1]]
    ], dtype=torch.int32, device=device)
    return B

def test_bit_compress1(def_B1):
    result = rsr.bit_compress(def_B1)
    expected = torch.tensor([
        [1, 1, 6, 6, 6],
        [1, 6, 1, 6, 6],
        [6, 1, 6, 6, 6],
        [6, 6, 1, 6, 6],
        [7, 6, 6, 7, 6]
    ], dtype=torch.int32, device=def_B1.device)

    assert torch.equal(result, expected), f"Expected {expected}, but got {result}"

def test_bit_compress2(def_B2):
    result = rsr.bit_compress(def_B2)
    expected = torch.tensor([
        [1, 4, 1, 4, 6],
        [6, 2, 1, 4, 6],
        [6, 1, 6, 4, 6],
        [1, 4, 6, 3, 6],
        [6, 2, 6, 3, 6],
        [6, 4, 1, 4, 6],
        [6, 4, 6, 3, 6]
    ], dtype=torch.int32, device=def_B2.device)

    assert torch.equal(result, expected), f"Expected {expected}, but got {result}"

def test_bit_compress3(def_B3):
    result = rsr.bit_compress(def_B3)
    expected = torch.tensor([
        [7, 7, 1, 4, 6],
        [7, 6, 6, 6, 6],
        [7, 1, 6, 6, 6]
    ], dtype=torch.int32, device=def_B3.device)

    assert torch.equal(result, expected), f"Expected {expected}, but got {result}"

def tensor_to_list(groups):
    # groups: list[list[tensor]] -> list[list[list]]
    return [[t.cpu().tolist() for t in col] for col in groups]

def test_groups_by_column_remhash_dict1(def_B1):
    B_com = rsr.bit_compress(def_B1)
    out = rsr.groups_by_column_remhash_dict(B_com)

    out = tensor_to_list(out)

    expected = [[torch.tensor([0,2], dtype=torch.int32, device=def_B1.device),torch.tensor([1,3], dtype=torch.int32, device=def_B1.device)],
                [], [], [], []]
    expected = tensor_to_list(expected)

    assert out == expected, f"Expected {expected}, but got {out}"

def test_groups_by_column_remhash_dict2(def_B2):
    B_com = rsr.bit_compress(def_B2)
    out = rsr.groups_by_column_remhash_dict(B_com)

    out = tensor_to_list(out)

    expected = [[torch.tensor([0,5], dtype=torch.int32, device=def_B2.device),torch.tensor([3,6], dtype=torch.int32, device=def_B2.device)],
                [torch.tensor([1, 5], dtype=torch.int32, device=def_B2.device), torch.tensor([4, 6], dtype=torch.int32, device=def_B2.device)],
                [], [], []]
    expected = tensor_to_list(expected)

    assert out == expected, f"Expected {expected}, but got {out}"

def test_groups_by_column_remhash_dict3(def_B3):
    B_com = rsr.bit_compress(def_B3)
    out = rsr.groups_by_column_remhash_dict(B_com)

    out = tensor_to_list(out)

    expected = [[],
                [torch.tensor([1, 2], dtype=torch.int32, device=def_B3.device)],
                [], [], []]
    expected = tensor_to_list(expected)

    assert out == expected, f"Expected {expected}, but got {out}"

def test_plan_merges1():
    groups_per_col = [
        [torch.tensor([0, 1, 2], dtype=torch.int32)],
        [torch.tensor([3, 4], dtype=torch.int32)],
        [torch.tensor([5, 6], dtype=torch.int32)]
    ]
    n_rows = 7
    expected = [(0, 1, 0), (3, 4, 1), (5, 6, 2)]
    out = rsr.plan_merges(groups_per_col, n_rows)
    assert out == expected, f"Expected {expected}, but got {out}"

def test_plan_merges2():
    groups_per_col = [
        [torch.tensor([0, 1, 2, 3], dtype=torch.int32)],
        [torch.tensor([3, 4], dtype=torch.int32)],
        [torch.tensor([5, 6], dtype=torch.int32)]
    ]
    n_rows = 7
    expected = [(0, 1, 0), (2, 3, 0), (5, 6, 2)]
    out = rsr.plan_merges(groups_per_col, n_rows)
    assert out == expected, f"Expected {expected}, but got {out}"

def test_plan_merges3():
    groups_per_col = [
        [torch.tensor([0, 1, 2, 3], dtype=torch.int32)],
        [torch.tensor([3, 4], dtype=torch.int32)],
        [],
        [torch.tensor([5, 6], dtype=torch.int32)]
    ]
    n_rows = 7
    expected = [(0, 1, 0), (2, 3, 0), (5, 6, 3)]
    out = rsr.plan_merges(groups_per_col, n_rows)
    assert out == expected, f"Expected {expected}, but got {out}"

def test_apply_merges1(def_B2):

    B = def_B2
    merge_plan = [(0, 5, 0), (3, 6, 0)]

    B_merged, kept_indices = rsr.apply_merges(B, merge_plan)

    expected = torch.tensor([
    [[0,1,1],[0,1,0],[1,0,0],[0,0,1],[0,1,1]],
    [[0,1,1],[1,0,0],[0,1,1],[0,0,1],[0,1,1]],
    [[0,1,1],[0,1,0],[0,1,1],[1,1,0],[0,1,1]],
    [[1,1,1],[0,0,1],[1,0,0],[0,0,1],[0,1,1]],
    [[1,1,1],[0,0,1],[0,1,1],[1,1,0],[0,1,1]]
    ], dtype=torch.int32, device=B.device)

    B_merged = to_set_of_tuples(B_merged.cpu())
    expected = to_set_of_tuples(expected)

    assert B_merged == expected, f"Expected {expected}, but got {B_merged}"

def test_apply_merges2(def_B1):

    B = def_B1
    merge_plan = [(0, 2, 0), (1, 3, 0)]

    B_merged, kept_indices = rsr.apply_merges(B, merge_plan)

    expected = torch.tensor([
    [[1,1,1],[0,1,1],[0,1,1],[1,1,1],[0,1,1]],
    [[1,1,1],[1,0,0],[0,1,1],[0,1,1],[0,1,1]],
    [[1,1,1],[0,1,1],[1,0,0],[0,1,1],[0,1,1]]
    ], dtype=torch.int32, device=B.device)

    B_merged = to_set_of_tuples(B_merged.cpu())
    expected = to_set_of_tuples(expected)

    assert B_merged == expected, f"Expected {expected}, but got {B_merged}"

def test_apply_merges3(def_B3):

    B = def_B3
    merge_plan = [(1, 2, 1)]

    B_merged, kept_indices = rsr.apply_merges(B, merge_plan)

    expected = torch.tensor([
    [[1,1,1],[1,1,1],[1,0,0],[0,0,1],[0,1,1]],
    [[1,1,1],[1,1,1],[0,1,1],[0,1,1],[0,1,1]]
    ], dtype=torch.int32, device=B.device)

    B_merged = to_set_of_tuples(B_merged.cpu())
    expected = to_set_of_tuples(expected)

    assert B_merged == expected, f"Expected {expected}, but got {B_merged}"

@pytest.fixture
def ex_upper_lower_rules():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rules_mat_upper = torch.tensor([
        [[1, 1, 1], [1, 1, 1], [1, 1, 1], [0, 0, 1]],
        [[1, 1, 1], [1, 1, 1], [0, 1, 1], [0, 1, 1]],
        [[1, 1, 1], [0, 1, 1], [1, 1, 1], [0, 1, 1]]
    ], dtype=torch.int32, device=device)
    rules_mat_lower = torch.tensor([
        [[1, 1, 1], [1, 1, 0], [1, 0, 0], [1, 0, 0]],
        [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 1]]
    ], dtype=torch.int32, device=device)

    probs = torch.tensor([
        [0.3, 0.7, 0.0],
        [0.1, 0.2, 0.7],
        [0.2, 0.8, 0.0],
        [0.1, 0.3, 0.6]
    ], dtype=torch.float32, device=device)

    return rules_mat_upper, rules_mat_lower, probs

@pytest.fixture
def ex_upper_lower_rules_with_dict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    row_names = ['x1', 'x2', 'x3', 'x4']
    rules_mat_upper = torch.tensor([
        [[1, 1, 1], [1, 1, 1], [1, 1, 1], [0, 0, 1]],
        [[1, 1, 1], [1, 1, 1], [0, 1, 1], [0, 1, 1]],
        [[1, 1, 1], [0, 1, 1], [1, 1, 1], [0, 1, 1]]
    ], dtype=torch.int32, device=device)
    rules_mat_lower = torch.tensor([
        [[1, 1, 1], [1, 1, 0], [1, 0, 0], [1, 0, 0]],
        [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 1]]
    ], dtype=torch.int32, device=device)

    rules_upper = [{'x4': ('>=', 2), 'sys': ('>=', 1)},
                  {'x3': ('>=', 1), 'x4': ('>=', 1), 'sys': ('>=', 1)},
                  {'x2': ('>=', 1), 'x4': ('>=', 1), 'sys': ('>=', 1)}]
    rules_lower = [{'x2': ('<=', 1), 'x3': ('<=', 0), 'x4': ('<=', 0), 'sys': ('<=', 0)},
                  {'x1': ('<=', 0),'x2': ('<=', 0), 'x3': ('<=', 0), 'sys': ('<=', 0)}]

    return rules_mat_upper, rules_mat_lower, rules_upper, rules_lower, row_names

def test_mask_from_first_one1():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0]], device='cuda:0', dtype=torch.int32)

    x_after = rsr.mask_from_first_one(x, mode="after")
    x_after_expected = torch.tensor([[1, 1, 1],
        [0, 1, 1],
        [1, 1, 1],
        [0, 1, 1]], device='cuda:0', dtype=torch.int32)

    x_before = rsr.mask_from_first_one(x, mode="before")
    x_before_expected = torch.tensor([[1, 0, 0],
        [1, 1, 0],
        [1, 0, 0],
        [1, 1, 0]], device='cuda:0', dtype=torch.int32)

    assert torch.equal(x_after, x_after_expected), f"Expected {x_after_expected}, but got {x_after}"
    assert torch.equal(x_before, x_before_expected), f"Expected {x_before_expected}, but got {x_before}"

def test_mask_from_first_one2():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor([[[1, 0, 0],
                      [0, 0, 1],
                      [0, 1, 0],
                      [1, 0, 0]],
                      [[0, 1, 0],
                      [0, 1, 0],
                      [0, 1, 0],
                      [1, 0, 0]]], device='cuda:0', dtype=torch.int32)

    x_after = rsr.mask_from_first_one(x, mode="after")
    x_after_expected = torch.tensor([[[1, 1, 1],
                      [0, 0, 1],
                      [0, 1, 1],
                      [1, 1, 1]],
                      [[0, 1, 1],
                      [0, 1, 1],
                      [0, 1, 1],
                      [1, 1, 1]]], device='cuda:0', dtype=torch.int32)

    x_before = rsr.mask_from_first_one(x, mode="before")
    x_before_expected = torch.tensor([[[1, 0, 0],
                      [1, 1, 1],
                      [1, 1, 0],
                      [1, 0, 0]],
                      [[1, 1, 0],
                      [1, 1, 0],
                      [1, 1, 0],
                      [1, 0, 0]]], device='cuda:0', dtype=torch.int32)

    assert torch.equal(x_after, x_after_expected), f"Expected {x_after_expected}, but got {x_after}"
    assert torch.equal(x_before, x_before_expected), f"Expected {x_before_expected}, but got {x_before}"



def test_update_rules1(ex_upper_lower_rules_with_dict):
    rules_mat_upper, rules_mat_lower, rules_upper, rules_lower, row_names = ex_upper_lower_rules_with_dict

    min_comps_st = {'x1': ('<=', 0), 'x3': ('<=', 0), 'sys': ('<=', 0)}
    rules_dict, rules_mat = rsr.update_rules(min_comps_st, rules_lower, rules_mat_lower, row_names)

    expected_rules_dict = [{'x2': ('<=', 1), 'x3': ('<=', 0), 'x4': ('<=', 0), 'sys': ('<=', 0)},
                           {'x1': ('<=', 0), 'x3': ('<=', 0), 'sys': ('<=', 0)}]

    expected_rules_mat = torch.tensor([
        [[1, 1, 1], [1, 1, 0], [1, 0, 0], [1, 0, 0]],
        [[1, 0, 0], [1, 1, 1], [1, 0, 0], [1, 1, 1]]
    ], dtype=torch.int32, device=rules_mat_lower.device)

    assert rules_dict == expected_rules_dict, f"Expected {expected_rules_dict}, but got {rules_dict}"
    assert torch.equal(rules_mat, expected_rules_mat), f"Expected {expected_rules_mat}, but got {rules_mat}"

def test_update_rules2(ex_upper_lower_rules_with_dict):
    rules_mat_upper, rules_mat_lower, rules_upper, rules_lower, row_names = ex_upper_lower_rules_with_dict

    min_comps_st = {'x1': ('>=', 1), 'x2': ('>=', 2), 'sys': ('>=', 1)}
    rules_dict, rules_mat = rsr.update_rules(min_comps_st, rules_upper, rules_mat_upper, row_names)

    expected_rules_dict = [{'x4': ('>=', 2), 'sys': ('>=', 1)},
                           {'x3': ('>=', 1), 'x4': ('>=', 1), 'sys': ('>=', 1)},
                           {'x2': ('>=', 1), 'x4': ('>=', 1), 'sys': ('>=', 1)},
                           {'x1': ('>=', 1), 'x2': ('>=', 2), 'sys': ('>=', 1)}]

    expected_rules_mat = torch.tensor([
        [[1, 1, 1], [1, 1, 1], [1, 1, 1], [0, 0, 1]],
        [[1, 1, 1], [1, 1, 1], [0, 1, 1], [0, 1, 1]],
        [[1, 1, 1], [0, 1, 1], [1, 1, 1], [0, 1, 1]],
        [[0, 1, 1], [0, 0, 1], [1, 1, 1], [1, 1, 1]]
    ], dtype=torch.int32, device=rules_mat_upper.device)

    assert rules_dict == expected_rules_dict, f"Expected {expected_rules_dict}, but got {rules_dict}"
    assert torch.equal(rules_mat, expected_rules_mat), f"Expected {expected_rules_mat}, but got {rules_mat}"

def test_update_rules3(ex_upper_lower_rules_with_dict):
    rules_mat_upper, rules_mat_lower, rules_upper, rules_lower, row_names = ex_upper_lower_rules_with_dict

    min_comps_st = {'x1': ('>=', 1), 'x4': ('>=', 2), 'sys': ('>=', 1)}
    rules_dict, rules_mat = rsr.update_rules(min_comps_st, rules_upper, rules_mat_upper, row_names)

    expected_rules_dict = [{'x4': ('>=', 2), 'sys': ('>=', 1)},
                           {'x3': ('>=', 1), 'x4': ('>=', 1), 'sys': ('>=', 1)},
                           {'x2': ('>=', 1), 'x4': ('>=', 1), 'sys': ('>=', 1)}]

    expected_rules_mat = torch.tensor([
        [[1, 1, 1], [1, 1, 1], [1, 1, 1], [0, 0, 1]],
        [[1, 1, 1], [1, 1, 1], [0, 1, 1], [0, 1, 1]],
        [[1, 1, 1], [0, 1, 1], [1, 1, 1], [0, 1, 1]]
    ], dtype=torch.int32, device=rules_mat_upper.device)

    assert rules_dict == expected_rules_dict, f"Expected {expected_rules_dict}, but got {rules_dict}"
    assert torch.equal(rules_mat, expected_rules_mat), f"Expected {expected_rules_mat}, but got {rules_mat}"

def test_update_rules4(ex_upper_lower_rules_with_dict):
    rules_mat_upper, rules_mat_lower, rules_upper, rules_lower, row_names = ex_upper_lower_rules_with_dict

    min_comps_st = {'x2': ('<=', 0), 'x3': ('<=', 0), 'x4': ('<=', 0), 'sys': ('<=', 0)}
    rules_dict, rules_mat = rsr.update_rules(min_comps_st, rules_lower, rules_mat_lower, row_names)

    expected_rules_dict = [{'x2': ('<=', 1), 'x3': ('<=', 0), 'x4': ('<=', 0), 'sys': ('<=', 0)},
                           {'x1': ('<=', 0),'x2': ('<=', 0), 'x3': ('<=', 0), 'sys': ('<=', 0)}]

    expected_rules_mat = torch.tensor([
        [[1, 1, 1], [1, 1, 0], [1, 0, 0], [1, 0, 0]],
        [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 1]]
    ], dtype=torch.int32, device=rules_mat_lower.device)

    assert rules_dict == expected_rules_dict, f"Expected {expected_rules_dict}, but got {rules_dict}"
    assert torch.equal(rules_mat, expected_rules_mat), f"Expected {expected_rules_mat}, but got {rules_mat}"


def test_update_rules_batch_matches_sequential(ex_upper_lower_rules_with_dict):
    """Batch update_rules should produce same results as sequential calls."""
    rules_mat_upper, rules_mat_lower, rules_upper, rules_lower, row_names = ex_upper_lower_rules_with_dict

    # Two new rules to add sequentially
    new_rules = [
        {'x1': ('<=', 0), 'x3': ('<=', 0), 'sys': ('<=', 0)},
        {'x2': ('<=', 0), 'x3': ('<=', 0), 'x4': ('<=', 0), 'sys': ('<=', 0)},
    ]

    # Sequential: apply update_rules one by one
    seq_dict = list(rules_lower)
    seq_mat = rules_mat_lower.clone()
    for rd in new_rules:
        seq_dict, seq_mat = rsr.update_rules(rd, seq_dict, seq_mat, row_names)

    # Batch: apply all at once
    batch_dict, batch_mat, n_added, n_removed = rsr.update_rules_batch(
        new_rules, list(rules_lower), rules_mat_lower.clone(), row_names)

    # Both should have the same rules (order may differ, so compare as sets of tuples)
    seq_set = {tuple(sorted(d.items())) for d in seq_dict}
    batch_set = {tuple(sorted(d.items())) for d in batch_dict}
    assert seq_set == batch_set, f"Dicts differ:\nseq={seq_set}\nbatch={batch_set}"

    # Matrices should match (compare sorted by content)
    seq_sorted = seq_mat[seq_mat.sum(dim=(1, 2)).argsort()]
    batch_sorted = batch_mat[batch_mat.sum(dim=(1, 2)).argsort()]
    assert torch.equal(seq_sorted, batch_sorted), f"Mats differ"


def test_update_rules_batch_empty_existing(ex_upper_lower_rules_with_dict):
    """Batch update with no existing rules."""
    _, _, _, _, row_names = ex_upper_lower_rules_with_dict

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    empty_mat = torch.zeros((0, len(row_names), 3), dtype=torch.int32, device=device)
    new_rules = [
        {'x1': ('<=', 0), 'sys': ('<=', 0)},
        {'x2': ('>=', 2), 'sys': ('>=', 1)},
    ]

    rules_dict, rules_mat, n_added, n_removed = rsr.update_rules_batch(
        new_rules, [], empty_mat, row_names)

    assert n_added == 2
    assert n_removed == 0
    assert len(rules_dict) == 2
    assert rules_mat.shape[0] == 2


def test_update_rules_batch_new_dominated_by_new():
    """When a new rule dominates another new rule, only the dominator survives."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    row_names = ['x1', 'x2']
    empty_mat = torch.zeros((0, 2, 2), dtype=torch.int32, device=device)

    new_rules = [
        {'x1': ('<=', 0), 'x2': ('<=', 0), 'sys': ('<=', 0)},  # more specific
        {'x1': ('<=', 0), 'sys': ('<=', 0)},  # less specific (dominates the above)
    ]

    rules_dict, rules_mat, n_added, n_removed = rsr.update_rules_batch(
        new_rules, [], empty_mat, row_names)

    assert n_added == 1
    assert len(rules_dict) == 1
    # The less specific rule should survive (it dominates the more specific one)
    assert 'x2' not in rules_dict[0] or rules_dict[0].get('x2', (None, None))[1] != 0


def test_update_rules_batch_duplicate_rules():
    """Duplicate rules should not eliminate each other — one copy must survive."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    row_names = ['x1', 'x2']
    empty_mat = torch.zeros((0, 2, 2), dtype=torch.int32, device=device)

    rule = {'x1': ('<=', 0), 'sys': ('<=', 0)}
    new_rules = [dict(rule), dict(rule), dict(rule)]  # 3 identical rules

    rules_dict, rules_mat, n_added, n_removed = rsr.update_rules_batch(
        new_rules, [], empty_mat, row_names)

    assert n_added == 1, f"Expected 1 rule added from 3 duplicates, got {n_added}"
    assert len(rules_dict) == 1
    assert rules_mat.shape[0] == 1


@pytest.fixture
def upper_lower_rules_ex_4comps():
    surv_rules = [{'x1': ('>=', 1), 'x2': ('>=', 2), 'x3': ('>=', 1), 'x4': ('>=', 2), 'sys': ('>=', 1)},
                  {'x1': ('>=', 2), 'x2': ('>=', 1), 'x3': ('>=', 2), 'x4': ('>=', 1), 'sys': ('>=', 1)}]
    fail_rules = [{'x1': ('<=', 0), 'sys': ('<=', 0)},
                  {'x1': ('<=', 1), 'x2': ('<=', 1), 'sys': ('<=', 0)},
                  {'x1': ('<=', 1), 'x4': ('<=', 1), 'sys': ('<=', 0)},
                  {'x3': ('<=', 1), 'x4': ('<=', 1), 'sys': ('<=', 0)},
                  {'x2': ('<=', 1), 'x3': ('<=', 1), 'sys': ('<=', 0)},
                  {'x2': ('<=', 0), 'sys': ('<=', 0)},
                  {'x3': ('<=', 0), 'sys': ('<=', 0)},
                  {'x4': ('<=', 0), 'sys': ('<=', 0)}]

    row_names = ['x1', 'x2', 'x3', 'x4']

    return surv_rules, fail_rules, row_names

def test_minimise_upper_states_random1(upper_lower_rules_ex_4comps):
    surv_rules, fail_rules, row_names = upper_lower_rules_ex_4comps

    comps_st = {x: 2 for x in row_names}

    def sfun(comps_st):
        # return value, system_status, info
        for s in surv_rules:
            if all(comps_st[k] >= v[1] for k, v in s.items() if k in comps_st):
                return None, 1, None
        return None, 0, None

    new_rule, info = rsr.minimise_upper_states_random(comps_st, sfun, sys_upper_st=1)

    assert new_rule in surv_rules, f"Expected one of {surv_rules}, but got {new_rule}"

def test_minimise_upper_states_random2(upper_lower_rules_ex_4comps):
    surv_rules, fail_rules, row_names = upper_lower_rules_ex_4comps

    comps_st = {x: 2 for x in row_names if x != 'sys'}
    comps_st['x1'] = 1

    def sfun(comps_st):
        for s in surv_rules:
            if all(comps_st[k] >= v[1] for k, v in s.items() if k in comps_st):
                return None, 1, None
        return None, 0, None

    new_rule, info = rsr.minimise_upper_states_random(comps_st, sfun, sys_upper_st=1)
    assert new_rule in surv_rules, f"Expected one of {surv_rules}, but got {new_rule}"

def test_minimise_lower_states_random1(upper_lower_rules_ex_4comps):
    surv_rules, fail_rules, row_names = upper_lower_rules_ex_4comps

    comps_st = {x: 0 for x in row_names if x != 'sys'}

    def sfun(comps_st):
        for s in surv_rules:
            if all(comps_st[k] >= v[1] for k, v in s.items() if k in comps_st):
                return None, 1, None
        return None, 0, None

    new_rule, info = rsr.minimise_lower_states_random(comps_st, sfun, sys_lower_st=0, max_state=2)
    assert new_rule in fail_rules, f"Got {new_rule}"

def test_minimise_lower_states_random2(upper_lower_rules_ex_4comps):
    surv_rules, fail_rules, row_names = upper_lower_rules_ex_4comps

    comps_st = {x: 0 for x in row_names if x != 'sys'}
    comps_st['x1'] = 1

    def sfun(comps_st):
        for s in surv_rules:
            if all(comps_st[k] >= v[1] for k, v in s.items() if k in comps_st):
                return None, 1, None
        return None, 0, None

    new_rule, info = rsr.minimise_lower_states_random(comps_st, sfun, sys_lower_st=0, max_state=2)
    assert new_rule in fail_rules, f"Got {new_rule}"

# ---------- Fixture: 5 components, binary states ----------
@pytest.fixture
def def_five_comp():
    """
    Five binary components; rules are 0/1 indicator matrices of shape (n_var, n_state).
    We'll interpret "subset" as (sample_onehot & rule) == sample_onehot.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # failure / upper rule tensors for the (single) threshold level (state >= 1)
    # Shape: (n_var, n_state) = (5, 2)
    # You can tweak these if you want different logical patterns.
    lower_rules = torch.Tensor(
        [[[1, 0], [1, 0], [1, 1], [1, 1], [1, 1]],
        [[1, 1], [1, 1], [1, 1], [1, 0], [1, 0]],
        [[1, 0], [1, 1], [1, 0], [1, 1], [1, 0]],
        [[1, 1], [1, 0], [1, 1], [1, 1], [1, 1]]]
    )

    upper_rules = torch.Tensor(
        [[[0, 1], [1, 1], [1, 1], [0, 1], [1, 1]],
        [[1, 1], [0, 1], [1, 1], [1, 1], [0, 1]],
        [[0, 1], [1, 1], [0, 1], [1, 1], [0, 1]],
        [[1, 1], [0, 1], [0, 1], [0, 1], [1, 1]]])

    # per-component categorical probabilities P(state=0), P(state=1)
    probs = torch.Tensor([
        [0.1, 0.9],
        [0.1, 0.9],
        [0.1, 0.9],
        [0.1, 0.9],
        [0.1, 0.9]])

    row_names = [f"x{i}" for i in range(1, 6)]

    # Fallback resolver: compute system state (0/1) using the same subset logic against upper_rules.
    # If a sample "survives" that rule, sys_state=1, else 0.
    def s_fun(comps_dict):
        # Build one-hot sample from integer states in comps_dict
        n_var, n_state = probs.shape
        sample = torch.zeros(n_var, n_state, dtype=torch.int32, device=device)
        for i, name in enumerate(row_names):
            s = int(comps_dict[name])
            sample[i, s] = 1

        # subset check: sample ⊆ upper_rules <=> (sample & rule) == sample
        rule = upper_rules.to(dtype=torch.bool)
        smp = sample.to(dtype=torch.bool)
        is_survive = torch.all((smp & rule) == smp)
        sys_state = 1 if bool(is_survive.item()) else 0
        return None, sys_state, None

    return lower_rules, upper_rules, probs, row_names, s_fun


# ---------- Multi-state API test (returns state probabilities {0,1}) ----------
def test_get_comp_cond_sys_prob_multi_two_state(def_five_comp):
    lower_rules, upper_rules, probs, row_names, s_fun = def_five_comp

    # The multi-state function expects consecutive keys from 0..max_st in BOTH dicts.
    device = probs.device

    rules_dict_upper = {
        1: upper_rules,  
    }
    rules_dict_lower = {
        1: lower_rules,  
    }

    # Use a reasonably large n_sample but not too slow for CI;
    # seed for determinism and a small relative tolerance
    torch.manual_seed(0)
    cond_probs = rsr.get_comp_cond_sys_prob_multi(
        rules_dict_upper,
        rules_dict_lower,
        probs,
        comps_st_cond={},          # no conditioning
        row_names=row_names,
        s_fun=s_fun,
        n_sample=300_000,
        n_batch=100_000,
    )

    # Expected (from your sketch): lower ~ 0.02152, upper ~ 0.97848
    # Here states are explicit: 0=failure, 1=survival
    assert cond_probs[0] == pytest.approx(0.02152, rel=2e-2, abs=5e-4)
    assert cond_probs[1] == pytest.approx(0.97848, rel=2e-2, abs=5e-4)


# ---------- Single-state API test (returns {"lower","upper"}) ----------
def test_get_comp_cond_sys_prob__two_state(def_five_comp):
    lower_rules, upper_rules, probs, row_names, s_fun = def_five_comp

    # For the single-threshold API, sys_upper_st=1 means system survives if state >= 1
    torch.manual_seed(0)
    cond_probs = rsr.get_comp_cond_sys_prob(
        rules_mat_upper=upper_rules,
        rules_mat_lower=lower_rules,
        probs=probs,
        comps_st_cond={},         # no conditioning
        row_names=row_names,
        s_fun=s_fun,
        sys_upper_st=1,
        n_sample=300_000,
        n_batch=100_000,
    )

    assert cond_probs["lower"]  == pytest.approx(0.02152, rel=2e-2, abs=5e-4)
    assert cond_probs["upper"] == pytest.approx(0.97848, rel=2e-2, abs=5e-4)
