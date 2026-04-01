import sys
import os
import pathlib
import torch
import json
import typer
import pdb
import pandas as pd

from pathlib import Path
import networkx as nx


HOME = Path(__file__).parent
sys.path.append(str(HOME.joinpath('../../../network-datasets/')))

from ndtools import fun_binary_graph as fbg # ndtools available at github.com/jieunbyun/network-datasets
from ndtools.graphs import build_graph

from rsr import rsr

app = typer.Typer()


def s_fun(comps_st):
    conn_pop_ratio, sys_st, info = fbg.eval_population_accessibility(
		comps_st,
		G_base,
		dests,
        avg_speed=60.0, # km/h
        target_time_max = 0.25, # hours: it shouldn't take longer than this to reach any destination
        target_pop_max = [0.95, 0.99], # fraction of population that should be reachable at each destination
        length_attr = 'length_km',
        population_attr = 'population',)

    min_comps_st = None
    return conn_pop_ratio, sys_st, min_comps_st


@app.command()
def check_system():

    prerequites()

    comps_st = {eid: 1 for eid in edges.keys()}
    conn_pop_ratio, sys_st, details = s_fun(comps_st)
    print(f"conn_pop_ratio: {conn_pop_ratio}, sys_st: {sys_st}, details: {details}")


def prerequites():

    global G_base, origin, dests, sys_upper_st, device, probs, edges, edge_names, n_state

    DATASET = HOME.joinpath('./data')

    nodes = json.loads((DATASET / "nodes.json").read_text(encoding="utf-8"))
    edges = json.loads((DATASET / "edges.json").read_text(encoding="utf-8"))
    probs_dict = json.loads((DATASET / "probs_eq.json").read_text(encoding="utf-8"))

    G_base = build_graph(nodes, edges, probs_dict)

    dests = ['n22', 'n66']
    sys_upper_st = 2

    edge_names = list(edges.keys())
    n_state = 2 # binary states of components

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probs = [[probs_dict[n]['0']['p'], probs_dict[n]['1']['p']] for n in edge_names]
    probs = torch.tensor(probs, dtype=torch.float32, device=device)


@app.command()
def find_rules():

    prerequites()

    # run rule extraction: two options available: rsr.run_rule_extraction or rsr.run_rule_extraction_by_mcs
    result = rsr.run_rule_extraction_by_mcs(
        sfun=s_fun,
        probs=probs,
        row_names=edge_names,
        n_state=n_state,
        output_dir="rsr_res",
        unk_prob_thres = 1e-3,
        unk_prob_opt = 'abs',
        sys_upper_st=sys_upper_st,
    )


def load_results(rsr_path):

    prerequites()

    rsr_path = Path(rsr_path)
    assert rsr_path.exists(), f'rsr_path does not exisist: {rsr_path}'

    rules_mat_upper = torch.load(rsr_path / f"rules_geq_{sys_upper_st}.pt", map_location="cpu")
    rules_mat_upper = rules_mat_upper.to(device)
    rules_mat_lower = torch.load(rsr_path / f"rules_leq_{sys_upper_st-1}.pt", map_location="cpu")
    rules_mat_lower = rules_mat_lower.to(device)

    return rules_mat_upper, rules_mat_lower


def load_results_multi(rsr_path):

    prerequites()

    rsr_path = Path(rsr_path)
    assert rsr_path.exists(), f'rsr_path does not exisist: {rsr_path}'

    rules_dict_mat_upper = {}
    rules_dict_mat_lower = {}

    for sys_upper_st in [1, 2]:  # either 1 or 2
        rules_mat_upper = torch.load(rsr_path / f"rules_geq_{sys_upper_st}.pt", map_location="cpu")
        rules_mat_upper = rules_mat_upper.to(device)
        rules_dict_mat_upper[sys_upper_st] = rules_mat_upper
        rules_mat_lower = torch.load(rsr_path / f"rules_leq_{sys_upper_st-1}.pt", map_location="cpu")
        rules_mat_lower = rules_mat_lower.to(device)
        rules_dict_mat_lower[sys_upper_st] = rules_mat_lower

    return rules_dict_mat_upper, rules_dict_mat_lower


@app.command()
def cal_probs(rsr_path):

    rules_mat_upper, rules_mat_lower = load_results(rsr_path)

    # marginal probability
    pr_cond = rsr.get_comp_cond_sys_prob(
        rules_mat_upper,
        rules_mat_lower,
        probs,
        comps_st_cond = {},
        row_names = edge_names,
        s_fun = s_fun,
        sys_upper_st = sys_upper_st
    )
    print(f"P(sys >= {sys_upper_st}) = {pr_cond['upper']:.3e}")
    print(f"P(sys <= {sys_upper_st-1} ) = {pr_cond['lower']:.3e}\n")


@app.command()
def cal_cond_probs(rsr_path):

    rules_mat_upper, rules_mat_lower = load_results(rsr_path)

    # conditional probability given one components' survival
    for x in edge_names:
        print(f"Eval P(sys | {x}=1)")
        pr_cond = rsr.get_comp_cond_sys_prob(
            rules_mat_upper,
            rules_mat_lower,
            probs,
            comps_st_cond = {x: 1},
            row_names=edge_names,
            s_fun=s_fun,
            sys_upper_st=sys_upper_st
        )
        print(f"P(sys >= {sys_upper_st} | {x}=1) = {pr_cond['upper']:.3e}")
        print(f"P(sys <= {sys_upper_st-1} | {x}=1) = {pr_cond['lower']:.3e}\n")


@app.command()
def cal_probs_multi(rsr_path):

    rules_dict_mat_upper, rules_dict_mat_lower = load_results_multi(rsr_path)

    # marginal probability
    pr_cond = rsr.get_comp_cond_sys_prob_multi(
        rules_dict_mat_upper,
        rules_dict_mat_lower,
        probs,
        comps_st_cond = {},
        row_names = edge_names,
        s_fun = s_fun,
    )
    print(f"P(sys) = {pr_cond}")

@app.command()
def cal_cond_probs_multi(rsr_path):

    rules_dict_mat_upper, rules_dict_mat_lower = load_results_multi(rsr_path)

    results = []
    for x in edge_names:

        # Calculate probabilities
        cond_probs = rsr.get_comp_cond_sys_prob_multi(
                        rules_dict_mat_upper,
                        rules_dict_mat_lower,
                        probs,
                        comps_st_cond = {x: 0}, # 1: upper, 0: lower
                        row_names=edge_names,
                        s_fun=s_fun
                    )

        # Print results
        print(f"P(sys | {x}=0):", cond_probs)

        # Append data as a dictionary to the list
        results.append({"Component": x,
                        "System failure": cond_probs[0],
                        "Partial failure": cond_probs[1],
                        "Survival": cond_probs[2]
                        })

    # Convert the list to a DataFrame
    df_results = pd.DataFrame(results)

    # Save to a JSON file
    output_file = HOME.joinpath('post-processing/cond_sys_probs.json')
    df_results.to_json(output_file, orient="records", indent=4)

    print(f"\nData saved to {output_file}")


if __name__=='__main__':
    app()
