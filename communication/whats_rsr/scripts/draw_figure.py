import json
import sys
from matplotlib.lines import Line2D
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import numpy as np
import networkx as nx
import pandas as pd
import time
from mbnpy import brc, branch, cpm, variable, inference
from ndtools.io import load_json
from ndtools.graphs import build_graph
from ndtools.fun_binary_graph import eval_1od_connectivity, eval_global_conn_k
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from rsr import rsr

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def draw_graph_from_json(
    nodes_path: str | Path,
    edges_path: str | Path,
    save_fname: str | Path,
    node_size: float = 30,      # base size for non-OD nodes
    od_scale: float = 1.2,      # OD nodes will be node_size * od_scale
    edge_width: float = 0.5,
    figsize: tuple[float, float] = (3.4, 3.0),
    show_legend: bool = False,
) -> None:
    """
    Draw a graph from nodes/edges JSON.
    - Node sizes are parameterised by `node_size` (user input).
    - Nodes with is_od==True are drawn as red 6-point stars.
    """

    nodes_path = Path(nodes_path)
    edges_path = Path(edges_path)

    with nodes_path.open("r") as f:
        nodes = json.load(f)
    with edges_path.open("r") as f:
        edges = json.load(f)

    G = nx.Graph()
    pos = {}
    is_od = {}

    # Nodes
    for nid, attrs in nodes.items():
        G.add_node(nid)
        pos[nid] = (attrs["x"], attrs["y"])
        is_od[nid] = bool(attrs.get("is_od", False))

    # Edges (treat as undirected; your file uses directed:false)
    for _, e in edges.items():
        G.add_edge(e["from"], e["to"])

    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "legend.fontsize": 9,
    })

    fig, ax = plt.subplots(figsize=figsize, dpi=150)

    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_width, alpha=1.0)

    # Draw non-OD nodes
    regular = [n for n in G.nodes if not is_od[n]]
    nx.draw_networkx_nodes(
        G, pos, nodelist=regular, ax=ax,
        node_size=node_size,
        node_color="#467EA4",
        linewidths=0.0,
        alpha=1.0,
    )

    # Draw OD nodes as red markers
    od_nodes = [n for n in G.nodes if is_od[n]]
    nx.draw_networkx_nodes(
        G, pos, nodelist=od_nodes, ax=ax,
        node_size=node_size * od_scale,
        node_color="red",
        node_shape="x",  
        linewidths=3.0,
        alpha=1.0,
    )

    ax.set_aspect("equal")
    ax.set_axis_off()

    # --- Legend (node types) ---
    if show_legend:
        legend_elements = [
            Line2D(
                [0], [0],
                marker="x",
                linestyle="None",
                label="OD node",
                color="red",
                markersize=8,
                markeredgewidth=3.0,
            ),
        ]

        ax.legend(
            handles=legend_elements,
            loc="upper left",
            frameon=True,
            handletextpad=0.2,
            borderpad=0.4,
        )

    # --- Manually set axis limits from node positions (to minimise whitespace) ---
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]

    pad = 0.02  # small fractional padding
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    dx = xmax - xmin
    dy = ymax - ymin

    ax.set_xlim(xmin - pad * dx, xmax + pad * dx)
    ax.set_ylim(ymin - pad * dy, ymax + pad * dy)

    fig.tight_layout()
    fig.savefig( save_fname, dpi=300, bbox_inches="tight", pad_inches=0.0 )


def draw_brc_monitor_data(
    data_paths: list,
    labels: list,
) -> None:
    """
    Draws four separate figures on BRC monitor data:
    - number of reference states vs. unknown probability (with legend)
    - number of reference states vs. number of disjoint events
    - number of reference states vs. memory usage
    - number of reference states vs. cumulative hours
    """

    if len(data_paths) != len(labels):
        raise ValueError("data_paths and labels must have the same length.")

    sec_to_hr = 1.0 / 3600.0

    n_rules, unk_prob, n_brs, mem, c_time = {}, {}, {}, {}, {}

    for path, label in zip(data_paths, labels):
        with open(path, "r") as f:
            data_ = json.load(f)

        n_rules[label] = data_["no_ra"]
        unk_prob[label] = data_["pr_bu"]
        n_brs[label] = data_["no_br"]
        mem[label] = data_["rss_gb"]
        c_time[label] = np.cumsum(data_["time"]) * sec_to_hr

    def _plot_all(ydata_dict, marker, ylabel, linestyle, save_fname, show_legend=False, ylog=False):

        if len(ydata_dict.keys()) != len(labels):
            raise ValueError("ydata_dict and labels must have the same length.")

        # Figure parameters
        font_name = "Times New Roman"
        plt.rcParams["font.family"] = font_name
        fsz = 11
        lw = 1.2
        # ---

        fig = plt.figure(figsize=(3.4, 3.0), constrained_layout=True)
        ax = fig.add_axes([0.12, 0.08, 0.83, 0.83])

        for i, label in enumerate(labels):
            x = n_rules[label]
            y = ydata_dict[label]
            n = min(len(x), len(y))  # safety
            ax.plot(
                x[:n], y[:n],
                linestyle=linestyle[i], linewidth=lw,
                marker=marker[i], markevery=10, markersize=2.5,
                label=label,
                color = "#0072B2"
            )

        ax.set_xlabel("Number of reference states", fontsize=fsz)
        ax.set_ylabel(ylabel, fontsize=fsz)
        ax.tick_params(axis="both", which="major", labelsize=fsz-1)
        ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)
        if ylog:
            ax.set_yscale("log")

        if show_legend:
            ax.legend(loc="best", frameon=True, fontsize=fsz-1.0, labelspacing=0.25, handletextpad=0.4, handlelength=1.1, borderpad=0.2,)

        # --- add memory limit line if plotting memory ---
        if ylabel.lower().startswith("rss"):
            y_limit = 20

            # horizontal line
            ax.axhline(
                y=y_limit,
                color="black",
                linestyle="-",
                linewidth=1.2,
            )

            # label near the right edge, slightly above the line
            ax.text(
                ax.get_xlim()[1]-10,           # xmax (data coordinates)
                y_limit * 1.01,              # a bit above the line 
                "Threshold",
                color="black",
                fontsize=fsz,
                ha="right",
                va="bottom",
            )
            ax.set_ylim(0, 22.5)  # ensure limit line is visible


        plt.savefig(save_fname, dpi=300, bbox_inches="tight", pad_inches=0.02)

    # --- Four separate figures ---
    HERE = Path(__file__).resolve().parent
    _plot_all(
        ydata_dict=unk_prob,
        ylabel="Unclassified probability",
        linestyle=['-', '--'],
        #marker=['o', 'd'],
        marker = [None]*2,
        save_fname=HERE / "figs/brc_refs_vs_log_unc_prob.png",
        ylog=True,
        show_legend=False,
    )

    _plot_all(
        ydata_dict=n_brs,
        ylabel="Number of disjoint events",
        linestyle=['-', '--'],
        #marker=['o', 'd'],
        marker = [None]*2,
        save_fname=HERE / "figs/brc_refs_vs_log_n_brs.png",
        ylog=True,
        show_legend=True,
    )

    _plot_all(
        ydata_dict=mem,
        ylabel="RSS (GB)",
        linestyle=['-', '--'],
        marker = [None]*2,
        save_fname=HERE / "figs/brc_refs_vs_mem.png",
    )

    _plot_all(
        ydata_dict=c_time,
        ylabel="Cumulative time for\nreference identification (hours)",
        linestyle=['-', '--'],
        marker = [None]*2,
        save_fname=HERE / "figs/brc_refs_vs_c_time.png",
    )

def draw_chen12_data() -> None:
    """
    Draw the computation statistics reported in Chen and Lin (2012)
    for their search for minimal paths.
    """

    HERE = Path(__file__).resolve().parent

    # Data from Chen & Lin (2012)
    n_edges = [17, 22, 24, 27, 31, 38, 40, 45, 49, 52, 58, 59, 60]
    n_mps = [
        38, 125, 184, 414, 976, 5382, 8512,
        29738, 79384, 163496, 752061, 896476, 1262816
    ]
    n_2power = [2**e for e in n_edges]

    # --- Figure / font parameters (journal-safe) ---
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    fig, ax = plt.subplots(figsize=(3.4, 3.0))

    # --- Plot ---
    ax.plot(
        n_edges,
        n_mps,
        label="Number of reference states",
        color="black",
        linestyle="--",
        linewidth=1.5,
    )
    """
    ax.plot(
        n_edges,
        n_2power,
        label="Component states ($2^N$)",
        color="red",
        linestyle="-",
        linewidth=1.5,
    )
    """
    # --- Labels ---
    ax.set_xlabel("Number of component variables")
    ax.set_ylabel("Number of reference states")

    ax.set_yscale("log")  # MP counts grow exponentially

    # --- Grid (subtle, print-safe) ---
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)

    fig.tight_layout()
    fig.savefig(HERE / "figs/chen12_mps.png", dpi=300, bbox_inches="tight", pad_inches=0.02)

def load_data_brc_rsr(
    data_paths: list,
    labels: list,
) -> tuple:
    """
    Load and return data for plotting.
    """

    if len(data_paths) != len(labels):
        raise ValueError("data_paths and labels must have the same length.")

    if not all(s.endswith("BRC") or s.endswith("TSUM") for s in labels):
        raise ValueError("All labels must end with 'BRC' or 'TSUM'.")

    sec_to_hr = 1.0 / 3600.0

    n_rules, unk_prob, mem, c_time = {}, {}, {}, {}
    p0_up, p0_low = {}, {}

    for path, label in zip(data_paths, labels):
        if label.endswith("BRC"):
            with open(path, "r") as f:
                data_ = json.load(f)

            n_rules[label] = data_["no_ra"]
            unk_prob[label] = data_["pr_bu"]
            mem[label] = data_["rss_gb"]
            c_time[label] = np.cumsum(data_["time"]) * sec_to_hr
            p0_up[label] = data_["pf_up"]
            p0_low[label] = data_["pf_low"]

        elif label.endswith("TSUM"):
            data_ = pd.read_json(path, lines=True)

            n_rules[label] = data_["round"].to_list()
            unk_prob[label] = data_["p_unknown"].to_list()
            mem[label] = data_["rss_gb"].to_list()
            #SAMPLE = 10**7 
            #samp_scale = SAMPLE / data_["n_sample_actual"]
            time_hr_cum = data_["time_sec"].cumsum() * sec_to_hr
            #c_time[label] = (samp_scale * time_hr_cum).to_list() 
            c_time[label] = time_hr_cum.to_list()
            p1_ = data_["p_upper"].to_list()
            p0_up[label] = [1.0-p for p in p1_]
            p0_low[label] = data_["p_lower"].to_list()

    return n_rules, unk_prob, mem, c_time, p0_up, p0_low

def draw_brc_vs_rsr_data(
    data_paths: list,
    labels: list,
) -> None:
    """
    Draws four separate figures on BRC monitor data:
    - number of reference states vs. unknown probability (with legend)
    - number of reference states vs. memory usage
    - number of reference states vs. cumulative hours
    - number of reference states vs. upper and lower bounds on failure probability
    """


    n_rules, unk_prob, mem, c_time, p0_up, p0_low = load_data_brc_rsr(data_paths, labels)


    def _plot_all(ydata_dict, ylabel, marker, linestyle, save_fname, show_legend=False, xlog=False, ylog=False, ylim=None):

        if len(ydata_dict.keys()) != len(labels):
            raise ValueError("ydata_dict and labels must have the same length.")

        # Figure parameters
        font_name = "Times New Roman"
        plt.rcParams["font.family"] = font_name
        fsz = 11
        lw = 1.2
        # ---

        fig = plt.figure(figsize=(3.4, 3.0), constrained_layout=True)
        ax = fig.add_axes([0.12, 0.08, 0.83, 0.83])

        for i, label in enumerate(labels):
            x = n_rules[label]
            y = ydata_dict[label]
            n = min(len(x), len(y))  # safety
            ax.plot(
                x[:n], y[:n],
                linestyle=linestyle[i], linewidth=lw, 
                marker=marker[i], markevery=10, markersize=2.5,
                label=label,
                color="#0072B2" if label.endswith("BRC") else "black",
            )

        ax.set_xlabel("Number of reference states", fontsize=fsz)
        ax.set_ylabel(ylabel, fontsize=fsz)
        ax.tick_params(axis="both", which="major", labelsize=fsz-1)
        ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)
        if ylog:
            ax.set_yscale("log")

        if xlog:
            ax.set_xscale("log")

        if show_legend:
            ax.legend(loc="best", frameon=True, fontsize=fsz-1.0, labelspacing=0.25, handletextpad=0.4, handlelength=1.1, borderpad=0.2,)

        # --- add memory limit line if plotting memory ---
        if ylabel.lower().startswith("rss"):
            y_limit = 20

            # horizontal line
            ax.axhline(
                y=y_limit,
                color="black",
                linestyle="-",
                linewidth=1.2,
            )

            # label near the right edge, slightly above the line
            ax.text(
                ax.get_xlim()[1]-10,           # xmax (data coordinates)
                y_limit * 1.01,              # a bit above the line 
                "Threshold",
                color="black",
                fontsize=fsz,
                ha="right",
                va="bottom",
            )
            ax.set_ylim(0, 20*1.5)  # ensure limit line is visible
            
        if ylim is not None:
            ax.set_ylim(ylim)

        plt.savefig(save_fname, dpi=300, bbox_inches="tight", pad_inches=0.02)


    # --- Figures ---
    HERE = Path(__file__).resolve().parent
    _plot_all(
        ydata_dict=unk_prob,
        ylabel="Unclassified probability",
        #marker=['o', 'd', 'x', '^'],
        marker = [None]*4,
        linestyle=['-', '--', '-', '--'],
        save_fname=HERE / "figs/brc_rsr_refs_vs_log_unc_prob.png",
        ylog=True,
        #xlog=True,
        show_legend=False,
    )

    _plot_all(
        ydata_dict=mem,
        ylabel="RSS (GB)",
        marker = [None]*4,
        linestyle=['-', '--', '-', '--'],
        save_fname=HERE / "figs/brc_rsr_refs_vs_mem.png",
        ylog=True,
    )

    _plot_all(
        ydata_dict=c_time,
        ylabel="Cumulative time for\nreference identification (hours)",
        marker = [None]*4,
        linestyle=['-', '--', '-', '--'],
        save_fname=HERE / "figs/brc_rsr_refs_vs_c_time.png",
        ylog=True,
        ylim=[5e-4, 1e3],
    )

    # --- Bounds ---
    plot_bounds_dual_y(
        p_low_L={k:v for k,v in p0_low.items() if k.startswith("Graph 1")},
        p_up_L ={k:v for k,v in p0_up.items()  if k.startswith("Graph 1")},
        ylabel_L="Bounds on $P(S\leq 0)$: Graph 1",

        p_low_R={k:v for k,v in p0_low.items() if k.startswith("Graph 2")},
        p_up_R ={k:v for k,v in p0_up.items()  if k.startswith("Graph 2")},
        ylabel_R="Graph 2",

        n_rules=n_rules,

        marker_L=[None]*2,
        linestyle_L=['-', '-'],

        marker_R=[None]*2,
        linestyle_R=['--', '--'],

        save_fname=HERE / "figs/brc_rsr_refs_vs_bounds_Phi1_Phi2_dualy.png",
        show_legend=True,
        ylog_L=True,
        ylog_R=True,
        #xlog =True,
    )

def plot_bounds_dual_y(
    *,
    p_low_L: dict,
    p_up_L: dict,
    ylabel_L: str,
    p_low_R: dict,
    p_up_R: dict,
    ylabel_R: str,
    n_rules: dict,
    marker_L: list,
    linestyle_L: list,
    marker_R: list,
    linestyle_R: list,
    save_fname,
    show_legend: bool = False,
    ylog_L: bool = False,
    ylog_R: bool = False,
    xlog: bool = False,
    markevery: int = 10,
):
    # ---- basic key checks
    keys_L = list(p_low_L.keys())
    keys_R = list(p_low_R.keys())

    if set(keys_L) != set(p_up_L.keys()):
        raise ValueError("Left: p_low_L and p_up_L must have the same keys.")
    if set(keys_R) != set(p_up_R.keys()):
        raise ValueError("Right: p_low_R and p_up_R must have the same keys.")

    if len(marker_L) < len(keys_L) or len(linestyle_L) < len(keys_L):
        raise ValueError("marker_L/linestyle_L shorter than number of left series.")
    if len(marker_R) < len(keys_R) or len(linestyle_R) < len(keys_R):
        raise ValueError("marker_R/linestyle_R shorter than number of right series.")

    # Figure parameters
    font_name = "Times New Roman"
    plt.rcParams["font.family"] = font_name
    fsz = 11

    fig, ax = plt.subplots(figsize=(3.6, 3.0), constrained_layout=True)
    ax2 = ax.twinx()

    # ---- left axis bounds
    for i, label in enumerate(keys_L):
        x = n_rules[label]
        y_low = p_low_L[label]
        y_up  = p_up_L[label]
        n = min(len(x), len(y_low), len(y_up))

        base_color = "#0072B2" if label.endswith("BRC") else "black"

        ax.plot(
            x[:n], y_low[:n],
            linestyle=linestyle_L[i], linewidth=1.2,
            marker=marker_L[i], markevery=markevery, markersize=2.5,
            color=base_color,
            label=f"{label}" if show_legend else None,
        )
        ax.plot(
            x[:n], y_up[:n],
            linestyle=linestyle_L[i], linewidth=1.2,
            marker=marker_L[i], markevery=markevery, markersize=2.5,
            color=base_color,
            label=None,
        )

    # ---- right axis bounds
    for i, label in enumerate(keys_R):
        x = n_rules[label]
        y_low = p_low_R[label]
        y_up  = p_up_R[label]
        n = min(len(x), len(y_low), len(y_up))

        base_color = "#0072B2" if label.endswith("BRC") else "black"

        ax2.plot(
            x[:n], y_low[:n],
            linestyle=linestyle_R[i], linewidth=1.0,
            marker=marker_R[i], markevery=markevery, markersize=2.5,
            color=base_color,
            label=f"{label}" if show_legend else None,
        )
        ax2.plot(
            x[:n], y_up[:n],
            linestyle=linestyle_R[i], linewidth=1.0,
            marker=marker_R[i], markevery=markevery, markersize=2.5,
            color=base_color,
            label=None,
        )

    # ---- axes formatting
    ax.set_xlabel("Number of reference states", fontsize=fsz)
    ax.set_ylabel(ylabel_L, fontsize=fsz)
    ax2.set_ylabel(ylabel_R, fontsize=fsz)
    ax.set_ylim(1e-3, 1e-1)
    ax2.set_ylim(1e-2, 1e-0)

    ax.tick_params(axis="both", which="major", labelsize=fsz-1)
    ax2.tick_params(axis="y", which="major", labelsize=fsz-1)

    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)

    if xlog:
        ax.set_xscale("log")
    if ylog_L:
        ax.set_yscale("log")
    if ylog_R:
        ax2.set_yscale("log")

    # optional combined legend (usually busy with 4 lines per label)
    if show_legend:
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(
            handles1 + handles2,
            labels1 + labels2,
            loc="upper right",
            #bbox_to_anchor=(0.5, 1.02),
            ncol=1,
            frameon=True,
            fontsize=fsz-2,
        )

    plt.savefig(save_fname, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

def draw_rsr_data(
    data_paths: list,
    labels: list,
) -> None:
    """
    Draws four separate figures on BRC monitor data:
    - number of reference states vs. unknown probability (with legend)
    - number of reference states vs. memory usage
    - number of reference states vs. cumulative hours
    """

    if len(data_paths) != len(labels):
        raise ValueError("data_paths and labels must have the same length.")

    sec_to_hr = 1.0 / 3600.0

    n_rules, unk_prob, mem, c_time = {}, {}, {}, {}
    p0_up, p0_low = {}, {}

    for path, label in zip(data_paths, labels):
        data_ = pd.read_json(path, lines=True)

        n_rules[label] = data_["round"].to_list()
        unk_prob[label] = data_["p_unknown"].to_list()
        mem[label] = data_["rss_gb"].to_list()
        #SAMPLE = 10**7 
        #samp_scale = SAMPLE / data_["n_sample_actual"]
        time_hr_cum = data_["time_sec"].cumsum() * sec_to_hr
        #c_time[label] = (samp_scale * time_hr_cum).to_list() 
        c_time[label] = time_hr_cum.to_list()
        p1_ = data_["p_upper"].to_list()
        p0_up[label] = [1.0-p for p in p1_]
        p0_low[label] = data_["p_lower"].to_list()

    def _plot_all(ydata_dict, ylabel, marker, markevery, linestyle, save_fname, show_legend=False, xlog=False, ylog=False, xlim=None, ylim=None):

        if len(ydata_dict.keys()) != len(labels):
            raise ValueError("ydata_dict and labels must have the same length.")

        # Figure parameters
        font_name = "Times New Roman"
        plt.rcParams["font.family"] = font_name
        fsz = 11
        lw = 1.2
        # ---

        fig = plt.figure(figsize = (2.2, 2.6), constrained_layout=True)
        ax = fig.add_axes([0.12, 0.08, 0.83, 0.83])

        if not ylabel.startswith("Unclassified"):
            for i, label in enumerate(labels):
                x = n_rules[label]
                y = ydata_dict[label]
                n = min(len(x), len(y))  # safety
                ax.plot(
                    x[:n], y[:n],
                    linestyle=linestyle[i], linewidth=lw, 
                    marker=marker[i], markevery=markevery[i], markersize=2.5,
                    label=label,
                    color="black" if label.endswith("OD") else "#E69F00",
                )
        else:            
            for i, label in enumerate(labels):
                if label.endswith("OD"):
                    thin = 1
                else:
                    thin = 10

                x = n_rules[label]
                y = ydata_dict[label]
                ax.plot(
                    x[::thin], y[::thin],
                    linestyle=linestyle[i], linewidth=lw, 
                    marker=marker[i], markevery=markevery[i], markersize=2.5,
                    label=label,
                    color="black" if label.endswith("OD") else "#E69F00",
                )

        ax.set_xlabel("Number of reference states", fontsize=fsz)
        ax.set_ylabel(ylabel, fontsize=fsz)
        ax.tick_params(axis="both", which="major", labelsize=fsz-1)
        ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)
        if ylog:
            ax.set_yscale("log")
        if xlog:
            ax.set_xscale("log")

        if show_legend:
            ax.legend(loc="upper left", 
                      bbox_to_anchor=(0.0, 0.94),  # ↓ smaller y = lower legend
                      frameon=True, fontsize=fsz-1.0, labelspacing=0.25, handletextpad=0.4, handlelength=1.1, borderpad=0.2,)


        # --- add memory limit line if plotting memory ---
        if ylabel.lower().startswith("rss"):
            y_limit = 20

            # horizontal line
            ax.axhline(
                y=y_limit,
                color="black",
                linestyle="-",
                linewidth=1.2,
            )

            # label near the right edge, slightly above the line
            ax.text(
                ax.get_xlim()[1]*0.55,           # xmax (data coordinates)
                y_limit * 1.01,              # a bit above the line 
                "Threshold",
                color="black",
                fontsize=fsz,
                ha="right",
                va="bottom",
            )
            ax.set_ylim(0, 20*1.5)  # ensure limit line is visible

        # --- add unclassified probability threshold line ---
        if ylabel.lower().startswith("unclassified"):
            y_thresh = 1e-5

            ax.axhline(
                y=y_thresh,
                color="black",
                linestyle="-",
                linewidth=1.2,
            )

            # label near the right edge, slightly below the line
            ax.text(
                ax.get_xlim()[1] * 0.55,     # near xmax (data coords)
                y_thresh / 1.3,              # slightly below (log-safe)
                r"Threshold",
                color="black",
                fontsize=fsz,
                ha="right",
                va="top",
            )

            # ensure the threshold is visible
            if ax.get_yscale() == "log":
                ax.set_ylim(bottom=min(ax.get_ylim()[0], y_thresh / 10))
            else:
                ax.set_ylim(bottom=0)

        if ylim is not None:
            ax.set_ylim(ylim)
        if xlim is not None:
            ax.set_xlim(xlim)

        plt.savefig(save_fname, dpi=300, bbox_inches="tight", pad_inches=0.02)


    # --- Figures ---
    HERE = Path(__file__).resolve().parent
    _plot_all(
        ydata_dict=unk_prob,
        ylabel="Unclassified probability",
        #marker=['x', '^', 'x', '^'],
        linestyle=['-', '--', '-', '--'],
        markevery=[10, 20, 50, 50],
        marker = [None]*4,
        save_fname=HERE / "figs/rsr_refs_vs_log_unc_prob.png",
        ylog=True,
        xlog=True,
        show_legend=False,
        xlim=[1, 1e4+100]
    )

    _plot_all(
        ydata_dict=mem,
        ylabel="RSS (GB)",
        #marker=['x', '^', 'x', '^'],
        linestyle=['-', '--', '-', '--'],
        markevery=[20, 20, 500, 500],
        marker = [None]*4,
        save_fname=HERE / "figs/rsr_refs_vs_mem.png",
        show_legend=True,
        ylog=True,
        xlog=True,
        xlim=[1, 1e4+100],
        ylim=[0.1, 20*1.5]
    )

    _plot_all(
        ydata_dict=c_time,
        ylabel="Cumulative time for\nreference identification (hours)",
        #marker=['x', '^', 'x', '^'],
        linestyle=['-', '--', '-', '--'],
        markevery=[20, 20, 500, 500],
        marker = [None]*4,
        save_fname=HERE / "figs/rsr_refs_vs_c_time.png",
        ylog=True,
        ylim=[5e-4, 1e3],
        show_legend=False,
        xlog=True,
        xlim=[1, 1e4+100]
    )

def get_results_summary(
    data_paths: list,
    labels: list,
    refs_paths: list,
    graph_paths: list,
    save_path: Path,
):
    
    n_rules, unk_prob, mem, c_time, p0_up, p0_low = load_data_brc_rsr(data_paths, labels)

    n_rules_summary = [n_rules[k][-1] for k in labels]
    unk_prob_summary = [unk_prob[k][-1] for k in labels]
    mem_summary = [mem[k][-1] for k in labels]
    ref_find_time_summary = [c_time[k][-1] for k in labels]
    pf_summary = [0.5*(p0_up[k][-1]+p0_low[k][-1]) for k in labels]

    # system probability evaluation
    sys_prob_time_summary = []
    for l, r, g in zip(labels, refs_paths, graph_paths):
        probs_dict = json.loads((g / "probs.json").read_text(encoding="utf-8"))

        if l.endswith("BRC"):
            varis, cpms = {}, {}
            for k, v in probs_dict.items():
                pf_k = v['0']['p']
                varis[k] = variable.Variable(name=k, values=['f', 's'])
                cpms[k] = cpm.Cpm(
                    variables=[varis[k]],
                    no_child = 1,
                    C = np.array([[0], [1]]),
                    p = np.array([pf_k, 1.0-pf_k]))
                
            brs = branch.load_brs_from_parquet(r)
            Csys = brc.get_csys(brs, varis, {'f': 0, 's': 1, 'u': 2})
            psys = np.array([1.0]*len(Csys))

            varis['sys'] = variable.Variable(name='sys', values=['f', 's'])
            cpms['sys'] = cpm.Cpm(
                variables=[varis['sys']]+[varis[k] for k in probs_dict.keys()],
                no_child = 1,
                C = Csys, p=psys)
            
            st = time.time()
            Msys = inference.variable_elim(cpms, [varis[k] for k in probs_dict.keys()], varis['sys'])
            end = time.time()
            print(f"{l} evaluation time: {end-st:.2f} seconds:")
            print(Msys)

        else: 
            nodes = json.loads( (g / "nodes.json").read_text(encoding="utf-8") )
            edges = json.loads((g / "edges.json").read_text(encoding="utf-8"))
            probs_dict = json.loads((g / "probs.json").read_text(encoding="utf-8"))

            row_names = list(edges.keys())
            n_state = 2 # binary states

            probs = [[probs_dict[n]['0']['p'], probs_dict[n]['1']['p']] for n in row_names]
            probs = torch.tensor(probs, dtype=torch.float32, device=device)

            G_ = build_graph(nodes, edges, probs_dict)

            if "1 OD" in l:
                ODs_ = [n for n,v in nodes.items() if v['is_od']]
                s_fun = lambda comps_st: eval_1od_connectivity(comps_st, G_, ODs_[0], ODs_[1])
            else: 
                def sys_func_global_conn_long(comps_st):
                    _, k, _ = eval_global_conn_k(comps_st, G_)
                    if k >= 1:
                        sys_st = 1
                    else:
                        sys_st = 0
                    return k, sys_st, None
                s_fun = lambda comps_st: sys_func_global_conn_long(comps_st)
                                
            rules_mat_upper = torch.load(r / f"rules_geq_1.pt", map_location="cpu")
            rules_mat_upper = rules_mat_upper.to(device)
            rules_mat_lower = torch.load(r / f"rules_leq_0.pt", map_location="cpu")
            rules_mat_lower = rules_mat_lower.to(device)

            print(f"Evaluating system probability for {l}...")
            st = time.time()
            psys = rsr.get_comp_cond_sys_prob(
                        rules_mat_upper,
                        rules_mat_lower,
                        probs,
                        comps_st_cond = {},
                        row_names = row_names,
                        s_fun = s_fun,
                        n_sample=int(10**6),
                        n_batch = 250_000,
                        sys_upper_st = 1
                    )
            end = time.time()
            print(f"{l} evaluation time: {end-st:.2f} seconds:")
            print(f"P(sys >= 1) = {psys['upper']:.3e}")
            print(f"P(sys <= 0) = {psys['lower']:.3e}\n")

        sys_prob_time_summary.append(end-st)

    summary = pd.DataFrame({
        "Label": labels,
        "Final number of reference states": n_rules_summary,
        "Final unclassified probability": unk_prob_summary,
        "Final memory usage (GB)": mem_summary,
        "Final reference finding time (hours)": ref_find_time_summary,
        "P(sys=0) estimate": pf_summary,
        "System probability evaluation time (seconds)": sys_prob_time_summary,
    })

    print(summary.to_string(index=False))
    summary.to_csv(save_path, index=False)

def get_direct_results_summary(
    data_paths: list,
    labels: list,
    save_path: Path,
):
    
    n_rules, unk_prob, mem, c_time, p0_up, p0_low = load_data_brc_rsr(data_paths, labels)

    threshold = 1e-5 # for fair comparison, truncate the data unk_prob >= threshold

    n_rules_summary = []
    unk_prob_summary = []
    mem_summary = []
    ref_find_time_summary = []
    pf_summary = []

    for k in labels:
        # find indices where unk_prob >= threshold
        valid_idx = [i for i, v in enumerate(unk_prob[k]) if v >= threshold]

        if not valid_idx:
            continue  # or raise an error if this shouldn't happen

        last_i = valid_idx[-1]

        n_rules_summary.append(n_rules[k][last_i])
        unk_prob_summary.append(unk_prob[k][last_i])
        mem_summary.append(mem[k][last_i])
        ref_find_time_summary.append(c_time[k][last_i])
        pf_summary.append(0.5 * (p0_up[k][last_i] + p0_low[k][last_i]))

    summary = pd.DataFrame({
        "Label": labels,
        "Number of reference states at threshold": n_rules_summary,
        "Unclassified probability at threshold": unk_prob_summary,
        "Memory usage at threshold (GB)": mem_summary,
        "Reference finding time at threshold (hours)": ref_find_time_summary,
        "P(sys=0) estimate at threshold": pf_summary,
    })

    print(summary.to_string(index=False))
    summary.to_csv(save_path, index=False)

def get_results_summary_rsr_unc(
    data_paths: list,
    labels: list,
    refs_paths: list,
    graph_paths: list,
    save_path: Path,
):
    """
    Allow for TSUM to return unknown probability > 0
    """
    
    n_rules, unk_prob, mem, c_time, p0_up, p0_low = load_data_brc_rsr(data_paths, labels)

    n_rules_summary = [n_rules[k][-1] for k in labels]
    unk_prob_summary = [unk_prob[k][-1] for k in labels]
    mem_summary = [mem[k][-1] for k in labels]
    ref_find_time_summary = [c_time[k][-1] for k in labels]
    pf_summary = [0.5*(p0_up[k][-1]+p0_low[k][-1]) for k in labels]

    # system probability evaluation
    sys_prob_time_summary = []
    for l, r, g in zip(labels, refs_paths, graph_paths):
        probs_dict = json.loads((g / "probs.json").read_text(encoding="utf-8"))

        if l.endswith("BRC"):
            varis, cpms = {}, {}
            for k, v in probs_dict.items():
                pf_k = v['0']['p']
                varis[k] = variable.Variable(name=k, values=['f', 's'])
                cpms[k] = cpm.Cpm(
                    variables=[varis[k]],
                    no_child = 1,
                    C = np.array([[0], [1]]),
                    p = np.array([pf_k, 1.0-pf_k]))
                
            brs = branch.load_brs_from_parquet(r)
            Csys = brc.get_csys(brs, varis, sys_st = {'f': 0, 's': 1, 'u': 2})
            psys = np.array([1.0]*len(Csys))

            varis['sys'] = variable.Variable(name='sys', values=['f', 's'])
            cpms['sys'] = cpm.Cpm(
                variables=[varis['sys']]+[varis[k] for k in probs_dict.keys()],
                no_child = 1,
                C = Csys, p=psys)
            
            st = time.time()
            Msys = inference.variable_elim(cpms, [varis[k] for k in probs_dict.keys()], varis['sys'])
            end = time.time()
            print(f"{l} evaluation time: {end-st:.2f} seconds:")
            print(Msys)

        else: 
            nodes = json.loads( (g / "nodes.json").read_text(encoding="utf-8") )
            edges = json.loads((g / "edges.json").read_text(encoding="utf-8"))
            probs_dict = json.loads((g / "probs.json").read_text(encoding="utf-8"))

            row_names = list(edges.keys())
            n_state = 2 # binary states

            probs = [[probs_dict[n]['0']['p'], probs_dict[n]['1']['p']] for n in row_names]
            probs = torch.tensor(probs, dtype=torch.float32, device=device)
                                
            rules_mat_upper = torch.load(r / f"rules_geq_1.pt", map_location="cpu")
            rules_mat_upper = rules_mat_upper.to(device)
            rules_mat_lower = torch.load(r / f"rules_leq_0.pt", map_location="cpu")
            rules_mat_lower = rules_mat_lower.to(device)

            print(f"Evaluating system probability for {l}...")
            st = time.time()
            psys = rsr.get_comp_cond_sys_prob(
                        rules_mat_upper,
                        rules_mat_lower,
                        probs,
                        comps_st_cond = {},
                        row_names = row_names,
                        s_fun = None,
                        n_sample=int(10**6),
                        n_batch = 250_000,
                        sys_upper_st = 1
                    )
            end = time.time()
            print(f"{l} evaluation time: {end-st:.2f} seconds:")
            print(f"P(sys >= 1) = {psys['upper']:.3e}")
            print(f"P(sys <= 0) = {psys['lower']:.3e}\n")

        sys_prob_time_summary.append(end-st)

    summary = pd.DataFrame({
        "Label": labels,
        "Final number of reference states": n_rules_summary,
        "Final unclassified probability": unk_prob_summary,
        "Final memory usage (GB)": mem_summary,
        "Final reference finding time (hours)": ref_find_time_summary,
        "P(sys=0) estimate": pf_summary,
        "System probability evaluation time (seconds)": sys_prob_time_summary,
    })

    print(summary.to_string(index=False))
    summary.to_csv(save_path, index=False)

if __name__ == "__main__":

    HERE = Path(__file__).resolve().parent

    # --- Draw random graphs from JSON ---
    """
    for rg, leg in zip(
        ['rg1', 'rg2', 'rg3', 'rg4'],
        [True, False, True, False],  
    ):
        draw_graph_from_json(
            nodes_path=HERE.parent / f"results/{rg}/v1/data/nodes.json",
            edges_path=HERE.parent / f"results/{rg}/v1/data/edges.json",
            save_fname=HERE / f"figs/{rg}_graph.png",
            show_legend=leg,
        )
    """

    # --- BRC monitor data ---
    #"""
    draw_brc_monitor_data(
        data_paths=[
            HERE.parent / "results/rg1/v1/brc/monitor_conn.json",
            HERE.parent / "results/rg2/v1/brc/monitor_conn.json",
        ],
        labels=[
            "Graph 1",
            "Graph 2",
        ]
    )
    #"""

    # --- Chen & Lin (2012) data ---
    #draw_chen12_data()

    # --- BRC vs. TSUM ---
    """
    draw_brc_vs_rsr_data(
        data_paths=[
            HERE.parent / "results/rg1/v1/brc/monitor_conn.json",
            HERE.parent / "results/rg2/v1/brc/monitor_conn.json",
            HERE.parent / "results/rg1/v1/rsr_conn/metrics.jsonl",
            HERE.parent / "results/rg2/v1/rsr_conn/metrics.jsonl",
        ],
        labels = [
            "Graph 1; BRC",
            "Graph 2; BRC",
            "Graph 1; TSUM",
            "Graph 2; TSUM",
        ]
    )
    """

    # ---TSUM: 1 OD conn vs g-conn ---
    #"""
    draw_rsr_data(
        data_paths=[
            HERE.parent / "results/rg1/v1/rsr_conn/metrics.jsonl",
            HERE.parent / "results/rg2/v1/rsr_conn/metrics.jsonl",
            HERE.parent / "results/rg1/v1/rsr_global_conn/metrics.jsonl",
            HERE.parent / "results/rg2/v1/rsr_global_conn/metrics.jsonl",
        ],
        labels = [
            "Graph 1; 1 OD",
            "Graph 2; 1 OD",
            "Graph 1; Global",
            "Graph 2; Global",
        ]
    )
    #"""

    # --- Get results summary table ---
    """
    get_results_summary(
        data_paths=[
            #HERE.parent / "results/rg2/v1/rsr_conn/metrics.jsonl",
            #HERE.parent / "results/rg1/v1/rsr_conn/metrics.jsonl",
            #HERE.parent / "results/rg1/v1/rsr_global_conn/metrics.jsonl",
            #HERE.parent / "results/rg2/v1/rsr_global_conn/metrics.jsonl",
            HERE.parent / "results/rg1/v1/brc/monitor_conn.json",
            HERE.parent / "results/rg2/v1/brc/monitor_conn.json",
        ],
        labels=[
            #"Graph 2; 1 OD; TSUM",
            #"Graph 1; 1 OD; TSUM",
            #"Graph 1; Global; TSUM",
            #"Graph 2; Global; TSUM",
            "Graph 1; 1 OD; BRC",
            "Graph 2; 1 OD; BRC",
        ], 
        refs_paths=[
            #HERE.parent / "results/rg2/v1/rsr_conn",
            #HERE.parent / "results/rg1/v1/rsr_conn",
            #HERE.parent / "results/rg1/v1/rsr_global_conn",
            #HERE.parent / "results/rg2/v1/rsr_global_conn",
            HERE.parent / "results/rg1/v1/brc/brs_conn.parquet",
            HERE.parent / "results/rg2/v1/brc/brs_conn.parquet",
        ],
        graph_paths=[
            #HERE.parent / f"results/rg2/v1/data",
            #HERE.parent / f"results/rg1/v1/data",
            #HERE.parent / f"results/rg1/v1/data",
            #HERE.parent / f"results/rg2/v1/data",
            HERE.parent / f"results/rg1/v1/data",
            HERE.parent / f"results/rg2/v1/data",
        ],
        #save_path=HERE / f"figs/results_summary.csv",
        save_path=HERE / f"figs/results_summary_brc_only.csv",
    )
    """

    # --- Get direct results summary (no sys prob eval) ---
    """
    get_direct_results_summary(
        data_paths=[
            HERE.parent / "results/rg1/v1/brc/monitor_conn.json",
            HERE.parent / "results/rg1/v1/rsr_conn/metrics.jsonl",
            HERE.parent / "results/rg1/v1/rsr_global_conn/metrics.jsonl",
            HERE.parent / "results/rg2/v1/brc/monitor_conn.json",
            HERE.parent / "results/rg2/v1/rsr_conn/metrics.jsonl",
            HERE.parent / "results/rg2/v1/rsr_global_conn/metrics.jsonl",            
        ],
        labels=[
            "Graph 1; 1 OD; BRC",
            "Graph 1; 1 OD; TSUM",
            "Graph 1; Global; TSUM",
            "Graph 2; 1 OD; BRC",
            "Graph 2; 1 OD; TSUM",
            "Graph 2; Global; TSUM",
        ], 
        save_path=HERE / f"figs/direct_results_summary.csv",
    )
    """

    # --- Get results summary allowing for TSUM to have unclassified samples ---
    """
    get_results_summary_rsr_unc(
        data_paths=[
            HERE.parent / "results/rg2/v1/rsr_global_conn/metrics.jsonl",
        ],
        labels=[
            "Graph 2; Global; TSUM",
        ], 
        refs_paths=[
            HERE.parent / "results/rg2/v1/rsr_global_conn",
        ],
        graph_paths=[
            HERE.parent / f"results/rg2/v1/data",
        ],
        save_path=HERE / f"figs/results_summary_unc.csv",
    )
    """