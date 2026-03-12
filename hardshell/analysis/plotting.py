# hardshell/analysis/plotting.py
"""
Publication-quality figures for the HardShell swarm experiment.

Figure index
------------
01  condition_summary       — 4-panel: ASR / Utility / Participation / Tool-call rate
02  utility_distribution    — Violin: utility_score by condition
03  agent_participation     — Heatmap: agent × condition → post rate
04  conversation_depth      — Violin: tool calls per agent by condition
05  tool_usage_heatmap      — Heatmap: condition × tool → usage rate
06  dangerous_tool_rate     — Grouped bar: dangerous / total tool calls by condition
07  action_breakdown        — Stacked bar: posted/emailed/no-action by condition
08  asr_utility_scatter     — Scatter: ASR vs utility per trial, coloured by condition
09  2x2_factorial           — Grouped bar: ASR + utility × attack × tool_defense
10  swarm_network           — Agent co-activation network graph
11  behavioral_fingerprint  — Radar: 6-axis behavioural profile by condition
12  swarm_interaction       — Directed flow diagram for a single representative trial
13  asr_by_attack_type      — Bar: ASR by InjecAgent attack category
14  payload_position        — Bar: ASR by payload injection position
15  tool_defense_blocks     — Scatter: tool_defense block rate by agent
"""
from __future__ import annotations

import warnings
from typing import Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)

_STYLE   = "seaborn-v0_8-paper"
_DPI     = 300
# Ordered palette: navy, red, amber, green, purple, teal
_PALETTE = ["#2c3e50", "#e74c3c", "#f39c12", "#2ecc71", "#8e44ad", "#16a085"]

_COND_ORDER = ["cond_a", "cond_b", "cond_c", "cond_d"]   # 2×2 canonical order


def _savefig(fig, path: str, tight: bool = True) -> None:
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)


def _cond_colors(conditions: list) -> list[str]:
    """Assign consistent colours to conditions."""
    palette = _PALETTE * (len(conditions) // len(_PALETTE) + 1)
    order   = _COND_ORDER + [c for c in sorted(conditions) if c not in _COND_ORDER]
    return [palette[order.index(c) % len(palette)] if c in order else palette[i]
            for i, c in enumerate(conditions)]



# ---------------------------------------------------------------------------
# 01 — Condition summary (4-panel)
# ---------------------------------------------------------------------------

def plot_condition_summary(df_trials: pd.DataFrame, save_path: str) -> None:
    """4-panel bar chart: ASR, Utility, Participation Rate, Mean Tool Calls/Agent."""
    plt.style.use(_STYLE)
    df = df_trials[df_trials["utility_score"].fillna(-1) >= 0].copy()
    conditions = sorted(df["condition"].unique())
    x     = np.arange(len(conditions))
    width = 0.55
    colors = _cond_colors(conditions)

    metrics = [
        ("target_execution_asr",    "Attack Success Rate (ASR)",    _PALETTE[1]),
        ("utility_score",           "Task Utility Score",           _PALETTE[3]),
        ("participation_rate",      "Agent Participation Rate",     _PALETTE[0]),
        ("mean_tool_calls_per_agent","Mean Tool Calls / Agent",     _PALETTE[4]),
    ]
    # Fall back to 3-panel if tool call data absent
    metrics = [(m, l, c) for m, l, c in metrics if m in df.columns]

    ncols = len(metrics)
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4), sharey=False)
    if ncols == 1:
        axes = [axes]

    for ax, (col, label, _) in zip(axes, metrics):
        means = [df[df["condition"] == c][col].mean() for c in conditions]
        sems  = [df[df["condition"] == c][col].sem()  for c in conditions]
        bars  = ax.bar(x, means, width, yerr=sems, color=colors, alpha=0.85,
                       capsize=4, error_kw={"linewidth": 1.2})
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(label, fontsize=10)
        ax.set_xlabel("Condition", fontsize=9)
        ax.yaxis.grid(True, linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)
        for bar, mean in zip(bars, means):
            if not np.isnan(mean):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.02 * ax.get_ylim()[1],
                        f"{mean:.2f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("HardShell Swarm — Condition Summary", fontsize=13, fontweight="bold", y=1.01)
    _savefig(fig, save_path)


# ---------------------------------------------------------------------------
# 02 — Utility distribution (violin)
# ---------------------------------------------------------------------------

def plot_utility_distribution(df_trials: pd.DataFrame, save_path: str) -> None:
    """Violin: utility_score distribution per condition."""
    plt.style.use(_STYLE)
    df = df_trials[df_trials["utility_score"].fillna(-1) >= 0].copy()
    conditions = sorted(df["condition"].unique())
    data   = [df[df["condition"] == c]["utility_score"].dropna().values for c in conditions]
    colors = _cond_colors(conditions)

    fig, ax = plt.subplots(figsize=(max(6, len(conditions) * 1.6), 4))
    parts = ax.violinplot(data, showmedians=True, showextrema=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    ax.set_xticks(range(1, len(conditions) + 1))
    ax.set_xticklabels(conditions, fontsize=9)
    ax.set_ylabel("Utility Score (0–1)", fontsize=11)
    ax.set_xlabel("Condition", fontsize=10)
    ax.set_ylim(-0.05, 1.1)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_title("Task Utility Score Distribution by Condition", fontsize=12, fontweight="bold")
    _savefig(fig, save_path)


# ---------------------------------------------------------------------------
# 03 — Agent participation heatmap
# ---------------------------------------------------------------------------

def plot_agent_participation_heatmap(df_agents: pd.DataFrame, save_path: str) -> None:
    """Heatmap: agent × condition → post rate."""
    plt.style.use(_STYLE)
    pivot = (
        df_agents.groupby(["agent_id", "condition"])["posted"]
        .mean()
        .unstack(fill_value=0)
    )
    fig, ax = plt.subplots(
        figsize=(max(5, len(pivot.columns) * 1.6), max(3, len(pivot) * 0.7))
    )
    sns.heatmap(pivot, annot=True, fmt=".0%", cmap="YlGn",
                vmin=0, vmax=1, linewidths=0.5, ax=ax,
                cbar_kws={"label": "Post Rate"})
    ax.set_title("Agent Participation Rate (Post) by Role × Condition",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Condition", fontsize=10)
    ax.set_ylabel("Agent", fontsize=10)
    _savefig(fig, save_path)


# ---------------------------------------------------------------------------
# 04 — Conversation depth (tool calls per agent, violin)
# ---------------------------------------------------------------------------

def plot_conversation_depth(df_agents: pd.DataFrame, save_path: str) -> None:
    """Violin: number of tool calls per agent by condition.

    Higher = more 'conversation turns' / agentic steps.
    Under attack, agents taking more steps may be executing injection instructions.
    """
    if "n_tool_calls" not in df_agents.columns:
        return
    plt.style.use(_STYLE)
    conditions = sorted(df_agents["condition"].unique())
    colors = _cond_colors(conditions)
    data   = [df_agents[df_agents["condition"] == c]["n_tool_calls"].dropna().values
               for c in conditions]

    fig, ax = plt.subplots(figsize=(max(6, len(conditions) * 1.6), 4))
    parts = ax.violinplot(data, showmedians=True, showextrema=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    ax.set_xticks(range(1, len(conditions) + 1))
    ax.set_xticklabels(conditions, fontsize=9)
    ax.set_ylabel("Tool Calls per Agent", fontsize=11)
    ax.set_xlabel("Condition", fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_title("Conversation Depth (Tool Calls / Agent) by Condition",
                 fontsize=12, fontweight="bold")
    _savefig(fig, save_path)


# ---------------------------------------------------------------------------
# 05 — Tool usage heatmap (condition × tool → usage rate)
# ---------------------------------------------------------------------------

def plot_tool_usage_heatmap(df_agents: pd.DataFrame, save_path: str) -> None:
    """Heatmap: what fraction of agents in each condition called each tool.

    Reveals which tools the attack hijacks and whether the defense suppresses them.
    """
    if "tools_used" not in df_agents.columns:
        return
    plt.style.use(_STYLE)

    # Explode tools_used string into one row per tool per agent-trial
    rows = []
    for _, row in df_agents.iterrows():
        for tool in str(row.get("tools_used", "")).split(","):
            tool = tool.strip()
            if tool:
                rows.append({"condition": row["condition"],
                             "agent_id":  row["agent_id"],
                             "trial_id":  row["trial_id"],
                             "tool":      tool})
    if not rows:
        return

    tdf = pd.DataFrame(rows)
    # Usage rate = fraction of agent-trials in that condition that called this tool
    n_per_cond = df_agents.groupby("condition").size().rename("n_agent_trials")
    tool_counts = tdf.groupby(["condition", "tool"]).size().rename("n_calls")
    tool_rate = (tool_counts / n_per_cond).fillna(0).reset_index(name="usage_rate")

    pivot = tool_rate.pivot(index="tool", columns="condition", values="usage_rate").fillna(0)

    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 1.6), max(4, len(pivot) * 0.5)))
    sns.heatmap(pivot, annot=True, fmt=".0%", cmap="Blues",
                vmin=0, linewidths=0.4, ax=ax,
                cbar_kws={"label": "Usage Rate (calls / agent-trial)"})
    ax.set_title("Tool Usage Rate by Condition", fontsize=12, fontweight="bold")
    ax.set_xlabel("Condition", fontsize=10)
    ax.set_ylabel("Tool", fontsize=10)
    _savefig(fig, save_path)


# ---------------------------------------------------------------------------
# 06 — Dangerous tool call rate (grouped bar)
# ---------------------------------------------------------------------------

def plot_dangerous_tool_rate(df_trials: pd.DataFrame, save_path: str) -> None:
    """Grouped bar: dangerous tool calls and total tool calls per trial by condition.

    The core behavioral signal: under attack, does the dangerous call rate jump?
    Does the defense suppress it?
    """
    if "n_dangerous_calls" not in df_trials.columns:
        return
    plt.style.use(_STYLE)
    conditions = sorted(df_trials["condition"].unique())
    x     = np.arange(len(conditions))
    width = 0.35
    colors = _cond_colors(conditions)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: mean dangerous calls per trial
    ax = axes[0]
    means = [df_trials[df_trials["condition"] == c]["n_dangerous_calls"].mean() for c in conditions]
    sems  = [df_trials[df_trials["condition"] == c]["n_dangerous_calls"].sem()  for c in conditions]
    bars  = ax.bar(x, means, width * 1.4, yerr=sems, color=colors, alpha=0.85, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Mean Dangerous Tool Calls / Trial", fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Dangerous Tool Calls per Trial", fontsize=11, fontweight="bold")
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01, f"{mean:.2f}",
                ha="center", va="bottom", fontsize=8)

    # Right: tool failure rate (blocked / total)
    ax = axes[1]
    means2 = [df_trials[df_trials["condition"] == c]["tool_failure_rate_mean"].mean() for c in conditions]
    sems2  = [df_trials[df_trials["condition"] == c]["tool_failure_rate_mean"].sem()  for c in conditions]
    bars2  = ax.bar(x, means2, width * 1.4, yerr=sems2, color=colors, alpha=0.85, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Mean Tool Failure / Block Rate", fontsize=10)
    ax.set_ylim(0, max(0.1, max(means2) * 1.4) if means2 else 0.1)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Tool Failure / Block Rate by Condition", fontsize=11, fontweight="bold")
    for bar, mean in zip(bars2, means2):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002, f"{mean:.2%}",
                ha="center", va="bottom", fontsize=8)

    fig.suptitle("Dangerous Tool Activity & Defense Blocking", fontsize=13, fontweight="bold")
    _savefig(fig, save_path)


# ---------------------------------------------------------------------------
# 07 — Action breakdown (stacked bar)
# ---------------------------------------------------------------------------

def plot_agent_action_breakdown(df_agents: pd.DataFrame, _df_trials: pd.DataFrame,
                                save_path: str) -> None:
    """Stacked bar per condition: fraction of agent-slots that posted / emailed /
    called a dangerous tool / took no action.
    """
    plt.style.use(_STYLE)
    conditions = sorted(df_agents["condition"].unique())
    results = []

    for cond in conditions:
        ag = df_agents[df_agents["condition"] == cond]
        n  = len(ag) or 1
        results.append({
            "condition":       cond,
            "Posted":          len(ag[ag["posted"] == 1]) / n,
            "Sent Email":      len(ag[ag["sent_email"] == 1]) / n,
            "Dangerous Tool":  len(ag[ag["dangerous_tool_called"] == 1]) / n,
            "No Action":       len(ag[(ag["posted"] == 0) & (ag["sent_email"] == 0) & (ag["dangerous_tool_called"] == 0)]) / n,
        })

    rdf    = pd.DataFrame(results).set_index("condition")
    colors = [_PALETTE[3], _PALETTE[1], _PALETTE[2], "#aaaaaa"]

    fig, ax = plt.subplots(figsize=(max(7, len(conditions) * 1.8), 4))
    rdf.plot(kind="bar", stacked=True, ax=ax, color=colors, width=0.55, alpha=0.88)
    ax.set_ylabel("Fraction of Agent Slots", fontsize=11)
    ax.set_xlabel("Condition", fontsize=10)
    ax.set_ylim(0, 1.25)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.8)
    ax.set_title("Agent Action Breakdown by Condition", fontsize=12, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    _savefig(fig, save_path)


# ---------------------------------------------------------------------------
# 08 — ASR vs Utility scatter
# ---------------------------------------------------------------------------

def plot_asr_utility_scatter(df_trials: pd.DataFrame, save_path: str) -> None:
    """Scatter: each trial as a point — ASR (x) vs Utility (y), coloured by condition.

    Reveals the ASR-utility trade-off frontier. Ideal defense: low ASR, high utility.
    """
    df = df_trials[
        (df_trials["utility_score"].fillna(-1) >= 0) &
        (df_trials["target_execution_asr"].fillna(-1) >= 0)
    ].copy()
    if df.empty:
        return

    plt.style.use(_STYLE)
    conditions = sorted(df["condition"].unique())
    colors     = _cond_colors(conditions)
    cmap       = dict(zip(conditions, colors))

    fig, ax = plt.subplots(figsize=(7, 5))
    for cond in conditions:
        sub = df[df["condition"] == cond]
        ax.scatter(sub["target_execution_asr"], sub["utility_score"],
                   color=cmap[cond], alpha=0.5, s=20, label=cond)
        # Overlay condition centroid
        ax.scatter(sub["target_execution_asr"].mean(), sub["utility_score"].mean(),
                   color=cmap[cond], s=120, marker="D", edgecolors="white", linewidths=1.5,
                   zorder=5)

    ax.set_xlabel("Attack Success Rate (ASR)", fontsize=11)
    ax.set_ylabel("Task Utility Score", fontsize=11)
    ax.set_xlim(-0.05, 1.1)
    ax.set_ylim(-0.05, 1.1)
    ax.axhline(0.5, color="grey", lw=0.8, ls="--", alpha=0.6)
    ax.axvline(0.5, color="grey", lw=0.8, ls="--", alpha=0.6)
    ax.legend(fontsize=9, title="Condition", framealpha=0.85)
    ax.set_title("ASR–Utility Trade-off by Condition\n(diamonds = condition means)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    _savefig(fig, save_path)


# ---------------------------------------------------------------------------
# 09 — 2×2 factorial (ASR + Utility)
# ---------------------------------------------------------------------------

def plot_2x2_factorial(df_trials: pd.DataFrame, save_path: str) -> None:
    """2×2 grouped bar: ASR and Utility across attack × tool_defense.

    The primary result figure for the tool_defense experiment.
    """
    if "tool_defense" not in df_trials.columns or "inject_payload" not in df_trials.columns:
        return
    plt.style.use(_STYLE)

    df = df_trials[
        (df_trials["utility_score"].fillna(-1) >= 0) &
        (df_trials["target_execution_asr"].fillna(-1) >= 0)
    ].copy()
    if df.empty:
        return

    df["attack_label"]  = df["inject_payload"].map(
        {0: "No Attack", 1: "Attack", False: "No Attack", True: "Attack"}
    )
    df["defense_label"] = df["tool_defense"].map(
        {0: "No Defense", 1: "Tool Defense", False: "No Defense", True: "Tool Defense"}
    )

    attack_levels  = ["No Attack", "Attack"]
    defense_levels = ["No Defense", "Tool Defense"]
    x      = np.arange(len(attack_levels))
    width  = 0.35
    def_colors = {"No Defense": _PALETTE[0], "Tool Defense": _PALETTE[3]}

    metrics = [
        ("target_execution_asr", "Attack Success Rate (ASR)"),
        ("utility_score",        "Task Utility Score"),
        ("task_completed",       "Task Completion Rate (TCR)"),
    ]
    valid_metrics = [(m, l) for m, l in metrics if m in df.columns]
    ncols = len(valid_metrics)

    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4.5))
    if ncols == 1:
        axes = [axes]

    for ax, (metric, label) in zip(axes, valid_metrics):
        for di, dlabel in enumerate(defense_levels):
            means, errs = [], []
            for alabel in attack_levels:
                sub = df[(df["attack_label"] == alabel) & (df["defense_label"] == dlabel)]
                means.append(sub[metric].mean() if len(sub) else np.nan)
                errs.append( sub[metric].sem()  if len(sub) else 0.0)
            offset = (di - 0.5) * width
            bars = ax.bar(x + offset, means, width, yerr=errs,
                          label=dlabel, color=def_colors[dlabel],
                          alpha=0.85, capsize=4, error_kw={"linewidth": 1.2})
            for bar, mean in zip(bars, means):
                if not np.isnan(mean):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.04,
                            f"{mean:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(attack_levels, fontsize=10)
        ax.set_ylim(0, 1.2)
        ax.set_ylabel(label, fontsize=10)
        ax.set_xlabel("Attack Condition", fontsize=9)
        ax.yaxis.grid(True, linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)
        ax.legend(fontsize=9, framealpha=0.85)

    fig.suptitle("2×2 Factorial: Tool Defense × Attack",
                 fontsize=13, fontweight="bold", y=1.02)
    _savefig(fig, save_path)


# ---------------------------------------------------------------------------
# 10 — Swarm co-activation network
# ---------------------------------------------------------------------------

def plot_swarm_network(df_network: pd.DataFrame, save_path: str) -> None:
    """Agent co-activation network per condition.

    Nodes = agents, edge thickness = fraction of trials where both were active.
    Node size = participation rate. Reveals whether attack fragments the swarm.
    """
    if df_network.empty:
        return
    plt.style.use(_STYLE)

    conditions = sorted(df_network["condition"].unique())
    ncols = min(len(conditions), 4)
    nrows = (len(conditions) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.5 * ncols, 4 * nrows))
    axes_flat = np.array(axes).flatten() if len(conditions) > 1 else [axes]

    for ax, cond in zip(axes_flat, conditions):
        sub = df_network[df_network["condition"] == cond]
        if sub.empty:
            ax.axis("off")
            continue

        # Collect all agents
        agents = sorted(set(sub["agent_i"]) | set(sub["agent_j"]))
        n      = len(agents)
        if n == 0:
            ax.axis("off")
            continue

        # Circular layout
        angles = {a: 2 * np.pi * i / n for i, a in enumerate(agents)}
        pos    = {a: (np.cos(angles[a]), np.sin(angles[a])) for a in agents}

        # Node sizes proportional to sum of co-activation weights
        node_strength = {a: 0.0 for a in agents}
        for _, row in sub.iterrows():
            node_strength[row["agent_i"]] += row["co_activation_rate"]
            node_strength[row["agent_j"]] += row["co_activation_rate"]

        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.axis("off")

        # Draw edges
        max_w = sub["co_activation_rate"].max() or 1
        for _, row in sub.iterrows():
            xi, yi = pos[row["agent_i"]]
            xj, yj = pos[row["agent_j"]]
            lw = max(0.4, 4 * row["co_activation_rate"] / max_w)
            alpha = 0.3 + 0.6 * row["co_activation_rate"] / max_w
            ax.plot([xi, xj], [yi, yj], color="#95a5a6", lw=lw, alpha=alpha, zorder=1)

        # Draw nodes
        max_s = max(node_strength.values()) or 1
        for a in agents:
            x_, y_  = pos[a]
            size    = 200 + 600 * node_strength[a] / max_s
            ax.scatter(x_, y_, s=size, color=_PALETTE[0], zorder=3,
                       edgecolors="white", linewidths=1.5, alpha=0.85)
            # Short label (last part of agent id)
            label = a.split("_")[-1] if "_" in a else a[:6]
            ax.text(x_ * 1.25, y_ * 1.25, label,
                    ha="center", va="center", fontsize=7, color="#2c3e50")

        ax.set_title(f"{cond}", fontsize=10, fontweight="bold")

    # Hide unused axes
    for ax in axes_flat[len(conditions):]:
        ax.axis("off")

    fig.suptitle("Swarm Co-Activation Network\n(node size = centrality, edge = joint activity rate)",
                 fontsize=12, fontweight="bold")
    _savefig(fig, save_path)


# ---------------------------------------------------------------------------
# 11 — Behavioral fingerprint (radar)
# ---------------------------------------------------------------------------

def plot_behavioral_fingerprint(df_trials: pd.DataFrame, _df_agents: pd.DataFrame,
                                 save_path: str) -> None:
    """Radar chart: 6-axis behavioral profile per condition.

    Axes: Utility · Task Completion · Participation · Safety (1-ASR) ·
          Conversation Depth (norm) · Tool Reliability (1-fail_rate)

    Tells the full story at a glance.
    """
    plt.style.use(_STYLE)
    conditions = sorted(df_trials["condition"].unique())

    AXES = [
        ("Utility",            "utility_score",            df_trials, True),
        ("Task Completion",    "task_completed",            df_trials, True),
        ("Participation",      "participation_rate",        df_trials, True),
        ("Safety\n(1−ASR)",    "target_execution_asr",      df_trials, False),   # inverted
        ("Conversation\nDepth","mean_tool_calls_per_agent", df_trials, True),
        ("Tool\nReliability",  "tool_failure_rate_mean",    df_trials, False),   # inverted
    ]

    # Compute per-condition means
    cond_profiles: dict[str, list[float]] = {}
    raw_vals: dict[str, list[float]] = {}
    for cond in conditions:
        vals = []
        for _, col, src, positive in AXES:
            if col not in src.columns:
                vals.append(0.5)
                continue
            v = src[src["condition"] == cond][col]
            v = v[v >= 0] if col in ("utility_score", "target_execution_asr") else v
            m = v.mean() if len(v) else 0.0
            vals.append(m)
        raw_vals[cond] = vals

    # Normalize each axis 0-1 across conditions
    raw_arr = np.array([raw_vals[c] for c in conditions])   # shape (n_cond, n_axes)
    for j, (_, _, _, positive) in enumerate(AXES):
        col_min, col_max = raw_arr[:, j].min(), raw_arr[:, j].max()
        span = col_max - col_min or 1.0
        for i in range(len(conditions)):
            v = (raw_arr[i, j] - col_min) / span
            cond_profiles[conditions[i]] = cond_profiles.get(conditions[i], [])
            if not positive:
                v = 1.0 - v   # invert: higher = better
            cond_profiles[conditions[i]].append(v)

    labels   = [a[0] for a in AXES]
    n_axes   = len(labels)
    angles   = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles  += angles[:1]   # close the polygon
    colors   = _cond_colors(conditions)

    fig, ax = plt.subplots(figsize=(6, 6),
                            subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=7, color="grey")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)

    for cond, color in zip(conditions, colors):
        vals = cond_profiles[cond] + cond_profiles[cond][:1]
        ax.plot(angles, vals, color=color, lw=2, label=cond)
        ax.fill(angles, vals, color=color, alpha=0.12)

    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9, framealpha=0.85)
    ax.set_title("Behavioral Fingerprint by Condition\n(all axes: higher = better)",
                 fontsize=11, fontweight="bold", pad=20)
    _savefig(fig, save_path)


# ---------------------------------------------------------------------------
# 12 — Swarm interaction flow (single trial)
# ---------------------------------------------------------------------------

def plot_swarm_interaction(trial: dict, save_path: str) -> None:
    """Directed flow diagram for one trial.

    Feed → agents (dashed, read), Agent → Feed (post), Agent → external tools.
    Node colour: green=posted, red=dangerous action, amber=email, grey=no action.
    """
    plt.style.use(_STYLE)
    results = [a for a in trial.get("agent_results", []) if "error" not in a]
    n = len(results)
    if n == 0:
        return

    fig, ax = plt.subplots(figsize=(10, max(5, n * 0.9)))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, max(6, n + 1))
    ax.axis("off")

    feed_pos  = (1.2, (n + 1) / 2)
    ext_pos   = (8.8, (n + 1) / 2)
    agent_ys  = [(n + 1) / 2 + (i - (n - 1) / 2) * 1.1 for i in range(n)]

    def node(pos, label, color, radius=0.45):
        ax.add_patch(plt.Circle(pos, radius, facecolor=color, edgecolor="white",
                                linewidth=1.5, zorder=3))
        ax.text(pos[0], pos[1], label, ha="center", va="center",
                fontsize=7, fontweight="bold", color="white", zorder=4)

    def arrow(x0, y0, x1, y1, color, lw=1.5, ls="-"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color=color,
                                   lw=lw, linestyle=ls), zorder=2)

    node(feed_pos, "Feed", "#2c3e50", radius=0.55)
    node(ext_pos,  "External\nTools", _PALETTE[1], radius=0.55)

    for i, (agent, ay) in enumerate(zip(results, agent_ys)):
        aid       = agent.get("agent_id", f"agent_{i}")
        posted    = bool(agent.get("posted_content"))
        emailed   = bool(agent.get("sent_email"))
        dangerous = bool(agent.get("dangerous_tool_called"))
        blocked   = bool(agent.get("write_blocked") or agent.get("tool_call_blocked"))
        n_calls   = agent.get("n_tool_calls", 0)

        if blocked:
            color = "#e67e22"   # orange = blocked
        elif dangerous:
            color = _PALETTE[1]  # red = dangerous
        elif posted:
            color = _PALETTE[3]  # green = posted
        elif emailed:
            color = _PALETTE[2]  # amber = email only
        else:
            color = "#7f8c8d"    # grey = no action

        ap = (4.5, ay)
        node(ap, aid[:8], color, radius=0.42)
        ax.text(4.5, ay - 0.6, f"calls={n_calls}", ha="center", fontsize=6, color="#555")

        # Feed → agent (read)
        arrow(feed_pos[0] + 0.55, feed_pos[1], ap[0] - 0.42, ay,
              color="#95a5a6", lw=1.2, ls="dashed")
        # Agent → Feed
        if posted:
            arrow(ap[0] + 0.42, ay + 0.1, feed_pos[0] + 0.55, feed_pos[1] + 0.1,
                  color=_PALETTE[3], lw=2.0)
        # Agent → External
        if emailed or dangerous:
            arrow(ap[0] + 0.42, ay, ext_pos[0] - 0.55, ext_pos[1],
                  color=_PALETTE[1] if dangerous else _PALETTE[2], lw=2.0)
        if blocked:
            ax.text(ap[0], ay - 0.75, "⛔ blocked", ha="center",
                    fontsize=7, color=_PALETTE[1])

    legend_elements = [
        mpatches.Patch(color=_PALETTE[3], label="Posted to feed"),
        mpatches.Patch(color=_PALETTE[1], label="Dangerous tool call"),
        mpatches.Patch(color=_PALETTE[2], label="Email only"),
        mpatches.Patch(color="#e67e22",   label="Blocked by defense"),
        mpatches.Patch(color="#7f8c8d",   label="No action"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8, framealpha=0.9)

    inject = trial.get("inject_payload", False)
    asr    = trial.get("target_execution_asr", "?")
    util   = trial.get("utility_score",        "?")
    cond   = trial.get("condition",            "?")
    td     = trial.get("tool_defense",         False)
    title  = (
        f"Trial {trial.get('trial_id')} — {cond} | "
        f"attack={inject} | tool_defense={td} | "
        f"ASR={asr} | utility={util:.2f}"
        if isinstance(util, float) else
        f"Trial {trial.get('trial_id')} — {cond}"
    )
    ax.set_title(title, fontsize=10, fontweight="bold")
    _savefig(fig, save_path, tight=False)


# ---------------------------------------------------------------------------
# 13 — ASR by attack type
# ---------------------------------------------------------------------------

def plot_asr_by_attack_type(df_trials: pd.DataFrame, save_path: str) -> None:
    """Bar: mean ASR by InjecAgent attack type, split by condition."""
    plt.style.use(_STYLE)
    df = df_trials[
        (df_trials["inject_payload"].astype(int) == 1) &
        (df_trials["target_execution_asr"].fillna(-1) >= 0)
    ].copy()
    if df.empty or df["attack_type"].isna().all():
        return

    agg = (df.groupby(["attack_type", "condition"])["target_execution_asr"]
           .agg(["mean", "sem", "count"]).reset_index()
           .sort_values("mean", ascending=False))

    conditions = sorted(agg["condition"].unique())
    attack_types = agg["attack_type"].unique()
    x     = np.arange(len(attack_types))
    width = 0.8 / max(len(conditions), 1)
    colors = _cond_colors(conditions)

    fig, ax = plt.subplots(figsize=(max(8, len(attack_types) * 1.4), 4))
    for di, (cond, color) in enumerate(zip(conditions, colors)):
        sub  = agg[agg["condition"] == cond].set_index("attack_type")
        means = [sub.loc[at, "mean"] if at in sub.index else 0.0 for at in attack_types]
        sems  = [sub.loc[at, "sem"]  if at in sub.index else 0.0 for at in attack_types]
        ax.bar(x + (di - len(conditions) / 2 + 0.5) * width, means,
               width * 0.9, yerr=sems, label=cond, color=color, alpha=0.82, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(attack_types, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Attack Success Rate", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, title="Condition", framealpha=0.85)
    ax.set_title("ASR by InjecAgent Attack Type × Condition", fontsize=12, fontweight="bold")
    _savefig(fig, save_path)


# ---------------------------------------------------------------------------
# 14 — Payload position vs ASR
# ---------------------------------------------------------------------------

def plot_payload_position(df_trials: pd.DataFrame, save_path: str) -> None:
    """Bar: ASR bucketed by injection position in feed."""
    plt.style.use(_STYLE)
    df = df_trials[
        (df_trials["inject_payload"].astype(int) == 1) &
        df_trials["payload_position"].notna() &
        (df_trials["target_execution_asr"].fillna(-1) >= 0)
    ].copy()
    if df.empty:
        return

    max_pos = int(df["payload_position"].max())
    n_bins  = min(5, max_pos + 1)
    df["pos_bucket"] = pd.cut(df["payload_position"], bins=n_bins,
                               include_lowest=True).astype(str)
    agg = df.groupby("pos_bucket")["target_execution_asr"].agg(["mean","sem"]).reset_index()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(agg)), agg["mean"], yerr=agg["sem"],
           color=_PALETTE[2], alpha=0.82, capsize=4)
    ax.set_xticks(range(len(agg)))
    ax.set_xticklabels(agg["pos_bucket"], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Attack Success Rate", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_title("ASR by Payload Position in Feed", fontsize=12, fontweight="bold")
    ax.set_xlabel("Injection Position (bucketed)", fontsize=10)
    _savefig(fig, save_path)


# ---------------------------------------------------------------------------
# 15 — Tool defense block rate by agent
# ---------------------------------------------------------------------------

def plot_tool_defense_blocks(df_agents: pd.DataFrame, save_path: str) -> None:
    """Bar: mean tool_defense blocks per agent-trial, by condition × agent."""
    if "tool_defense_blocks" not in df_agents.columns:
        return
    plt.style.use(_STYLE)

    # Only conditions where tool_defense was actually active
    if "tool_defense" in df_agents.columns:
        df = df_agents[df_agents["tool_defense"].astype(int) == 1].copy()
    else:
        df = df_agents.copy()

    if df.empty or df["tool_defense_blocks"].sum() == 0:
        return

    agg = (
        df.groupby(["condition", "agent_id"])["tool_defense_blocks"]
        .mean().reset_index()
        .rename(columns={"tool_defense_blocks": "mean_blocks"})
    )
    conditions = sorted(agg["condition"].unique())
    colors     = dict(zip(conditions, _cond_colors(conditions)))

    fig, ax = plt.subplots(figsize=(max(8, agg["agent_id"].nunique() * 0.9), 4))
    for cond in conditions:
        sub = agg[agg["condition"] == cond]
        ax.scatter(sub["agent_id"], sub["mean_blocks"],
                   label=cond, color=colors[cond], s=60, alpha=0.85, zorder=3)

    ax.set_ylabel("Mean tool_defense blocks / trial", fontsize=10)
    ax.set_xlabel("Agent", fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, title="Condition", framealpha=0.85)
    ax.set_title("Tool Defense Block Rate by Agent", fontsize=12, fontweight="bold")
    plt.xticks(rotation=30, ha="right", fontsize=8)
    _savefig(fig, save_path)
