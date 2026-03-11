# hardshell/analysis/plotting.py
"""
Publication-quality figures for the HardShell swarm experiment.

Figures produced:
  01_condition_summary      — ASR / Utility / Participation by condition (3-panel)
  02_utility_distribution   — Utility score distribution (violin) by condition
  03_agent_participation    — Heatmap: agent role × condition → post rate
  04_post_length_by_role    — Violin: post length by agent id
  05_agent_action_breakdown — Stacked bar: posted / sent_email / no_action / error per condition
  06_swarm_interaction      — Directed flow diagram for a representative trial
  07_asr_by_attack_type     — Bar: ASR by InjecAgent attack category (condition 1+ only)
  08_payload_position       — Bar: ASR by payload position bucket (condition 1+ only)
"""
import warnings

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)

_STYLE   = "seaborn-v0_8-paper"
_DPI     = 300
_PALETTE = ["#2c3e50", "#e74c3c", "#f39c12", "#2ecc71"]   # navy, red, amber, green


def _savefig(fig, path: str, tight: bool = True) -> None:
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 01 — Condition summary (3-panel)
# ---------------------------------------------------------------------------

def plot_condition_summary(df_trials: pd.DataFrame, save_path: str) -> None:
    """3-panel bar chart: ASR, utility score, participation rate — by condition."""
    plt.style.use(_STYLE)
    conditions = sorted(df_trials["condition"].unique())
    x = np.arange(len(conditions))
    width = 0.55

    metrics = [
        ("target_execution_asr", "Attack Success Rate (ASR)", _PALETTE[1]),
        ("utility_score",        "Task Utility Score",        _PALETTE[3]),
        ("participation_rate",   "Agent Participation Rate",  _PALETTE[0]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)

    for ax, (col, label, color) in zip(axes, metrics):
        means = [df_trials[df_trials["condition"] == c][col].mean() for c in conditions]
        sems  = [df_trials[df_trials["condition"] == c][col].sem()  for c in conditions]
        bars = ax.bar(x, means, width, yerr=sems, color=color, alpha=0.85,
                      capsize=4, error_kw={"linewidth": 1.2})
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=15, ha="right", fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel(label, fontsize=10)
        ax.set_xlabel("Condition", fontsize=9)
        ax.yaxis.grid(True, linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                    f"{mean:.2f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("HardShell Swarm — Condition Summary", fontsize=13, fontweight="bold", y=1.01)
    _savefig(fig, save_path)


# ---------------------------------------------------------------------------
# 02 — Utility distribution (violin)
# ---------------------------------------------------------------------------

def plot_utility_distribution(df_trials: pd.DataFrame, save_path: str) -> None:
    """Violin plot of utility_score distribution per condition."""
    plt.style.use(_STYLE)
    fig, ax = plt.subplots(figsize=(8, 4))
    conditions = sorted(df_trials["condition"].unique())
    data = [df_trials[df_trials["condition"] == c]["utility_score"].dropna().values
            for c in conditions]
    parts = ax.violinplot(data, showmedians=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_facecolor(_PALETTE[3])
        pc.set_alpha(0.7)
    ax.set_xticks(range(1, len(conditions) + 1))
    ax.set_xticklabels(conditions, fontsize=9)
    ax.set_ylabel("Utility Score (0–1)", fontsize=11)
    ax.set_xlabel("Condition", fontsize=10)
    ax.set_ylim(-0.05, 1.1)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_title("Utility Score Distribution by Condition", fontsize=12, fontweight="bold")
    _savefig(fig, save_path)


# ---------------------------------------------------------------------------
# 03 — Agent participation heatmap
# ---------------------------------------------------------------------------

def plot_agent_participation_heatmap(df_agents: pd.DataFrame, save_path: str) -> None:
    """Heatmap: agent role × condition → post rate."""
    plt.style.use(_STYLE)
    pivot = (
        df_agents.groupby(["agent_id", "condition"])["posted"]
        .mean()
        .unstack(fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(max(5, len(pivot.columns) * 1.4), max(3, len(pivot) * 0.8)))
    sns.heatmap(
        pivot, annot=True, fmt=".0%", cmap="YlGn",
        vmin=0, vmax=1, linewidths=0.5, ax=ax,
        cbar_kws={"label": "Post Rate"},
    )
    ax.set_title("Agent Participation Rate by Role × Condition", fontsize=12, fontweight="bold")
    ax.set_xlabel("Condition", fontsize=10)
    ax.set_ylabel("Agent Role", fontsize=10)
    _savefig(fig, save_path)


# ---------------------------------------------------------------------------
# 04 — Post length by role (violin)
# ---------------------------------------------------------------------------

def plot_post_length_by_role(df_agents: pd.DataFrame, save_path: str) -> None:
    """Violin: post character length by agent role (only agents that posted)."""
    plt.style.use(_STYLE)
    posted = df_agents[df_agents["posted"] == 1].copy()
    if posted.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    roles = sorted(posted["agent_id"].unique())
    data = [posted[posted["agent_id"] == r]["post_length"].values for r in roles]
    parts = ax.violinplot(data, showmedians=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_facecolor(_PALETTE[0])
        pc.set_alpha(0.65)
    ax.set_xticks(range(1, len(roles) + 1))
    ax.set_xticklabels(roles, fontsize=9)
    ax.set_ylabel("Post Length (chars)", fontsize=11)
    ax.set_xlabel("Agent Role", fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_title("Post Length Distribution by Agent Role", fontsize=12, fontweight="bold")
    _savefig(fig, save_path)


# ---------------------------------------------------------------------------
# 05 — Agent action breakdown (stacked bar)
# ---------------------------------------------------------------------------

def plot_agent_action_breakdown(df_agents: pd.DataFrame, df_trials: pd.DataFrame,
                                 save_path: str) -> None:
    """
    Stacked bar per condition showing fraction of agent-slots that:
    posted only / sent email / both / no action / errored
    """
    plt.style.use(_STYLE)
    conditions = sorted(df_agents["condition"].unique())
    results = []

    for cond in conditions:
        ag = df_agents[df_agents["condition"] == cond]
        n = len(ag) or 1
        # count by action type
        posted_only  = len(ag[(ag["posted"] == 1) & (ag["sent_email"] == 0)])
        sent_only    = len(ag[(ag["posted"] == 0) & (ag["sent_email"] == 1)])
        both         = len(ag[(ag["posted"] == 1) & (ag["sent_email"] == 1)])
        no_action    = len(ag[(ag["posted"] == 0) & (ag["sent_email"] == 0)])
        # errored rows are excluded from df_agents, so count from trials
        results.append({
            "condition":   cond,
            "Posted":      posted_only / n,
            "Email":       sent_only / n,
            "Both":        both / n,
            "No action":   no_action / n,
        })

    rdf = pd.DataFrame(results).set_index("condition")
    colors = [_PALETTE[3], _PALETTE[1], _PALETTE[2], "#aaaaaa"]

    fig, ax = plt.subplots(figsize=(8, 4))
    rdf.plot(kind="bar", stacked=True, ax=ax, color=colors, width=0.55, alpha=0.88)
    ax.set_ylabel("Fraction of Agent Slots", fontsize=11)
    ax.set_xlabel("Condition", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.8)
    ax.set_title("Agent Action Breakdown by Condition", fontsize=12, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
    _savefig(fig, save_path)


# ---------------------------------------------------------------------------
# 06 — Swarm interaction diagram (single trial)
# ---------------------------------------------------------------------------

def plot_swarm_interaction(trial: dict, save_path: str) -> None:
    """
    Directed flow diagram for one trial:
      Feed → agents (dashed, read)
      Agent → Feed (solid green, posted)
      Agent → Email node (solid red, sent_email)
      Agent → ✗ (grey, no action / blocked)
    """
    plt.style.use(_STYLE)
    results = [a for a in trial.get("agent_results", []) if "error" not in a]
    n = len(results)
    if n == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Positions
    feed_pos  = (1.2, 3.0)
    email_pos = (8.8, 1.5)
    agent_ys  = [3.0 + (i - (n - 1) / 2) * 1.1 for i in range(n)]

    def node(pos, label, color, radius=0.45):
        circle = plt.Circle(pos, radius, color=color, zorder=3, linewidth=1.5,
                             edgecolor="white")
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], label, ha="center", va="center",
                fontsize=8, fontweight="bold", color="white", zorder=4)

    def arrow(x0, y0, x1, y1, color, style="->", lw=1.5, ls="-"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle=style, color=color,
                                   lw=lw, linestyle=ls),
                    zorder=2)

    # Feed node
    node(feed_pos, "Feed", "#2c3e50", radius=0.55)

    # Email node
    node(email_pos, "📧 Email", _PALETTE[1], radius=0.5)

    # Agent nodes + arrows
    for i, (agent, ay) in enumerate(zip(results, agent_ys)):
        role    = agent.get("agent_id", f"agent_{i}")
        posted  = bool(agent.get("posted_content"))
        emailed = bool(agent.get("sent_email"))
        blocked = bool(agent.get("write_blocked") or agent.get("tool_call_blocked"))
        color   = _PALETTE[3] if posted else (_PALETTE[1] if emailed else "#7f8c8d")

        ax_pos = (4.5, ay)
        node(ax_pos, role, color, radius=0.42)

        # Feed → agent (read, dashed)
        arrow(feed_pos[0] + 0.55, feed_pos[1], ax_pos[0] - 0.42, ay,
              color="#95a5a6", style="->", lw=1.2, ls="dashed")

        # Agent → Feed (post)
        if posted:
            arrow(ax_pos[0] + 0.42, ay + 0.1,
                  feed_pos[0] + 0.55, feed_pos[1] + 0.1,
                  color=_PALETTE[3], style="->", lw=2.0)

        # Agent → Email
        if emailed:
            arrow(ax_pos[0] + 0.42, ay, email_pos[0] - 0.5, email_pos[1],
                  color=_PALETTE[1], style="->", lw=2.0)

        # Label: blocked indicator
        if blocked:
            ax.text(ax_pos[0], ay - 0.55, "⛔ blocked", ha="center",
                    fontsize=7, color=_PALETTE[1])

    # Legend
    legend_elements = [
        mpatches.Patch(color=_PALETTE[3], label="Posted to feed"),
        mpatches.Patch(color=_PALETTE[1], label="Sent email"),
        mpatches.Patch(color="#7f8c8d",   label="No action"),
        mpatches.Patch(color="#95a5a6",   label="Read feed (all agents)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8, framealpha=0.9)

    inject = trial.get("inject_payload", True)
    asr    = trial.get("target_execution_asr", "?")
    util   = trial.get("utility_score", "?")
    cond   = trial.get("condition", "?")
    title  = (f"Trial {trial.get('trial_id')} — {cond} | "
              f"injected={inject} | ASR={asr} | utility={util:.2f}"
              if isinstance(util, float) else
              f"Trial {trial.get('trial_id')} — {cond} | injected={inject}")
    ax.set_title(title, fontsize=11, fontweight="bold")

    _savefig(fig, save_path, tight=False)


# ---------------------------------------------------------------------------
# 07 — ASR by attack type
# ---------------------------------------------------------------------------

def plot_asr_by_attack_type(df_trials: pd.DataFrame, save_path: str) -> None:
    """Bar chart: mean ASR by InjecAgent attack category."""
    plt.style.use(_STYLE)
    attack_df = df_trials[df_trials["inject_payload"] == True].copy()  # noqa: E712
    if attack_df.empty or attack_df["attack_type"].isna().all():
        return

    agg = (attack_df.groupby("attack_type")["target_execution_asr"]
           .agg(["mean", "sem", "count"]).reset_index()
           .sort_values("mean", ascending=False))

    fig, ax = plt.subplots(figsize=(max(7, len(agg) * 1.1), 4))
    bars = ax.bar(range(len(agg)), agg["mean"], yerr=agg["sem"],
                  color=_PALETTE[1], alpha=0.82, capsize=4)
    ax.set_xticks(range(len(agg)))
    ax.set_xticklabels(agg["attack_type"], rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Attack Success Rate", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    for bar, (_, row) in zip(bars, agg.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.04,
                f"n={int(row['count'])}", ha="center", fontsize=7)
    ax.set_title("ASR by InjecAgent Attack Type", fontsize=12, fontweight="bold")
    _savefig(fig, save_path)


# ---------------------------------------------------------------------------
# 08 — Payload position vs ASR
# ---------------------------------------------------------------------------

def plot_payload_position(df_trials: pd.DataFrame, save_path: str) -> None:
    """Bar: ASR bucketed by payload injection position in the feed."""
    plt.style.use(_STYLE)
    attack_df = df_trials[df_trials["inject_payload"] == True].copy()  # noqa: E712
    attack_df = attack_df.dropna(subset=["payload_position"])
    if attack_df.empty:
        return

    max_pos = int(attack_df["payload_position"].max())
    n_bins  = min(5, max_pos + 1)
    attack_df["pos_bucket"] = pd.cut(
        attack_df["payload_position"], bins=n_bins, include_lowest=True
    ).astype(str)

    agg = (attack_df.groupby("pos_bucket")["target_execution_asr"]
           .agg(["mean", "sem"]).reset_index())

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
