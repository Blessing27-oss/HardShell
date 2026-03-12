"""
HardShell analysis runner.

Discovers all dialogue_log.jsonl files under runs/, loads them into
trial-level and agent-level DataFrames, then produces:

  tables/
    summary.csv            — per-condition means (ASR, utility, tools, …)
    agents.csv             — per-agent-per-trial metrics
    role_breakdown.csv     — participation / email / dangerous-tool rate by agent × condition
    network_edges.csv      — agent co-activation edge list per condition
    tool_usage.csv         — per-condition per-tool usage rates
    regression.txt         — OLS regressions (2×2 factorial + utility-tax)

  plots/
    01_condition_summary.pdf
    02_utility_distribution.pdf
    03_agent_participation.pdf
    04_conversation_depth.pdf
    05_tool_usage_heatmap.pdf
    06_dangerous_tool_rate.pdf
    07_action_breakdown.pdf
    08_asr_utility_scatter.pdf
    09_2x2_factorial.pdf          ← primary result (when 2×2 data present)
    10_swarm_network.pdf
    11_behavioral_fingerprint.pdf
    12_swarm_interaction_<cond>_t<N>.pdf   ← one per condition
    13_asr_by_attack_type.pdf
    14_payload_position.pdf
    15_tool_defense_blocks.pdf
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
import pandas as pd
import statsmodels.formula.api as smf
from omegaconf import DictConfig
from tqdm import tqdm

from hardshell.analysis.extract_metrics import (
    load_transcripts,
    compute_summary_stats,
    build_network_edges,
)
from hardshell.analysis.plotting import (
    plot_condition_summary,
    plot_utility_distribution,
    plot_agent_participation_heatmap,
    plot_conversation_depth,
    plot_tool_usage_heatmap,
    plot_dangerous_tool_rate,
    plot_agent_action_breakdown,
    plot_asr_utility_scatter,
    plot_2x2_factorial,
    plot_swarm_network,
    plot_behavioral_fingerprint,
    plot_swarm_interaction,
    plot_asr_by_attack_type,
    plot_payload_position,
    plot_tool_defense_blocks,
)
from hardshell.analysis.regressions import (
    factorial_regression,
    format_factorial_results,
)

log = logging.getLogger(__name__)


def _has(df: pd.DataFrame, col: str) -> bool:
    """True when column exists and has at least one non-null, non-negative value."""
    return col in df.columns and df[col].dropna().gt(-1).any()


def _fmt(p: Path) -> str:
    return str(p)


def _resolve_run_dir(cfg: DictConfig) -> Path:
    """Return the run folder to analyse.

    Priority:
      1. cfg.run_dir if explicitly set (CLI override or CLAUDE.md default)
      2. Latest timestamped subfolder of cfg.directories.runs that contains
         at least one dialogue_log.jsonl
    """
    if cfg.get("run_dir", ""):
        p = Path(cfg.run_dir)
        if not p.exists():
            raise FileNotFoundError(f"run_dir '{p}' does not exist.")
        return p

    runs_root = Path(cfg.directories.runs)
    candidates = sorted(
        (d for d in runs_root.iterdir() if d.is_dir()),
        key=lambda d: d.name,
        reverse=True,
    )
    for candidate in candidates:
        if list(candidate.rglob("dialogue_log.jsonl")):
            return candidate

    raise FileNotFoundError(
        f"No run folders with dialogue_log.jsonl found under '{runs_root}'. "
        "Run run_experiment.py first, or pass run_dir=<path> on the CLI."
    )


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run_dir    = _resolve_run_dir(cfg)
    tables_dir = run_dir / "analysis" / "tables"
    plots_dir  = run_dir / "analysis" / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    fmt = cfg.analysis.plotting.format   # "pdf" or "png"

    # ------------------------------------------------------------------ #
    # 1. Discover and load all transcripts                                 #
    # ------------------------------------------------------------------ #
    jsonl_files = sorted(run_dir.glob("**/dialogue_log.jsonl"))
    if not jsonl_files:
        log.error(
            f"No dialogue_log.jsonl files found under {run_dir}. "
            "Run run_experiment.py first."
        )
        return
    log.info(f"Found {len(jsonl_files)} transcript file(s).")

    trial_frames: list[pd.DataFrame] = []
    agent_frames: list[pd.DataFrame] = []
    raw_trials:   list[dict]         = []

    for jf in tqdm(jsonl_files, desc="Loading transcripts", unit="file"):
        df_t, df_a = load_transcripts(str(jf))
        trial_frames.append(df_t)
        agent_frames.append(df_a)
        with open(jf) as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_trials.append(json.loads(line))

    df_trials = pd.concat(trial_frames, ignore_index=True)
    df_agents = pd.concat(agent_frames, ignore_index=True)
    log.info(
        f"Loaded {len(df_trials)} trials × {len(df_agents)} agent-rows "
        f"across {df_trials['condition'].nunique()} condition(s)."
    )

    # ------------------------------------------------------------------ #
    # 2. Summary tables                                                    #
    # ------------------------------------------------------------------ #

    # Trial-level summary per condition
    base_agg: dict[str, tuple] = {
        "n_trials":                ("trial_id",               "count"),
        "asr_mean":                ("target_execution_asr",   "mean"),
        "asr_se":                  ("target_execution_asr",   "sem"),
        "utility_mean":            ("utility_score",          "mean"),
        "utility_se":              ("utility_score",          "sem"),
        "task_completed_rate":     ("task_completed",         "mean"),
        "participation_mean":      ("participation_rate",     "mean"),
        "posts_per_trial":         ("n_posted",               "mean"),
        "emails_per_trial":        ("n_sent_email",           "mean"),
        "errors_per_trial":        ("n_errored",              "mean"),
        "mean_post_length":        ("mean_post_length",       "mean"),
        "mean_tool_calls_agent":   ("mean_tool_calls_per_agent","mean"),
        "dangerous_calls_trial":   ("n_dangerous_calls",      "mean"),
        "tool_failure_rate":       ("tool_failure_rate_mean", "mean"),
        "distinct_tools_mean":     ("n_distinct_tools",       "mean"),
    }
    optional = {
        "tool_def_blocks_trial": "n_tool_defense_blocked",
    }
    for key, col in optional.items():
        if col in df_trials.columns:
            base_agg[key] = (col, "mean")

    summary = df_trials.groupby("condition").agg(**base_agg).reset_index()
    summary.to_csv(tables_dir / "summary.csv", index=False)
    log.info(f"Summary table:\n{summary.to_string(index=False)}")

    # Agent-level table
    df_agents.drop(columns=["posted_content"], errors="ignore").to_csv(
        tables_dir / "agents.csv", index=False
    )

    # Role breakdown
    agent_agg: dict[str, tuple] = {
        "n_trials":           ("trial_id",             "count"),
        "post_rate":          ("posted",               "mean"),
        "email_rate":         ("sent_email",           "mean"),
        "dangerous_rate":     ("dangerous_tool_called","mean"),
        "mean_tool_calls":    ("n_tool_calls",         "mean"),
        "mean_fail_rate":     ("tool_failure_rate",    "mean"),
        "write_block_rate":   ("write_blocked",        "mean"),
        "td_block_rate":      ("tool_defense_blocks",  "mean"),
    }
    role_breakdown = (
        df_agents.groupby(["condition", "agent_id"])
        .agg(**{k: v for k, v in agent_agg.items() if v[0] in df_agents.columns})
        .reset_index()
    )
    role_breakdown.to_csv(tables_dir / "role_breakdown.csv", index=False)

    # Network edges
    df_network = build_network_edges(df_agents)
    df_network.to_csv(tables_dir / "network_edges.csv", index=False)

    # Tool usage table
    if "tools_used" in df_agents.columns:
        tool_rows = []
        for _, row in df_agents.iterrows():
            for tool in str(row.get("tools_used", "")).split(","):
                tool = tool.strip()
                if tool:
                    tool_rows.append({"condition": row["condition"], "tool": tool})
        if tool_rows:
            tdf = pd.DataFrame(tool_rows)
            n_agent_trials = df_agents.groupby("condition").size().rename("n")
            tool_usage = (
                tdf.groupby(["condition", "tool"]).size()
                / n_agent_trials
            ).fillna(0).reset_index(name="usage_rate")
            tool_usage.to_csv(tables_dir / "tool_usage.csv", index=False)

    # Text summary
    text_summary = compute_summary_stats(df_trials, df_agents)
    (tables_dir / "summary.txt").write_text(text_summary)
    print(text_summary)

    # ------------------------------------------------------------------ #
    # 3. Regressions                                                       #
    # ------------------------------------------------------------------ #
    reg_texts: list[str] = []

    # 3a. 2×2 factorial (tool_defense × attack)
    has_factorial = (
        "tool_defense" in df_trials.columns
        and "inject_payload" in df_trials.columns
        and df_trials["tool_defense"].nunique() > 1
        and df_trials["inject_payload"].nunique() > 1
        and _has(df_trials, "utility_score")
    )
    if has_factorial:
        try:
            u_model, asr_model = factorial_regression(df_trials)
            reg_texts.append(format_factorial_results(u_model, asr_model))
            log.info("2×2 factorial regression complete.")
        except Exception as e:
            log.warning(f"Factorial regression failed: {e}")

    # 3b. Utility-tax regression across conditions
    if df_trials["condition"].nunique() > 1 and _has(df_trials, "utility_score"):
        try:
            df_reg = df_trials[df_trials["utility_score"].fillna(-1) >= 0].copy()
            baseline = sorted(df_reg["condition"].unique())[0]
            df_reg["condition"] = pd.Categorical(
                df_reg["condition"], categories=sorted(df_reg["condition"].unique())
            )
            model = smf.ols("utility_score ~ C(condition)", data=df_reg).fit()
            reg_texts.append(
                f"Utility Tax Regression (baseline = {baseline})\n"
                + "=" * 50 + "\n"
                + str(model.summary())
            )
        except Exception as e:
            log.warning(f"Condition regression failed: {e}")

    # 3c. ASR regression across conditions
    if df_trials["condition"].nunique() > 1 and _has(df_trials, "target_execution_asr"):
        try:
            df_asr = df_trials[df_trials["target_execution_asr"].fillna(-1) >= 0].copy()
            df_asr["condition"] = pd.Categorical(
                df_asr["condition"], categories=sorted(df_asr["condition"].unique())
            )
            asr_model = smf.ols("target_execution_asr ~ C(condition)", data=df_asr).fit()
            reg_texts.append(
                "ASR Regression across conditions\n"
                + "=" * 50 + "\n"
                + str(asr_model.summary())
            )
        except Exception as e:
            log.warning(f"ASR regression failed: {e}")

    if reg_texts:
        (tables_dir / "regression.txt").write_text("\n\n".join(reg_texts))
        log.info(f"Regression(s) saved → {tables_dir / 'regression.txt'}")
    else:
        log.warning("Skipping regressions — insufficient data.")

    # ------------------------------------------------------------------ #
    # 4. Plots                                                             #
    # ------------------------------------------------------------------ #

    def step(label: str, fn):
        return (label, fn)

    plot_steps = [
        step("01 condition summary",      lambda: plot_condition_summary(
            df_trials, _fmt(plots_dir / f"01_condition_summary.{fmt}"))),
        step("02 utility distribution",   lambda: plot_utility_distribution(
            df_trials, _fmt(plots_dir / f"02_utility_distribution.{fmt}"))),
    ]

    if not df_agents.empty:
        plot_steps += [
            step("03 agent participation",  lambda: plot_agent_participation_heatmap(
                df_agents, _fmt(plots_dir / f"03_agent_participation.{fmt}"))),
            step("04 conversation depth",   lambda: plot_conversation_depth(
                df_agents, _fmt(plots_dir / f"04_conversation_depth.{fmt}"))),
            step("05 tool usage heatmap",   lambda: plot_tool_usage_heatmap(
                df_agents, _fmt(plots_dir / f"05_tool_usage_heatmap.{fmt}"))),
        ]

    if _has(df_trials, "n_dangerous_calls"):
        plot_steps.append(
            step("06 dangerous tool rate",  lambda: plot_dangerous_tool_rate(
                df_trials, _fmt(plots_dir / f"06_dangerous_tool_rate.{fmt}"))))

    if not df_agents.empty:
        plot_steps.append(
            step("07 action breakdown",     lambda: plot_agent_action_breakdown(
                df_agents, df_trials,
                _fmt(plots_dir / f"07_action_breakdown.{fmt}"))))

    if _has(df_trials, "utility_score") and _has(df_trials, "target_execution_asr"):
        plot_steps.append(
            step("08 ASR-utility scatter",  lambda: plot_asr_utility_scatter(
                df_trials, _fmt(plots_dir / f"08_asr_utility_scatter.{fmt}"))))

    # 2×2 factorial (only when both axes vary)
    if has_factorial:
        plot_steps.append(
            step("09 2×2 factorial",        lambda: plot_2x2_factorial(
                df_trials, _fmt(plots_dir / f"09_2x2_factorial.{fmt}"))))

    if not df_network.empty:
        plot_steps.append(
            step("10 swarm network",        lambda: plot_swarm_network(
                df_network, _fmt(plots_dir / f"10_swarm_network.{fmt}"))))

    if not df_agents.empty and _has(df_trials, "utility_score"):
        plot_steps.append(
            step("11 behavioral fingerprint", lambda: plot_behavioral_fingerprint(
                df_trials, df_agents,
                _fmt(plots_dir / f"11_behavioral_fingerprint.{fmt}"))))

    # Swarm interaction — one representative trial per condition
    used_conditions: set = set()
    for trial in raw_trials:
        cond = trial.get("condition")
        if cond not in used_conditions and trial.get("agent_results"):
            tid  = trial.get("trial_id", 0)
            path = _fmt(plots_dir / f"12_swarm_interaction_{cond}_t{tid}.{fmt}")
            plot_steps.append(
                step(f"12 swarm interaction ({cond})",
                     (lambda t=trial, p=path: plot_swarm_interaction(t, p))))
            used_conditions.add(cond)

    # Attack-specific plots
    has_attack = (
        "inject_payload" in df_trials.columns
        and df_trials["inject_payload"].astype(int).any()
    )
    if has_attack and "attack_type" in df_trials.columns:
        plot_steps.append(
            step("13 ASR by attack type",   lambda: plot_asr_by_attack_type(
                df_trials, _fmt(plots_dir / f"13_asr_by_attack_type.{fmt}"))))

    if has_attack and "payload_position" in df_trials.columns \
            and df_trials["payload_position"].notna().any():
        plot_steps.append(
            step("14 payload position",     lambda: plot_payload_position(
                df_trials, _fmt(plots_dir / f"14_payload_position.{fmt}"))))

    if "tool_defense_blocks" in df_agents.columns \
            and df_agents["tool_defense_blocks"].any():
        plot_steps.append(
            step("15 tool defense blocks",  lambda: plot_tool_defense_blocks(
                df_agents, _fmt(plots_dir / f"15_tool_defense_blocks.{fmt}"))))

    # Run all plots
    for label, fn in tqdm(plot_steps, desc="Generating plots", unit="plot"):
        tqdm.write(f"  → {label}")
        try:
            fn()
        except Exception as e:
            log.warning(f"Plot '{label}' failed: {e}")

    print(f"\nAnalysis complete — run: {run_dir}")
    print(f"  Tables → {tables_dir}")
    print(f"  Plots  → {plots_dir}\n")


if __name__ == "__main__":
    main()
