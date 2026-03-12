"""
HardShell analysis runner.

Discovers all dialogue_log.jsonl files under runs/, loads them into
two DataFrames (trial-level and agent-level), then produces:

  tables/
    summary.csv           — per-condition means (ASR, utility, participation, …)
    agents.csv            — per-agent-per-trial metrics
    role_breakdown.csv    — post rate / email rate by agent role × condition
    regression.txt        — OLS utility-tax regression summary

  plots/
    01_condition_summary.pdf
    02_utility_distribution.pdf
    03_agent_participation.pdf
    04_post_length_by_role.pdf
    05_agent_action_breakdown.pdf
    06_swarm_interaction_trial<N>.pdf   ← one per condition (representative trial)
    07_asr_by_attack_type.pdf           ← if injection data exists
    08_payload_position.pdf             ← if injection data exists
"""
import json
import logging
from pathlib import Path

import hydra
import pandas as pd
import statsmodels.formula.api as smf
from omegaconf import DictConfig
from tqdm import tqdm

from hardshell.analysis.extract_metrics import load_transcripts, compute_summary_stats
from hardshell.analysis.plotting import (
    plot_condition_summary,
    plot_utility_distribution,
    plot_agent_participation_heatmap,
    plot_post_length_by_role,
    plot_agent_action_breakdown,
    plot_swarm_interaction,
    plot_asr_by_attack_type,
    plot_payload_position,
    plot_2x2_factorial,
    plot_tool_defense_block_rate,
)
from hardshell.analysis.regressions import (
    calculate_utility_tax,
    factorial_regression,
    format_factorial_results,
)

log = logging.getLogger(__name__)


def _fmt(p: Path) -> str:
    return str(p)


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    runs_dir   = Path(cfg.directories.runs)
    tables_dir = runs_dir / "analysis" / "tables"
    plots_dir  = runs_dir / "analysis" / "plots"
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    fmt = cfg.analysis.plotting.format  # pdf or png

    # ------------------------------------------------------------------ #
    # 1. Discover and load all transcripts                                 #
    # ------------------------------------------------------------------ #
    jsonl_files = sorted(runs_dir.glob("**/dialogue_log.jsonl"))
    if not jsonl_files:
        log.error(f"No dialogue_log.jsonl files found under {runs_dir}. "
                  "Run run_experiment.py first.")
        return
    log.info(f"Found {len(jsonl_files)} transcript file(s).")

    trial_frames, agent_frames = [], []
    raw_trials: list[dict] = []   # for the interaction diagram

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
    log.info(f"Loaded {len(df_trials)} trials × {len(df_agents)} agent-rows "
             f"across {df_trials['condition'].nunique()} condition(s).")

    # ------------------------------------------------------------------ #
    # 2. Summary tables                                                    #
    # ------------------------------------------------------------------ #
    agg_cols = {
        "n_trials":           ("trial_id",            "count"),
        "asr_mean":           ("target_execution_asr","mean"),
        "asr_se":             ("target_execution_asr","sem"),
        "utility_mean":       ("utility_score",       "mean"),
        "utility_se":         ("utility_score",       "sem"),
        "task_completed":     ("task_completed",      "mean"),
        "participation_mean": ("participation_rate",  "mean"),
        "posts_per_trial":    ("n_posted",            "mean"),
        "emails_per_trial":   ("n_sent_email",        "mean"),
        "errors_per_trial":   ("n_errored",           "mean"),
        "mean_post_length":   ("mean_post_length",    "mean"),
    }
    if "n_tool_defense_blocked" in df_trials.columns:
        agg_cols["tool_def_blocks_per_trial"] = ("n_tool_defense_blocked", "mean")

    summary = df_trials.groupby("condition").agg(**agg_cols).reset_index()
    summary.to_csv(tables_dir / "summary.csv", index=False)
    log.info(f"Summary table:\n{summary.to_string(index=False)}")

    df_agents.to_csv(tables_dir / "agents.csv", index=False)

    role_breakdown = df_agents.groupby(["condition", "agent_id"]).agg(
        n_trials         = ("trial_id",    "count"),
        post_rate        = ("posted",      "mean"),
        mean_post_length = ("post_length", "mean"),
        email_rate       = ("sent_email",  "mean"),
        block_rate       = ("write_blocked","mean"),
    ).reset_index()
    role_breakdown.to_csv(tables_dir / "role_breakdown.csv", index=False)

    # Text summary
    text_summary = compute_summary_stats(df_trials, df_agents)
    (tables_dir / "summary.txt").write_text(text_summary)
    print(text_summary)

    # ------------------------------------------------------------------ #
    # 3. Regressions                                                        #
    # ------------------------------------------------------------------ #
    reg_texts = []

    # 3a. 2×2 factorial regression (tool_defense × attack)
    has_factorial = (
        "tool_defense" in df_trials.columns
        and "inject_payload" in df_trials.columns
        and df_trials["tool_defense"].nunique() > 1
        and df_trials["inject_payload"].nunique() > 1
        and df_trials["utility_score"].notna().any()
    )
    if has_factorial:
        try:
            u_model, asr_model = factorial_regression(df_trials)
            reg_texts.append(format_factorial_results(u_model, asr_model))
            log.info("2×2 factorial regression complete.")
        except Exception as e:
            log.warning(f"Factorial regression failed: {e}")

    # 3b. Legacy utility-tax regression across conditions
    if df_trials["condition"].nunique() > 1 and df_trials["utility_score"].notna().any():
        try:
            df_reg = df_trials.copy()
            baseline = df_reg["condition"].min()
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

    if reg_texts:
        (tables_dir / "regression.txt").write_text("\n\n".join(reg_texts))
        log.info(f"Regression(s) saved → {tables_dir / 'regression.txt'}")
    else:
        log.warning("Skipping regressions — insufficient data.")

    # ------------------------------------------------------------------ #
    # 4. Plots                                                             #
    # ------------------------------------------------------------------ #

    # Build the plot work list (label, callable)
    plot_steps = [
        ("01 condition summary",    lambda: plot_condition_summary(
            df_trials, _fmt(plots_dir / f"01_condition_summary.{fmt}"))),
        ("02 utility distribution", lambda: plot_utility_distribution(
            df_trials, _fmt(plots_dir / f"02_utility_distribution.{fmt}"))),
    ]
    if not df_agents.empty:
        plot_steps += [
            ("03 agent participation", lambda: plot_agent_participation_heatmap(
                df_agents, _fmt(plots_dir / f"03_agent_participation.{fmt}"))),
            ("04 post length by role", lambda: plot_post_length_by_role(
                df_agents, _fmt(plots_dir / f"04_post_length_by_role.{fmt}"))),
            ("05 action breakdown",    lambda: plot_agent_action_breakdown(
                df_agents, df_trials,
                _fmt(plots_dir / f"05_agent_action_breakdown.{fmt}"))),
        ]

    # Swarm interaction — one per condition
    used_conditions: set = set()
    for trial in raw_trials:
        cond = trial.get("condition")
        if cond not in used_conditions and trial.get("agent_results"):
            tid = trial.get("trial_id", 0)
            path = _fmt(plots_dir / f"06_swarm_interaction_{cond}_t{tid}.{fmt}")
            plot_steps.append(
                (f"06 swarm interaction ({cond})",
                 (lambda t=trial, p=path: plot_swarm_interaction(t, p)))
            )
            used_conditions.add(cond)

    # 2×2 factorial plot (only when both factors vary)
    if (
        "tool_defense" in df_trials.columns
        and df_trials["tool_defense"].nunique() > 1
        and "inject_payload" in df_trials.columns
        and df_trials["inject_payload"].nunique() > 1
    ):
        plot_steps.append(
            ("07 2×2 factorial",  lambda: plot_2x2_factorial(
                df_trials, _fmt(plots_dir / f"07_2x2_factorial.{fmt}")))
        )
    elif df_trials.get("inject_payload", pd.Series(False)).any():
        plot_steps.append(
            ("07 ASR by attack type", lambda: plot_asr_by_attack_type(
                df_trials, _fmt(plots_dir / f"07_asr_by_attack_type.{fmt}")))
        )

    if df_trials.get("payload_position", pd.Series(dtype=float)).notna().any():
        plot_steps.append(
            ("08 payload position", lambda: plot_payload_position(
                df_trials, _fmt(plots_dir / f"08_payload_position.{fmt}")))
        )

    if (
        "tool_defense" in df_trials.columns
        and df_trials["tool_defense"].any()
        and not df_agents.empty
    ):
        plot_steps.append(
            ("09 tool defense block rate", lambda: plot_tool_defense_block_rate(
                df_agents, _fmt(plots_dir / f"09_tool_defense_block_rate.{fmt}")))
        )

    for label, fn in tqdm(plot_steps, desc="Generating plots", unit="plot"):
        tqdm.write(f"  → {label}")
        fn()

    print(f"\nAnalysis complete.")
    print(f"  Tables → {tables_dir}")
    print(f"  Plots  → {plots_dir}\n")


if __name__ == "__main__":
    main()
