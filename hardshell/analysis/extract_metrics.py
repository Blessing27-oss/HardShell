# hardshell/analysis/extract_metrics.py
import json
import pandas as pd
from pathlib import Path


def load_transcripts(jsonl_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse a JSONL transcript into two DataFrames:

    df_trials  — one row per trial:
        trial_id, condition, inject_payload, num_agents,
        payload, attack_type, payload_position,
        target_execution_asr, task_completed, utility_score, judge_rationale,
        n_posted, n_sent_email, n_errored, participation_rate, mean_post_length

    df_agents  — one row per agent per trial:
        (all trial columns above, minus judge_rationale) +
        agent_id, tools_available, posted, posted_content, post_length,
        sent_email, had_email_tool, write_blocked, feed_blocks, tool_call_blocked
    """
    trial_rows = []
    agent_rows = []
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"Transcript not found: {jsonl_path}")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            trial = json.loads(line)

            results = trial.get("agent_results", [])
            successful = [a for a in results if "error" not in a]
            errored    = [a for a in results if "error" in a]

            n_posted     = sum(1 for a in successful if a.get("posted_content"))
            n_sent_email = sum(1 for a in successful if a.get("sent_email"))
            n_agents     = trial.get("num_agents", len(results)) or 1
            post_lengths = [len(a["posted_content"]) for a in successful if a.get("posted_content")]

            trial_row = {
                "trial_id":           trial.get("trial_id"),
                "condition":          trial.get("condition"),
                "inject_payload":     trial.get("inject_payload", True),
                "num_agents":         trial.get("num_agents"),
                "payload":            trial.get("payload"),
                "attack_type":        trial.get("attack_type"),
                "payload_position":   trial.get("payload_position"),
                "target_execution_asr": trial.get("target_execution_asr"),
                "task_completed":     trial.get("task_completed"),
                "utility_score":      trial.get("utility_score"),
                "judge_rationale":    trial.get("judge_rationale"),
                "n_posted":           n_posted,
                "n_sent_email":       n_sent_email,
                "n_errored":          len(errored),
                "participation_rate": n_posted / n_agents,
                "mean_post_length":   sum(post_lengths) / len(post_lengths) if post_lengths else 0.0,
            }
            trial_rows.append(trial_row)

            # Agent-level rows (drop judge_rationale — trial-level only)
            trial_meta = {k: v for k, v in trial_row.items() if k != "judge_rationale"}
            for agent in successful:
                had_email_tool = "send_email" in agent.get("tools_available", [])
                content = agent.get("posted_content")
                agent_rows.append({
                    **trial_meta,
                    "agent_id":           agent.get("agent_id"),
                    "tools_available":    json.dumps(agent.get("tools_available", [])),
                    "posted":             int(bool(content)),
                    "posted_content":     content,
                    "post_length":        len(content) if content else 0,
                    "sent_email":         int(bool(agent.get("sent_email"))),
                    "had_email_tool":     int(had_email_tool),
                    "write_blocked":      int(agent.get("write_blocked", False)),
                    "feed_blocks":        agent.get("feed_blocks", 0),
                    "tool_call_blocked":  int(agent.get("tool_call_blocked", False)),
                })

    return pd.DataFrame(trial_rows), pd.DataFrame(agent_rows)


def compute_summary_stats(df_trials: pd.DataFrame, df_agents: pd.DataFrame) -> str:
    """
    Returns a formatted text summary of key quantitative metrics per condition.
    """
    lines = ["=" * 60, "HardShell Swarm — Quantitative Summary", "=" * 60]

    for cond, grp in df_trials.groupby("condition"):
        n = len(grp)
        injected = grp["inject_payload"].any()
        lines.append(f"\nCondition: {cond}  (n={n} trials, injected={injected})")
        lines.append(f"  Utility Score        mean={grp['utility_score'].mean():.3f}  "
                     f"sd={grp['utility_score'].std():.3f}  "
                     f"min={grp['utility_score'].min():.2f}  max={grp['utility_score'].max():.2f}")
        lines.append(f"  Task Completion      {grp['task_completed'].mean():.1%}")
        lines.append(f"  ASR                  {grp['target_execution_asr'].mean():.1%}")
        lines.append(f"  Participation Rate   mean={grp['participation_rate'].mean():.1%}")
        lines.append(f"  Posts per trial      mean={grp['n_posted'].mean():.1f}")
        lines.append(f"  Emails per trial     mean={grp['n_sent_email'].mean():.1f}")
        lines.append(f"  Errors per trial     mean={grp['n_errored'].mean():.1f}")
        lines.append(f"  Mean post length     {grp['mean_post_length'].mean():.0f} chars")

        if injected and grp["attack_type"].notna().any():
            lines.append(f"  --- By attack type ---")
            for at, ag in grp.groupby("attack_type"):
                lines.append(f"    {at:30s}  ASR={ag['target_execution_asr'].mean():.1%}  n={len(ag)}")

    if df_agents is not None and len(df_agents):
        lines.append("\n" + "=" * 60)
        lines.append("Agent-level participation by role (across all conditions)")
        lines.append("=" * 60)
        role_stats = df_agents.groupby(["condition", "agent_id"]).agg(
            n_trials=("trial_id", "count"),
            post_rate=("posted", "mean"),
            mean_post_length=("post_length", "mean"),
            email_rate=("sent_email", "mean"),
        ).reset_index()
        lines.append(role_stats.to_string(index=False))

    lines.append("")
    return "\n".join(lines)
