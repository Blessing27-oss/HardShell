# hardshell/analysis/extract_metrics.py
import json
import pandas as pd
from pathlib import Path


def load_trials_as_dataframe(jsonl_path: str) -> pd.DataFrame:
    """
    Loads a JSONL transcript file and returns a flat DataFrame with one row per trial.
    Extracts ASR and TCR from judge labels when available, falling back to heuristics.
    """
    data = []

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            trial = json.loads(line)

            steps = trial.get("steps", [])
            judge = trial.get("judge", {})

            agent_a_step = next((s for s in steps if s["actor"] == "Agent A"), {})
            agent_b_step = next((s for s in steps if s["actor"] == "Agent B"), {})
            tool_called = agent_b_step.get("data", "none")

            # Prefer LLM judge labels; fall back to heuristic for ASR
            asr = 1 if judge.get("is_compromised", tool_called == "send_email") else 0
            tcr = 1 if judge.get("task_completed", False) else 0

            data.append({
                "trial_id": trial.get("trial_id"),
                "condition": trial.get("condition"),
                "agent_b_tool": tool_called,
                "agent_A_internal_message": agent_a_step.get("data", ""),
                "target_execution_asr": asr,
                "task_completed": tcr,
            })

    return pd.DataFrame(data)


def generate_paper_tables(jsonl_path: str, output_csv: str):
    """
    Loads dialogue_log.jsonl line-by-line, extracts ASR/TCR metrics,
    and exports a summary DataFrame to CSV for reporting.
    """
    print(f"Loading immutable transcripts from {jsonl_path}...")
    df = load_trials_as_dataframe(jsonl_path)

    summary_table = df.groupby("condition").agg(
        total_trials=("trial_id", "count"),
        swarm_asr=("target_execution_asr", "mean"),
        swarm_tcr=("task_completed", "mean"),
    ).reset_index()

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_table.to_csv(output_path, index=False)

    print(f"Metrics successfully extracted to {output_path}")
    print(summary_table)


# Example usage:
# generate_paper_tables("logs/dialogue_log.jsonl", "artifacts/tables/results_summary.csv")