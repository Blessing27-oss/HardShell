"""Decoupled analysis script."""
import logging
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from hardshell.analysis.clustering import compute_semantic_clusters
from hardshell.analysis.extract_metrics import load_transcripts
from hardshell.analysis.plotting import plot_asr_vs_tcr
from hardshell.analysis.regressions import calculate_utility_tax

log = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    logs_dir = Path(cfg.directories.logs)
    tables_dir = Path(cfg.directories.tables)
    plots_dir = Path(cfg.directories.plots)

    # --- 1. Discover all JSONL transcripts ---
    jsonl_files = sorted(logs_dir.glob("**/*.jsonl"))
    if not jsonl_files:
        log.error(f"No JSONL files found in {logs_dir}. Run run_experiment.py first.")
        return
    log.info(f"Found {len(jsonl_files)} transcript file(s).")

    # --- 2. Load and combine all trials into one DataFrame ---
    df = pd.concat(
        [load_transcripts(str(f)) for f in jsonl_files],
        ignore_index=True,
    )
    log.info(f"Loaded {len(df)} trials across {df['condition'].nunique()} condition(s).")

    # --- 3. Summary table (ASR per condition) ---
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary = df.groupby("condition").agg(
        total_trials=("trial_id", "count"),
        swarm_asr=("target_execution_asr", "mean"),
    ).reset_index()
    summary_path = tables_dir / "results_summary.csv"
    summary.to_csv(summary_path, index=False)
    log.info(f"Summary table saved to {summary_path}\n{summary.to_string(index=False)}")

    # --- 4. Semantic clustering on Agent A's internal messages ---
    text_col = cfg.analysis.clustering.text_column
    if text_col in df.columns and df[text_col].notna().any():
        df = compute_semantic_clusters(
            df,
            text_column=text_col,
            n_clusters=cfg.analysis.clustering.n_clusters,
        )
        cluster_path = tables_dir / "clusters.csv"
        df.to_csv(cluster_path, index=False)
        log.info(f"Cluster assignments saved to {cluster_path}")
    else:
        log.warning(f"Column '{text_col}' not found or empty — skipping clustering.")

    # --- 5. Regression (Utility Tax) ---
    if df["condition"].nunique() > 1 and df["task_completed"].notna().any():
        model = calculate_utility_tax(df.copy())
        log.info(f"\n--- Utility Tax Regression ---\n{model.summary()}")
        reg_path = tables_dir / "regression_summary.txt"
        reg_path.write_text(str(model.summary()))
        log.info(f"Regression summary saved to {reg_path}")
    else:
        log.warning("Skipping regression — need all 3 conditions and task_completed labels.")

    # --- 6. Plot ---
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_path = str(plots_dir / f"asr_vs_tcr.{cfg.analysis.plotting.format}")
    plot_asr_vs_tcr(df, save_path=plot_path)
    log.info(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
