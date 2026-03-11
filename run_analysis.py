"""Decoupled analysis script."""
import pandas as pd
import hydra
from omegaconf import DictConfig
from hardshell.analysis.extract_metrics import load_jsonl_to_df
from hardshell.analysis.clustering import compute_semantic_clusters
from hardshell.analysis.regressions import calculate_utility_tax
from hardshell.analysis.plotting import plot_asr_vs_tcr

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # 1. Load data from the machine-readable JSONL
    log_file = f"{cfg.directories.logs}/dialogue_log.jsonl"
    df = load_jsonl_to_df(log_file)
    
    # 2. Extract Behavioral Archetypes
    df = compute_semantic_clusters(df, text_column='agent_A_internal_message')
    
    # 3. Statistical Analysis
    stats_model = calculate_utility_tax(df)
    with open(f"{cfg.directories.tables}/regression_summary.txt", "w") as f:
        f.write(stats_model.summary().as_text())
    
    # 4. Export Table for Paper
    df.to_csv(f"{cfg.directories.tables}/raw_metrics.csv", index=False)
    
    # 5. Visualizations
    plot_asr_vs_tcr(df, f"{cfg.directories.plots}/main_results.pdf")
    
    print("Analysis complete. Check 'artifacts/' for your submission-ready files.")

if __name__ == "__main__":
    main()