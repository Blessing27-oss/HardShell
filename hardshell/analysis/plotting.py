"""matplotlib + scienceplots publication figures."""
# hardshell/analysis/plotting.py
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_asr_vs_tcr(df: pd.DataFrame, save_path: str):
    """
    Generates a professional bar chart comparing Attack Success Rate (ASR) 
    and Task Completion Rate (TCR) across the three conditions.
    """
    # Set professional style (use 'seaborn-v0_8-paper' if scienceplots isn't installed)
    plt.style.use('seaborn-v0_8-paper')
    
    # Aggregate data
    agg = df.groupby('condition').agg({
        'target_execution_asr': 'mean',
        'task_completed': 'mean'
    }).reset_index()
    
    # Reshape for seaborn
    melted = agg.melt(id_vars='condition', var_name='Metric', value_name='Score')
    melted['Metric'] = melted['Metric'].replace({
        'target_execution_asr': 'Swarm ASR',
        'task_completed': 'Swarm TCR'
    })

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=melted, x='condition', y='Score', hue='Metric', palette='viridis')
    
    plt.title("The Utility Tax: Security vs. Functional Performance", fontsize=14)
    plt.ylabel("Success Rate (0.0 - 1.0)", fontsize=12)
    plt.xlabel("Experimental Condition", fontsize=12)
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()