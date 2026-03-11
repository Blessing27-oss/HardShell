"""statsmodels OLS for TCR/Utility Tax."""
# hardshell/analysis/regressions.py
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

def calculate_utility_tax(df: pd.DataFrame):
    """
    Runs a linear regression to quantify the 'Utility Tax'.
    DV: task_completed (TCR)
    IV: condition (Baseline, Perimeter, Zero-Trust)
    """
    # Convert 'condition' into categorical factors (dummies)
    # Baseline is the reference group
    df['condition'] = pd.Categorical(df['condition'], categories=['1', '2', '3'])
    
    # Model: TCR ~ Condition
    model = smf.ols(formula="task_completed ~ condition", data=df).fit()
    
    # The coefficient for 'zero_trust' is your mathematical Utility Tax.
    return model