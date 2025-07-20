from scipy.stats import chi2_contingency, ttest_ind,f_oneway
import pandas as pd
def t_test_for_equivalence(group1_df, group2_df, column_name, alpha, test_name=""):
    """Performs an independent samples t-test for equivalence between groups for a numerical column."""
    if column_name not in group1_df.columns or column_name not in group2_df.columns:
        # print(f"  Skipping '{column_name}': Not found in one or both groups.")
        return False, 1.0 # Return False for significance and p-value 1.0

    # Drop NaNs for the specific column to avoid issues with t-test
    data_g1 = group1_df[column_name].dropna()
    data_g2 = group2_df[column_name].dropna()

    if len(data_g1) < 2 or len(data_g2) < 2: # t-test requires at least 2 samples per group
        # print(f"  Skipping T-test for '{column_name}': Insufficient data after dropping NaNs (Group A: {len(data_g1)}, Group B: {len(data_g2)})")
        return False, 1.0

    t_stat, p_value = ttest_ind(data_g1, data_g2, equal_var=False) # Welch's t-test, more robust for unequal variances
    is_significant = p_value < alpha
    # print(f"  '{column_name}': T-stat={t_stat:.2f}, p-value={p_value:.4f} -> {'Significant' if is_significant else 'Not Significant'}")
    return is_significant, p_value

def chi_squared_test_for_equivalence(group1_df, group2_df, column_name, alpha, test_name=""):
    """Performs a Chi-Squared test for independence between groups for a categorical column."""
    if column_name not in group1_df.columns or column_name not in group2_df.columns:
        # print(f"  Skipping '{column_name}': Not found in one or both groups.")
        return False, 1.0 # Return False for significance and p-value 1.0

    # Ensure the column is treated as categorical
    combined_data = pd.concat([
        group1_df[[column_name]].assign(Group='A'),
        group2_df[[column_name]].assign(Group='B')
    ])
    contingency_table = pd.crosstab(combined_data['Group'], combined_data[column_name])

    # Check if the contingency table is valid for chi-squared test
    if contingency_table.empty or contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
        # print(f"  Skipping Chi-Squared for '{column_name}': Insufficient data or invalid contingency table.")
        return False, 1.0

    # Check for zero marginals (rows/columns sum to zero)
    if (contingency_table.sum(axis=0) == 0).any() or (contingency_table.sum(axis=1) == 0).any():
        # print(f"  Skipping Chi-Squared for '{column_name}': Zero marginals detected, test not applicable.")
        return False, 1.0

    # Handle cases where expected frequencies are too low
    try:
        chi2, p_value, _, _ = chi2_contingency(contingency_table)
    except ValueError as e:
        # print(f"  Chi-Squared test failed for '{column_name}' due to ValueError: {e}. Likely sparse data.")
        return False, 1.0 # Assume non-significant if test cannot be performed

    is_significant = p_value < alpha
    # print(f"  '{column_name}': Chi2={chi2:.2f}, p-value={p_value:.4f} -> {'Significant' if is_significant else 'Not Significant'}")
    return is_significant, p_value