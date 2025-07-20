import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from itertools import combinations
def plot_boxplot(df:pd.DataFrame, cols:list[str]):
    if len(cols) == 0:
        cols = df.columns
    for col in cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            plt.boxplot(df[col])
            plt.xlabel(col, fontsize=8)
            plt.figure(figsize=(10,7))
            plt.show()

def plot_histogram(df:pd.DataFrame,cols:list[str], top_n:int=20):
    if len(cols) == 0:
        cols = df.columns
    num_plots = len(cols)
     # Calculate grid dimensions: roughly square layout
    num_rows = int(num_plots**0.5)
    num_cols = (num_plots + num_rows - 1)
    plt.figure(figsize=(num_cols * 5, num_rows * 4))
    for i, col in enumerate(cols):
        ax = plt.subplot(num_rows, num_cols, i + 1) # Create a subplot for each plot
        
        if pd.api.types.is_numeric_dtype(df[col]):
            sns.histplot(df[col].dropna(), kde=True, bins=30, ax=ax)
            ax.set_title(f'Distribution of {col}', fontsize=10)
            ax.set_xlabel(col, fontsize=8)
            ax.set_ylabel('Frequency', fontsize=8)
            ax.tick_params(axis='both', which='major', labelsize=7)
        else:
            value_counts = df[col].value_counts()
            if len(value_counts) > top_n:
                # Select top_n values and group the rest as 'Other'
                top_values = value_counts.head(top_n).index
                df_to_plot = df[col].apply(lambda x: x if x in top_values else 'Other')
                
                sns.countplot(data=pd.DataFrame(df_to_plot), x=col, 
                              order=top_values.tolist() + ['Other'] if 'Other' in df_to_plot.unique() else top_values.tolist(), 
                              palette='viridis', ax=ax)
                ax.set_title(f'Counts of {col} (Top {top_n} + Others)', fontsize=10)
            else:
                # Plot all categories if not too many
                sns.countplot(data=df, x=col, order=value_counts.index, palette='viridis', ax=ax)
                ax.set_yscale('log')
                ax.set_title(f'Counts of {col}', fontsize=10)
            
            ax.set_xlabel(col, fontsize=8)
            ax.set_ylabel('Count', fontsize=8)
            ax.tick_params(axis='y', which='major', labelsize=7)
            plt.xticks(rotation=45, ha='right', fontsize=7)

    plt.tight_layout()
    plt.show()

def calculate_correlations(df, method='all'):
    """
    Calculates Pearson and/or Spearman correlation coefficients across a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing numerical data.
        method (str): The type of correlation to calculate.
                      'pearson': Calculates Pearson correlation only.
                      'spearman': Calculates Spearman correlation only.
                      'all': Calculates both Pearson and Spearman correlations (default).

    Returns:
        dict: A dictionary containing correlation matrices.
              - 'pearson_corr' (pd.DataFrame) if Pearson is requested.
              - 'spearman_corr' (pd.DataFrame) if Spearman is requested.
              Returns an empty dictionary if the input DataFrame is empty
              or contains no numeric columns.
    """
    if df.empty:
        print("Warning: Input DataFrame is empty.")
        return {}

    # Select only numeric columns for correlation calculation
    numeric_df = df.select_dtypes(include=np.number)

    if numeric_df.empty:
        print("Warning: No numeric columns found in the DataFrame. Cannot calculate correlations.")
        return {}

    results = {}

    if method == 'pearson' or method == 'all':
        pearson_corr_matrix = numeric_df.corr(method='pearson')
        results['pearson_corr'] = pearson_corr_matrix
        print("\n--- Pearson Correlation Matrix ---")
        print(pearson_corr_matrix)

    if method == 'spearman' or method == 'all':
        spearman_corr_matrix = numeric_df.corr(method='spearman')
        results['spearman_corr'] = spearman_corr_matrix
        print("\n--- Spearman Correlation Matrix ---")
        print(spearman_corr_matrix)

    if method not in ['pearson', 'spearman', 'all']:
        print(f"Warning: Invalid method '{method}'. Please choose 'pearson', 'spearman', or 'all'.")

    return results

def correlation_matrix(df:pd.DataFrame, cols:list, name:str):
    """
    Generates and displays a correlation matrix heatmap for the specified columns of a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        cols (list): List of column names to include in the correlation matrix.
        name (str): A name or label to use in the plot title.

    Displays:
        A heatmap plot of the correlation matrix for the specified columns.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', square=True)
    plt.title(f'Correlation Matrix for {name}')
    plt.show()

def generate_all_categorical_crosstabs(df, include_margins=False, normalize_method=None):
    """
    Identifies all categorical columns in a DataFrame and generates contingency tables
    for every unique pair of these columns.

    Args:
        df (pd.DataFrame): The input DataFrame.
        include_margins (bool, optional): If True, adds row/column totals to the crosstabs. Defaults to False.
        normalize_method (str, optional): Normalization method for the crosstabs.
                                          - None: Display counts (default).
                                          - 'all': Normalize by grand total.
                                          - 'index': Normalize each row by its sum.
                                          - 'columns': Normalize each column by its sum.

    Returns:
        dict: A dictionary where keys are strings representing the pair of columns
              (e.g., 'Column1_vs_Column2') and values are the corresponding
              pandas DataFrame contingency tables. Returns an empty dictionary
              if no categorical columns are found or if the DataFrame is empty.
    """
    if df.empty:
        print("Warning: Input DataFrame is empty. No crosstabs can be generated.")
        return {}

    # Identify categorical columns (object dtype for strings, or 'category' dtype)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if not categorical_cols or len(categorical_cols) < 2:
        print("Warning: Less than two categorical columns found. Cannot generate crosstabs for pairs.")
        if categorical_cols:
            print(f"Found categorical columns: {categorical_cols}")
        else:
            print("No categorical columns found.")
        return {}

    crosstabs = {}
    print(f"Found {len(categorical_cols)} categorical columns: {categorical_cols}")
    print("\nGenerating contingency tables for all unique pairs...")

    # Generate all unique pairs of categorical columns
    for col1, col2 in combinations(categorical_cols, 2):
        print(f"\n--- Crosstab: '{col1}' vs '{col2}' ---")
        try:
            # Create the contingency table
            crosstab_df = pd.crosstab(
                index=df[col1],
                columns=df[col2],
                margins=include_margins,
                normalize=normalize_method
            )
            crosstabs[f"{col1}_vs_{col2}"] = crosstab_df
            print(crosstab_df)
        except Exception as e:
            print(f"Error generating crosstab for '{col1}' vs '{col2}': {e}")

    return crosstabs




