# Insurance Risk Analytics - Python Modules

This project contains three main Python modules for data preprocessing, statistical testing, and visualization:

- [`scripts/preprocess.py`](scripts/preprocess.py): Data cleaning, transformation, and utility functions.
- [`scripts/stat_test.py`](scripts/stat_test.py): Statistical tests for comparing groups.
- [`scripts/plots.py`](scripts/plots.py): Visualization utilities for exploratory data analysis.

---

## `preprocess.py`

This module provides functions for cleaning, transforming, and preparing insurance data for analysis and modeling.

### Key Functions

- **load_data(path)**  
  Loads a CSV file from the given path, parsing the 'Timestamp' column as dates.

- **drop_column(df, cols)**  
  Drops specified columns from a DataFrame.

- **find_columns_with_missing_value(df, threshold=0.05)**  
  Returns a list of columns with a proportion of missing values above the given threshold (default 5%).

- **find_outliers(df, threshold=3)**  
  Identifies columns containing outliers based on the z-score method (default threshold 3).

- **find_and_replace_outliers_with_median(df, cols, threshold=3)**  
  Detects outliers in specified numeric columns using the z-score method and replaces them with the column median.

- **IQR_outlier(df, cols=None)**  
  Identifies outliers in specified columns using the IQR method. If `cols` is None, checks all columns.

- **normalize_date(df, date_col)**  
  Normalizes a date column to remove time information.

- **convert_money_tofloat(value)**  
  Converts a monetary value to a float-compatible string, handling commas and NaNs.

---

## `stat_test.py`

This module provides statistical tests for comparing groups.

### Key Functions

- **t_test_for_equivalence(group1_df, group2_df, column_name, alpha, test_name="")**  
  Performs an independent samples t-test for equivalence between two groups for a numerical column.  
  Returns a tuple: (is_significant, p_value).

- **chi_squared_test_for_equivalence(group1_df, group2_df, column_name, alpha, test_name="")**  
  Performs a Chi-Squared test for independence between two groups for a categorical column.  
  Returns a tuple: (is_significant, p_value).

---

## `feature_pipeline.py`

This module provides pipeline to run feature engineering 

### Key Functions

- **process_frude(fraud_data, numerical_cols, categories_col)**
    init stage for pipeline to feature engineer fraud_data

## `plots.py`

This module provides plotting utilities for exploratory data analysis.

### Key Functions

- **plot_boxplot(df, cols)**  
  Plots boxplots for the specified numeric columns. If `cols` is empty, plots all columns.

- **plot_histogram(df, cols)**  
  Plots histograms for numeric columns and count plots for categorical columns. If `cols` is empty, plots all columns.

- **correlation_matrix(df, cols, name)**  
  Displays a heatmap of the correlation matrix for the specified columns.

---

## Usage

Import the functions you need in your analysis scripts or notebooks:

```python
from preprocess import (
    load_data, drop_column, find_columns_with_missing_value, find_outliers,
    find_and_replace_outliers_with_median, IQR_outlier, normalize_date, convert_money_tofloat
)
from stat_test import (
    t_test_for_equivalence, chi_squared_test_for_equivalence
)
from plots import plot_boxplot, plot_histogram, correlation_matrix