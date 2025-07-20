# README.md - EDA.ipynb 

This notebook `EDA.ipynb` (Exploratory Data Analysis) is designed to perform a comprehensive initial analysis of fraud detection datasets. It covers data loading, preprocessing, outlier detection, and univariate/multivariate analysis, preparing the data for subsequent model training.

## Table of Contents

1. [Setup and Imports](https://www.google.com/search?q=%231-setup-and-imports "null")
    
2. [Data Loading](https://www.google.com/search?q=%232-data-loading "null")
    
3. [Data Preprocessing](https://www.google.com/search?q=%233-data-preprocessing "null")
    
4. [Outlier Detection and Handling](https://www.google.com/search?q=%234-outlier-detection-and-handling "null")
    
5. [Univariate Analysis](https://www.google.com/search?q=%235-univariate-analysis "null")
    
6. [Multivariate Analysis](https://www.google.com/search?q=%236-multivariate-analysis "null")
    
7. [Feature Engineering Pipeline](https://www.google.com/search?q=%237-feature-engineering-pipeline "null")
    

## 1. Setup and Imports

This section initializes the environment by importing necessary libraries and setting up the path to the data directory.

```
import pandas as pd
import sys
# Add the 'scripts' directory to the system path to import custom modules
sys.path.append('../scripts')
# Import plotting functions from the 'plots' module
from plots import plot_histogram, plot_boxplot, calculate_correlations, generate_all_categorical_crosstabs, correlation_matrix
# Import preprocessing functions from the 'preprocess' module
from preprocess import normalize_date, to_datetime, find_outliers, find_and_replace_outliers_with_median
```

## 2. Data Loading

This section defines the path to the data directory and loads the three primary datasets used in this analysis: `Fraud_Data.csv`, `IpAddress_to_Country.csv`, and `creditcard.csv`.

```
DATA_PATH = '../data'

# Load the main fraud data
fraud_data = pd.read_csv(f'{DATA_PATH}/Fraud_Data.csv')

# Load IP address to country mapping data
ip_data = pd.read_csv(f'{DATA_PATH}/IpAddress_to_Country.csv')

# Load credit card transaction data (likely for a separate or comparative analysis)
credit_carddata = pd.read_csv(f'{DATA_PATH}/creditcard.csv')
```

## 3. Data Preprocessing

This section focuses on converting relevant columns to datetime objects to facilitate time-based analysis and feature engineering.

```
# Convert 'purchase_time' and 'signup_time' columns in fraud_data to datetime objects
fraud_data = to_datetime(fraud_data,'purchase_time')
fraud_data = to_datetime(fraud_data, 'signup_time')
```

## 4. Outlier Detection and Handling

This section identifies and handles outliers in the `fraud_data` dataset. Specifically, it replaces outliers in the 'purchase_value' column with its median and then identifies outliers across various other columns.

```
# Find and replace outliers in 'purchase_value' column with the median value
# The function `find_and_replace_outliers_with_median` is assumed to be defined in 'preprocess.py'
fraud_data = find_and_replace_outliers_with_median(fraud_data, ['purchase_value'])

# Find and print outliers for all columns in the fraud_data DataFrame
# The function `find_outliers` is assumed to be defined in 'preprocess.py'
find_outliers(fraud_data)
```

## 5. Univariate Analysis

This section performs univariate analysis by plotting box plots for numerical columns and histograms for categorical columns to understand their distributions.

```
# Plot box plots for numerical columns in fraud_data
# The function `plot_boxplot` is assumed to be defined in 'plots.py'
# The empty list `[]` suggests that all numerical columns are considered by default within the function.
plot_boxplot(fraud_data,[])

# Plot histograms for categorical columns in fraud_data
# The function `plot_histogram` is assumed to be defined in 'plots.py'
# The empty list `[]` suggests that all categorical columns are considered by default within the function.
plot_histogram(fraud_data, [])
```

## 6. Multivariate Analysis

This section delves into the relationships between different variables through correlation analysis and cross-tabulations.

```
# Calculate and display correlations for numerical columns
# The function `calculate_correlations` is assumed to be defined in 'plots.py'
calculate_correlations(fraud_data, [])

# Generate and display cross-tabulations for all categorical columns
# The function `generate_all_categorical_crosstabs` is assumed to be defined in 'plots.py'
generate_all_categorical_crosstabs(fraud_data, [])

# Generate and display a correlation matrix for numerical columns
# The function `correlation_matrix` is assumed to be defined in 'plots.py'
correlation_matrix(fraud_data, [])
```

## 7. Feature Engineering Pipeline

This section demonstrates the application of a feature engineering pipeline to the `fraud_data`. It separates numerical and categorical columns and then processes the data using a custom `process_frude` function.

```
# Import the `process_frude` function from `feature_pipeline.py`
from feature_pipeline import process_frude

numerical_cols = []
category_cols = []

# Iterate through columns to separate numerical and categorical features
for col in fraud_data.columns:
    if col == 'class':  # Skip the target variable 'class'
        continue
    if pd.api.types.is_numeric_dtype(fraud_data[col]):
        numerical_cols.append(col)
    else:
        category_cols.append(col)

# Apply the feature engineering pipeline
# The `process_frude` function is assumed to perform operations like encoding, scaling, and SMOTE
train_setup = process_frude(fraud_data, numerical_cols, category_cols)

# Display class distribution before SMOTE (from the output of the notebook)
# class
# 0    190144
# 1      9906
# Name: count, dtype: int64
# --------------------------------------------------

# Display class distribution after SMOTE (from the output of the notebook)
# class
# 0    95872
# 1    95872
# Name: count, dtype: int64
# class
# 0    50.0
# 1    50.0
# Name: proportion, dtype: float64
# --------------------------------------------------

# Confirm successful completion of the feature engineering pipeline
# Feature Engineering Pipeline successfully created and applied. Data is ready for model training. ✨
```