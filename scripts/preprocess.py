import pandas as pd
import numpy as np
import csv
DATA_FOLDER = '../data'

def load(filename):
    try:
        return pd.read_csv(f'../data/{filename}', delimiter='|')
    except:
        print(f'{DATA_FOLDER}/{filename}')
        print()




def save_csv(data:pd.DataFrame, output_filepath:str, delimiter:str):
    """
    Reads a CSV file delimited by '|' (pipe) and writes its content
    to a new CSV file delimited by ',' (comma).

    Args:
        input_filepath (str): The path to the input pipe-delimited CSV file.
        output_filepath (str): The path where the new comma-delimited CSV
                                file will be saved.
    """

    with open(output_filepath, 'w', newline='', encoding='utf-8') as outfile:
                # Create a CSV writer object (default delimiter is comma)
            comma_writer = csv.writer(outfile, delimiter='|')

                # Iterate over each row from the input file
            for row in data:
                    # Write the row to the output file
                comma_writer.writerow(row)
    
    print(f"Successfully converted to '{output_filepath}'")

def convert_pipe_to_comma_csv(input_filepath, output_filepath):
    """
    Reads a CSV file delimited by '|' (pipe) and writes its content
    to a new CSV file delimited by ',' (comma).

    Args:
        input_filepath (str): The path to the input pipe-delimited CSV file.
        output_filepath (str): The path where the new comma-delimited CSV
                                file will be saved.
    """
    
        # Open the input file for reading with the pipe delimiter
    with open(input_filepath, 'r', newline='', encoding='utf-8') as infile:
            # Create a CSV reader object, specifying the pipe delimiter
        pipe_reader = csv.reader(infile, delimiter='|')

            # Open the output file for writing with the default comma delimiter
        with open(output_filepath, 'w', newline='', encoding='utf-8') as outfile:
                # Create a CSV writer object (default delimiter is comma)
            comma_writer = csv.writer(outfile, delimiter='|')

                # Iterate over each row from the input file
            for row in pipe_reader:
                    # Write the row to the output file
                comma_writer.writerow(row)
    
    print(f"Successfully converted '{input_filepath}' to '{output_filepath}'")


def find_and_replace_outliers_with_median(df, cols, threshold=3):
    """
    Detects outliers in specified numeric columns of a DataFrame using the z-score method and replaces them with the column median.
    Parameters:
        df (pd.DataFrame): The input DataFrame to process.
        cols (list of str): List of column names to check for outliers and replace them.
        threshold (float, optional): The z-score threshold to identify outliers. Default is 3.
    Returns:
        pd.DataFrame: A copy of the DataFrame with outliers in the specified columns replaced by the median value of each column.
    Notes:
        - Only numeric columns are processed; non-numeric columns are skipped with a warning.
        - If a column's standard deviation is zero, outlier detection is skipped for that column.
        - Outliers are defined as values with an absolute z-score greater than the specified threshold.
        - The function prints progress and warnings during execution.
    """
    df_cleaned = df.copy()  # Create a copy to avoid modifying the original DataFrame

    print(f"Processing columns: {cols}")

    for col in cols:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in DataFrame. Skipping.")
            continue

        # Ensure the column is numeric
        if not pd.api.types.is_numeric_dtype(df_cleaned[col]):
            print(f"Warning: Column '{col}' is not numeric. Skipping outlier detection/replacement.")
            continue

        col_mean = df_cleaned[col].mean()
        col_std = df_cleaned[col].std()
        if col_std == 0:
            print(f"Warning: Standard deviation is zero for column '{col}'. Skipping outlier detection/replacement.")
            continue

        z_scores = (df_cleaned[col] - col_mean) / col_std
        outlier_mask = z_scores.abs() > threshold
        outlier_indices_col = df_cleaned.index[outlier_mask]

        if len(outlier_indices_col) == 0:
            print(f"No outliers found in column '{col}' using z-score threshold {threshold}.")
            continue

        print(f"Found {len(outlier_indices_col)} outliers in column '{col}'.")
        median_value = df_cleaned[col].median()
        print(f"Median value for '{col}' (used for replacement): {median_value}")

        df_cleaned.loc[outlier_indices_col, col] = median_value
        print(f"Outliers in column '{col}' replaced with median.")
        

    return df_cleaned


def find_outliers(df:pd.DataFrame, threshold=3):
    """
    Identifies columns in a DataFrame that contain outliers based on the z-score method.

    Parameters:
        df (pd.DataFrame): The input DataFrame to check for outliers.
        threshold (float, optional): The z-score threshold to use for identifying outliers. 
            Values with absolute z-score greater than this threshold are considered outliers. Default is 3.

    Returns:
        list: A list of column names that contain outliers.

    Notes:
        - Only numeric columns are considered for outlier detection.
        - Columns with zero standard deviation are skipped.
        - Prints warnings for non-numeric columns and columns with zero standard deviation.
        - Prints the number of outliers found per column or a message if none are found.
    """
    cols = []
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"Warning: Standard deviation is zero for column '{col}'. Skipping outlier detection/replacement.")
            continue
        col_mean = df[col].mean()
        col_std = df[col].std()
        if col_std == 0:
            print(f"Warning: Standard deviation is zero for column '{col}'. Skipping outlier detection/replacement.")
            continue

        z_scores = (df[col] - col_mean) / col_std
        outlier_mask = z_scores.abs() > threshold
        outlier_indices_col = df.index[outlier_mask]
        if len(outlier_indices_col) == 0:
            print(f"No outliers found in column '{col}' using z-score threshold {threshold}.")
            continue
        cols.append(col)
        print(f"Found {len(outlier_indices_col)} outliers in column '{col}'.")
    return cols

def IQR_outlier(df: pd.DataFrame, cols: list[str] = None):
    """
    Identifies outliers in specified columns of a DataFrame using the IQR method.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        cols (list of str, optional): List of column names to check for outliers. If None or empty, all columns are checked.

    Returns:
        pd.DataFrame: The original DataFrame (no modifications).
    """
    if not cols:
        cols = df.columns.tolist()
    df_no_outliers = None
    for col in cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"Warning: Column '{col}' is not numeric. Skipping outlier detection.")
            continue

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        df_no_outliers = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        if not outliers.empty:
            print(f"Found {len(outliers)} outliers in column '{col}' using IQR method.")
        else:
            print(f"No outliers found in column '{col}' using IQR method.")
    return df_no_outliers
     

def find_columns_with_missing_value(df:pd.DataFrame, threshold=0.05)->list:
    """
    Identifies columns in a DataFrame with a proportion of missing values above a specified threshold.
    Parameters:
        df (pd.DataFrame): The input DataFrame to analyze for missing values.
        threshold (float, optional): The minimum proportion (between 0 and 1) of missing values required for a column to be considered as having excessive missing data. Defaults to 0.05 (5%).
    Returns:
        list: A list of column names where the proportion of missing values exceeds the given threshold.
    Prints:
        A message indicating that columns above the threshold are being returned.
    """
    null_columns = df.isnull().sum()
    total_row = len(df)
    null_percentage = (null_columns/total_row)*100;
    missing_columns = df.columns[null_percentage > threshold]
    print('columns above the threshold')
    return missing_columns.to_list()

def normalize_date(df:pd.DataFrame, date_col:str):
    df[date_col] = to_datetime(df,date_col).dt.normalize()
    return df

def to_datetime(df:pd.DataFrame, date_col:str):
    df[date_col] =  pd.to_datetime(df[date_col])
    return df

    
def load_data(path:str):
    """
    Loads data from a CSV file at the specified path, parsing the 'Timestamp' column as dates.

    Args:
        path (str): The file path to the CSV file.

    Returns:
        pandas.DataFrame: The loaded data with parsed dates if successful.
        None: If the provided path is not a string.

    Raises:
        None: Prints an error message if the path is not a string.
    """
    try:
        return pd.read_csv(path, parse_dates=['Timestamp']);
    except TypeError:
        print("Path not a string")
        return
    

def drop_column(df:pd.DataFrame, cols:list):
    """
    Drops specified columns from a pandas DataFrame, forward-fills missing values, and resets the index.

    Parameters:
        df (pd.DataFrame): The input DataFrame from which columns will be dropped.
        cols (list): A list of column names to be dropped from the DataFrame.

    Returns:
        pd.DataFrame: The modified DataFrame with specified columns dropped, missing values forward-filled, and index reset.
    """
    df.drop(cols, axis=1, inplace=True)
        
    return df

def convert_money_tofloat(value:object):
    """
    Converts a monetary value to a float-compatible string by removing commas and handling NaN values.

    Args:
        value (object): The monetary value to convert. Can be a string, float, or other object.

    Returns:
        str: The value as a string with commas removed. If the input is NaN, returns '0.0'.
    """
    if(str(value) == 'nan' or str(value) == 'None'):
        value = '0.0'
    return ''.join(str(value).split(','))