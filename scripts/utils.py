import pandas as pd

def split_columns(df:pd.DataFrame, exclude:list[str]):
    """
    Splits the columns of a DataFrame into numerical and categorical lists based on their data types.

    Columns specified in the `exclude` list will be ignored and not included in either the
    numerical or categorical lists.

    Args:
        df (pd.DataFrame): The input DataFrame whose columns are to be split.
        exclude (list[str]): A list of column names to exclude from the splitting process.
                             These columns will not appear in the returned numerical or
                             categorical lists.

    Returns:
        tuple[list[str], list[str]]: A tuple containing two lists:
                                     - The first list contains the names of numerical columns.
                                     - The second list contains the names of categorical columns.
    """
    numerical_cols = []
    category_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numerical_cols.append(col)
        else:
            category_cols.append(col)
    return (numerical_cols, category_cols)