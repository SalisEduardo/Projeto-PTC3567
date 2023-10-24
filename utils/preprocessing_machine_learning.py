
import pandas as pd
import numpy as np


def preprocess_dataframe(dataframe, preprocess_type='normalize', log_difference_order=1, exclude_columns=[]):
    """
    Preprocess a DataFrame using normalization, standardization, or logarithmic difference transformation.

    Parameters:
    dataframe (pandas.DataFrame): The input DataFrame.
    preprocess_type (str): The type of preprocessing to apply ('normalize', 'standardize', 'log_difference').
    log_difference_order (int): The order of the logarithmic difference transformation (1 for first order, 2 for second order, etc.).
    exclude_columns (list): List of column names to exclude from preprocessing.

    Returns:
    pandas.DataFrame: The preprocessed DataFrame.
    """

    columns_to_preprocess = [col for col in dataframe.columns if col not in exclude_columns]

    if preprocess_type == 'normalize':
        # Normalize the selected columns in the DataFrame
        normalized_df = dataframe.copy()
        normalized_df[columns_to_preprocess] = (dataframe[columns_to_preprocess] - dataframe[columns_to_preprocess].min()) / (dataframe[columns_to_preprocess].max() - dataframe[columns_to_preprocess].min())
        return normalized_df

    elif preprocess_type == 'standardize':
        # Standardize the selected columns in the DataFrame
        standardized_df = dataframe.copy()
        standardized_df[columns_to_preprocess] = (dataframe[columns_to_preprocess] - dataframe[columns_to_preprocess].mean()) / dataframe[columns_to_preprocess].std()
        return standardized_df

    elif preprocess_type == 'log_difference':
        if log_difference_order < 1:
            raise ValueError("log_difference_order should be at least 1.")

        log_diff_df = dataframe.copy()
        log_diff_df[columns_to_preprocess] = np.log(dataframe[columns_to_preprocess])

        if log_difference_order == 1:
            # First-order logarithmic difference
            log_diff_df[columns_to_preprocess] = log_diff_df[columns_to_preprocess].diff()
        else:
            # Higher-order logarithmic difference
            log_diff_df[columns_to_preprocess] = log_diff_df[columns_to_preprocess].diff(log_difference_order)

        return log_diff_df

    else:
        raise ValueError("Invalid preprocess_type. Use 'normalize', 'standardize', or 'log_difference'.")