
import pandas as pd
import numpy as np

def resample_time_series(time_series, desired_freq, aggregation_method='last'):
    """
    Resample a time series to the desired frequency with specified aggregation method.

    Parameters:
    - time_series: pandas Series or DataFrame with datetime index.
    - desired_freq: string representing the desired frequency (e.g., 'M' for monthly).
    - aggregation_method: string representing the aggregation method ('last', 'mean', 'median', etc.).

    Returns:
    - Resampled time series.
    """
    # Ensure the time series is a DataFrame with a datetime index
    if not isinstance(time_series, pd.DataFrame):
        time_series = pd.DataFrame(time_series)

    # Resample the time series
    if aggregation_method == 'last':
        resampled_data = time_series.resample(desired_freq).last()
    elif aggregation_method == 'mean':
        resampled_data = time_series.resample(desired_freq).mean()
    elif aggregation_method == 'median':
        resampled_data = time_series.resample(desired_freq).median()
    else:
        raise ValueError("Unsupported aggregation method. Choose from 'last', 'mean', 'median', etc.")

    return resampled_data




def fillna_time_series(time_series, fill_method='forward_fill', interpolation_method=None, **kwargs):
    """
    Fill NaN values in a time series using the specified method or interpolation method.

    Parameters:
    - time_series: pandas Series or DataFrame with datetime index.
    - fill_method: string representing the fill method ('forward_fill', 'backward_fill', 'mean_fill', 'median_fill', etc.).
    - interpolation_method: string representing the interpolation method ('linear', 'polynomial', 'spline', etc.).
      If interpolation_method is provided, it takes precedence over fill_method.
    - **kwargs: Additional keyword arguments to customize the interpolation method.

    Returns:
    - Time series with NaN values filled.
    """
    # Ensure the time series is a DataFrame with a datetime index
    if not isinstance(time_series, pd.DataFrame):
        time_series = pd.DataFrame(time_series)

    if interpolation_method:
        filled_data = time_series.interpolate(method=interpolation_method, **kwargs)
    elif fill_method == 'forward_fill':
        filled_data = time_series.fillna(method='ffill')
    elif fill_method == 'backward_fill':
        filled_data = time_series.fillna(method='bfill')
    elif fill_method == 'mean_fill':
        filled_data = time_series.fillna(time_series.mean())
    elif fill_method == 'median_fill':
        filled_data = time_series.fillna(time_series.median())
    else:
        raise ValueError("Unsupported fill method or interpolation method.")

    return filled_data

def transform_time_series(time_series, transformation_method='log', lag_periods=None, diff_periods=None):
    """
    Perform transformations on a time series.

    Parameters:
    - time_series: pandas Series or DataFrame with datetime index.
    - transformation_method: string representing the transformation method ('log', 'diff', 'lag', etc.).
    - lag_periods: integer specifying the number of lag periods for the 'lag' transformation.
    - diff_periods: integer specifying the number of periods for differencing in the 'diff' transformation.

    Returns:
    - Transformed time series.
    """
    # Ensure the time series is a DataFrame with a datetime index
    if not isinstance(time_series, pd.DataFrame):
        time_series = pd.DataFrame(time_series)

    if transformation_method == 'log':
        transformed_data = time_series.apply(lambda x: x.apply(lambda y: None if pd.isna(y) else (None if y <= 0 else pd.np.log(y))))
    elif transformation_method == 'lag':
        if lag_periods is None:
            raise ValueError("Specify the number of lag periods for the 'lag' transformation.")
        transformed_data = time_series.shift(lag_periods)
    elif transformation_method == 'diff':
        if diff_periods is None:
            raise ValueError("Specify the number of periods for differencing in the 'diff' transformation.")
        transformed_data = time_series.diff(periods=diff_periods)
    else:
        raise ValueError("Unsupported transformation method. Choose from 'log', 'diff', 'lag', etc.")

    return transformed_data