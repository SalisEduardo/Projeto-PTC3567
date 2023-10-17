
import pandas as pd
import numpy as np

from functools import reduce
from utils.extract_data import get_data_dictionary

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


def fillna_time_series(time_series, fill_method='forward_fill', interpolation_method=None, **interpolation_kwargs):
    """
    Fill NaN values in a time series using the specified method or interpolation method.

    Parameters:
    - time_series: pandas Series or DataFrame with datetime index.
    - fill_method: string representing the fill method ('forward_fill', 'backward_fill', 'mean_fill', 'median_fill', etc.).
    - interpolation_method: string representing the interpolation method ('linear', 'polynomial', 'spline', etc.).
      If interpolation_method is provided, it takes precedence over fill_method.
    - **interpolation_kwargs: Additional keyword arguments to customize the interpolation method.

    Returns:
    - Time series with NaN values filled.
    """
    # Ensure the time series is a DataFrame with a datetime index
    if not isinstance(time_series, pd.DataFrame):
        time_series = pd.DataFrame(time_series)

    if interpolation_method:
        filled_data = time_series.interpolate(method=interpolation_method, **interpolation_kwargs)
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


def gathering_data_for_modeling(tickers_dataset,
                                infos_dataset,
                                chosen_target,
                                frequency_ajustment='higher_to_lower',
                                begin_date=None,
                                end_date=None,
                                include_seasonal_ajusted_series=True,
                                agg_method = "last",
                                fillna_method ='forward_fill',
                                **dict_kwargs):
    
    infos_dictionary = get_data_dictionary(tickers_dataset,infos_dataset,**dict_kwargs)
    

    category_frequencies = {"Y": 1,"Q": 2,"M": 3,"D":4}
    
    # Define a custom key function to map category labels to frequencies
    
    def custom_key(category):
        return category_frequencies.get(category, 0) 
    
    max_frequency = max(infos_dataset['frequency_short'], key=custom_key)
    min_frequeny =  min(infos_dataset['frequency_short'], key=custom_key)

    if frequency_ajustment == 'higher_to_lower':
        chosen_freq = max_frequency
    elif frequency_ajustment == 'lower_to_higher':
        chosen_freq = min_frequeny
    elif frequency_ajustment == 'target_frequency':
        chosen_freq = infos_dataset.query(f"id == '{chosen_target}' ")['frequency_short'].unique()[0]
    else:
        raise ValueError("Unsupported frequency ajustment. Choose from ''higher_to_lower' ,'lower_to_higher' ,'target_frequency'")

    
    target_series = infos_dictionary[chosen_target]['ts'].copy()


    trasnformed = []



    for k in infos_dictionary.keys():
        
        
        serie = infos_dictionary[k]['ts']


        if begin_date is None:
            start = target_series.index.min()
        else:
            start = begin_date
        if end_date is None:
            end = target_series.index.max()
        else:
            end = end_date
        
        serie = serie[start:end]



        if type(agg_method) == str: 
            resampled_ts = resample_time_series(serie,desired_freq=chosen_freq,aggregation_method=agg_method)
            
        elif type(agg_method) == dict:
            for agg in agg_method.keys():
                if k in agg_method[agg]:
                    resampled_ts = resample_time_series(serie,desired_freq=chosen_freq,aggregation_method=agg)
                else:
                    pass
            
        transformed_serie = resampled_ts.copy()
        print(resampled_ts)
        if type(fillna_method) == str: 
            transformed_serie = fillna_time_series(transformed_serie,fill_method=fillna_method)
        elif type(fillna_method) == dict:
            for method in fillna_method.keys():
                if k in fillna_method[agg]:
                    transformed_serie = fillna_time_series(transformed_serie,fill_method=method)
                else:
                    pass
         
        transformed_serie= transformed_serie.reset_index().rename(columns={"index":"date"})

        trasnformed.append(transformed_serie)
    
    
    #df_complete = pd.concat(trasnformed)
    df_complete  = reduce(lambda left, right: pd.merge(left, right, on='date', how='outer'), trasnformed)


    # df_complete=df_complete.reset_index().rename(columns={'index':"date"})
    
    return(df_complete)
        
        