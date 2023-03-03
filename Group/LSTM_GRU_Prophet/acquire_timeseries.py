import pandas as pd
import datetime


def fill_missing_weeks_with_zeros(input_series):
    """
    Parameters
    ----------
    input_series: pandas.Series with index of datetime and some numeric
        values

    Returns
    -------
    pandas.Series of the same format, but with all calendar weeks (weeks, that
        are missed in the input series have the value 0)
    """
    start_ts = input_series.index[0]
    finish_ts = input_series.index[-1]
    current_ts = start_ts
    out_ts = pd.Series(dtype=input_series.dtype)
    while current_ts <= finish_ts:
        if current_ts in input_series.index:
            out_ts[current_ts] = input_series[current_ts]
        else:
            out_ts[current_ts] = 0
        current_ts = current_ts + datetime.timedelta(days=7)
    return out_ts


def acquire_timeseries(
        input_csv = 'group_samples/group_popularity_until_13.02.2023.csv',
        input_format = 'DAOD',
        input_subformat = 'HIGG',
        input_project = 'mc16_13TeV'
    ):
    """
    Acquire time series from table of data

    Parameters
    ----------
    input_csv (str): File with the table of data.
        Default: 'group_samples/group_popularity_until_13.02.2023.csv'
    input_format (str): Format of data in datasets (AOD, DAOD, etc.).
        Default: 'DAOD'
    input_subformat (str): a string, which is used to filter datasets according
        to the 'input_format_desc' column.
        Default: 'HIGG'
    input_project (str): project, which datasets belong to (filtering in the
        corresponding column)
        Default: 'mc16_13TeV'
    Returns
    -------
    pandas.Series with index of type datetime (weekly time series);
    values are n_tasks at particular week (number of tasks which had access
    to the datasets of the chosen group)
    """
    df = pd.read_csv(input_csv)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['input_format_desc'].fillna('-', inplace=True)

    ds_group_ts = df[(df['input_format_desc'].str.contains(input_subformat)) &
                     (df['input_format_short'] == input_format) &
                     (df['input_project'] == input_project)][
        ['datetime', 'n_tasks']]
    ds_group_ts = ds_group_ts.groupby('datetime')['n_tasks'].sum()
    ds_group_ts = fill_missing_weeks_with_zeros(ds_group_ts)

    # discard data before 2018
    ds_group_ts = ds_group_ts[pd.Timestamp('2018-01-01'):]

    return ds_group_ts