import pandas as pd
import numpy as np
from tqdm import tqdm

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

def transform_filter_data(
    source_csv = 'datasets_popularity_DAOD_PHYS_mc16_13TeV_202308141254.csv',
    finish_date = pd.Timestamp('2022-12-31'),
    horizon_date = pd.Timestamp('2023-01-07'),
    span_threshold = pd.Timedelta('90 days'),
    history_len = pd.Timedelta('365 days'),
    accesses_threshold = 30
):

    df = pd.read_csv(source_csv)
    
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    history = df[df['datetime'] <= finish_date]
    
    target_period = df[(df['datetime'] > finish_date) &
                       (df['datetime'] <= horizon_date)]
    
    history_gb = history.groupby(['datasetname'])
    target_period_gb = target_period.groupby(['datasetname'])
    
    n_accesses = history_gb['datetime'].count()
    access_dates = history_gb['datetime'].apply(lambda x : x.to_numpy())
    n_users = history_gb['username'].nunique()
    n_access_days = history_gb['datetime'].nunique()
    last_date = history_gb['datetime'].max()
    
    future_popularity = target_period_gb['datetime'].count()
    
    objs_df = pd.DataFrame(
        {
            'n_accesses' : n_accesses,
            'access_dates' : access_dates,
            'n_users' : n_users,
            'n_access_days' : n_access_days,
            'last_date' : last_date
        }
    )
    
    objs_df['y'] = 0

    print('filling \'y\' column')    
    for ds_name, ds_features in tqdm(objs_df.iterrows(), total=len(objs_df)):
        if ds_name in future_popularity:
            objs_df.loc[ds_name, 'y'] = future_popularity[ds_name]
    
    objs_df = objs_df[objs_df['n_accesses'] > accesses_threshold]

    print('after filtering by access number: ' +
          f"{len(objs_df[objs_df['y'] > 0])} / {len(objs_df)}")
    
    objs_df['has_history'] = False
    
    ds_histories = {}

    print('filtering by history presence')
    for ds_name, ds_features in tqdm(objs_df.iterrows(), total=len(objs_df)):
        arr = ds_features['access_dates']
        arr.sort()
        latest_start = finish_date - history_len + pd.Timedelta('1 day')
        possible_starts = arr[arr <= latest_start]
    
        # at first, try to use history starting with an access
        for i in range(len(possible_starts) - 1, -1, -1): 
            start = possible_starts[i]
            finish = start + history_len - pd.Timedelta('1 day')
            arr2 = arr[(arr >= start) & (arr <= finish)]
            if arr2[-1] - arr2[0] >= span_threshold:
                objs_df.loc[ds_name, 'has_history'] = True
                ds_histories[ds_name] = (start, arr2)
                break
        else: # if not found, try to use the last interval
            start = latest_start
            finish = finish_date
            arr2 = arr[(arr >= start) & (arr <= finish)]
            if len(arr2) and (arr2[-1] - arr2[0] >= span_threshold):
                objs_df.loc[ds_name, 'has_history'] = True
                ds_histories[ds_name] = (start, arr2)
    
    objs_df = objs_df[objs_df['has_history']]
    objs_df['history'] = ds_histories

    print('after filtering: ' +
          f"{len(objs_df[objs_df['y'] > 0])} / {len(objs_df)}")

    print('transforming date arrays to timeseries')

    bar = tqdm(total=len(objs_df))
    
    def dates_array_to_timeseries(arg):
        start, history_dates = arg
        finish = start + history_len - pd.Timedelta('1 day')
        res = pd.Series(0,
                        index=pd.date_range(start, finish, freq='D'),
                        dtype=np.dtype('int32'))
    
        for d in history_dates:
            res[d] += 1
    
        bar.update()
        return res.to_numpy()
    
    sequences = objs_df['history'].apply(dates_array_to_timeseries)
    
    objs_df['history_ts'] = sequences

    objs_df.drop('access_dates', axis='columns', inplace=True)
    objs_df.drop('has_history', axis='columns', inplace=True)
    objs_df.drop('history', axis='columns', inplace=True)

    return objs_df
