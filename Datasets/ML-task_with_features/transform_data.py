import pandas as pd
from tqdm import tqdm

def transform_data(
        timestamp_cutoff = pd.to_datetime('2023-01-01'),
        timestamp_horizon = pd.to_datetime('2023-04-01'),

        source_csv =
        'datasets_popularity_DAOD_PHYS_mc16_13TeV_202308141254.csv'
):
    """
    Prepare dataset for ML task. Data samples are datasets with features:

    'n_users' --- number of users, that accessed the dataset (int),
    'n_accesses' --- number of accesses (int),
    'n_access_days' --- number of days, during which accesses took place (int),
    'last_date' --- the last timestamp, at which the last access took place
                    (pd.Timestamp),

    all these features are counted for the time interval from the beginning
    up to 'timestamp_cutoff' time point inclusive.

    The 'y' (target) variable of each sample is a number of accesses within
    the time interval ('timestamp_cutoff', 'timestamp_horizon']. Type: int.

    The input data is taken from 'source_csv', which is a table, where each row
    is an access to a specific dataset, necessary columns are:
    'datetime', 'datasetname', 'username'.

    Returns
    -------

    pd.DataFrame with samples for ML task

    """

    df = pd.read_csv(source_csv)
    df['datetime'] = pd.to_datetime(df['datetime'])

    df_before = df[df['datetime'] <= timestamp_cutoff]
    
    df_after = df[(df['datetime'] > timestamp_cutoff) &
                  (df['datetime'] <= timestamp_horizon)]
    
    df_gb_before = df_before.groupby(['datasetname'])
    df_gb_after = df_after.groupby(['datasetname'])
    
    n_users = df_gb_before['username'].nunique()
    n_accesses = df_gb_before['datetime'].count()
    n_access_days = df_gb_before['datetime'].nunique()
    last_date = df_gb_before['datetime'].max()
    
    objs_df = pd.DataFrame(
        {
            'n_users' : n_users,
            'n_accesses' : n_accesses,
            'n_access_days' : n_access_days,
            'last_date' : last_date
        }
    )
    
    future_popularity = df_gb_after['datetime'].count()
    
    objs_df['y'] = 0
    
    for ds_name, ds_features in tqdm(objs_df.iterrows(), total=len(objs_df)):
        if ds_name in future_popularity:
            objs_df.loc[ds_name, 'y'] = future_popularity[ds_name]

    return objs_df
