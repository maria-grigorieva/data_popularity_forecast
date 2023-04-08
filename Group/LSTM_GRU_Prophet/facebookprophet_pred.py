import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
from acquire_timeseries import acquire_timeseries

from prophet import Prophet


def smape(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))


if __name__ == "__main__":
    random.seed(1234)
    np.random.seed(123)

    ds_group_ts = acquire_timeseries(
        input_csv='group_samples/group_popularity_until_13.02.2023.csv',
        input_format='DAOD',  # AOD
        input_subformat='TOPQ',  # TOPQ HIGG SUSY PHYS
        input_project='mc16_13TeV'  # data16_13TeV
    )

    # ds_group_ts = ds_group_ts[ : -24]

    ds_group_ts_log = np.log(ds_group_ts + 1)

    NTest = 12

    train_ts = ds_group_ts_log[:-NTest]

    train_df = pd.DataFrame({'ds' : train_ts.index, 'y' : train_ts})
    m = Prophet(yearly_seasonality=True,
                changepoint_prior_scale=0.05)
    m.fit(train_df)

    all_df = pd.DataFrame({'ds' : ds_group_ts.index, 'y' : ds_group_ts_log})

    forecast = m.predict(all_df)

    real = ds_group_ts_log[-NTest:].to_numpy()
    pred = forecast['yhat'][-NTest:].to_numpy()
    smape_log = smape(real, pred)
    print(f'{smape_log=}')

    mape_log = mean_absolute_percentage_error(real, pred)
    print(f'{mape_log=}')

    # cut off for the clarity of the forthcoming plot
    ds_group_ts = ds_group_ts['2022-07-01':]
    ds_group_ts_log = ds_group_ts_log['2022-07-01':]
    forecast = forecast[forecast['ds'] >= '2022-07-01']

    fig, ax = plt.subplots()
    fig.set(figwidth=7, figheight=5)
    ax.set(xlabel='Date', ylabel='Log N_Tasks')
    ax.plot(ds_group_ts_log, 'k--', label='actual log(n_tasks + 1)')
    ax.plot(ds_group_ts.index[:-NTest], forecast['yhat'][:-NTest],
            'kx--', label='train predicted')
    ax.plot(ds_group_ts.index[-NTest:], forecast['yhat'][-NTest:],
            'ko-', label='test predicted')
    ax.legend()
    plt.show()
    # fig.savefig('../../prophet-topq.eps', bbox_inches='tight')
