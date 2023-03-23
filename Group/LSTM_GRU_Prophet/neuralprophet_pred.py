import pandas as pd
import numpy as np
import datetime
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
from acquire_timeseries import acquire_timeseries

from neuralprophet import NeuralProphet, set_random_seed, set_log_level


if __name__ == "__main__":
    set_log_level("ERROR")
    set_random_seed(0)

    ds_group_ts = acquire_timeseries(
        input_csv='group_samples/group_popularity_until_13.02.2023.csv',
        input_format='DAOD',  # AOD
        input_subformat='HIGG',  # TOPQ HIGG SUSY PHYS
        input_project='mc16_13TeV'  # data16_13TeV
    )

    ds_group_ts = np.log(ds_group_ts + 1)

    NTest = 12
    train_ts = ds_group_ts[:-NTest]

    train_df = pd.DataFrame({'ds' : train_ts.index, 'y' : train_ts})

    params = {'learning_rate': 0.2,
              'epochs': 200,
              'yearly_seasonality': True,
              'weekly_seasonality': False,
              'daily_seasonality': False,
              #'seasonality_mode': 'multiplicative',
              'n_changepoints': 10,
              'changepoints_range': 0.8,
              'n_lags': 16,
              'n_forecasts': 1}

    m = NeuralProphet(**params)
    m.fit(train_df, progress='bar')

    all_df = pd.DataFrame({'ds' : ds_group_ts.index, 'y' : ds_group_ts})

    forecast = m.predict(all_df)

    real = ds_group_ts[-NTest:].tolist()
    pred = forecast['yhat1'][-NTest:].tolist()
    mape = mean_absolute_percentage_error(real, pred)
    print("MAPE:", mape)

    real_orig = np.exp(real) - 1
    pred_orig = np.exp(pred) - 1
    mape = mean_absolute_percentage_error(real_orig, pred_orig)
    print("MAPE:", mape)

    fig, ax = plt.subplots()
    fig.set(figwidth=15, figheight=5)
    ax.set(xlabel='Date', ylabel='N_Tasks')
    ax.plot(ds_group_ts, '-', label='n_tasks')
    ax.plot(ds_group_ts.index[:-NTest], forecast['yhat1'][:-NTest],
            '-', label='train_predicted')
    ax.plot(ds_group_ts.index[-NTest:], forecast['yhat1'][-NTest:],
            '-', label='test_predicted')
    plt.legend()
    plt.show()

    # future_df = m.make_future_dataframe(train_df, periods=NTest)
    # m.plot(forecast)
    # ds_group_ts.plot(figsize=(15, 5))
