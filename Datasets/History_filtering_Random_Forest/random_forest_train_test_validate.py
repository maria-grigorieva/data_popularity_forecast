import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score

from transform_filter_data import transform_filter_data
from visualize_heatmap_accesses import plot_heatmap_accesses


def train_test_validate(
    history_len = pd.Timedelta('365 days'),
    span_threshold = pd.Timedelta('90 days'),
    accesses_threshold = 30,
    source_csv = (
        '/home/mshubin/Desktop/datasets-popularity-task-data/' +
        'datasets_popularity_DAOD_HIGGD_mc16_13TeV_202309211655.csv'
    ),
    finish_date_train_test = pd.Timestamp('2021-10-27'),
    finish_date_validate = pd.Timestamp('2022-11-24'),
    horizon = pd.Timedelta('7 days'),
    Ntest = 1000,
):

    result = {}

    objs_df = transform_filter_data(
        source_csv = source_csv,
        finish_date = finish_date_train_test,
        horizon_date = finish_date_train_test + horizon,
        span_threshold = span_threshold,
        history_len = history_len,
        accesses_threshold = accesses_threshold,
        merge_tids=True
    )
    
    objs_df = objs_df.sample(frac=1, random_state=42)
    
    X = np.stack(objs_df['history_ts'].to_numpy())
    
    y = (objs_df['y'].to_numpy() > 0).astype('int')
    
    Xtrain = X[:-Ntest]
    Xtest = X[-Ntest:]
    Ytrain = y[:-Ntest]
    Ytest = y[-Ntest:]
    
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(Xtrain, Ytrain)
    
    Ypred = clf.predict(Xtest)

    result['test_accuracy'] = accuracy_score(Ytest, Ypred)
    result['test_confusion_matrix'] = confusion_matrix(Ytest, Ypred)
    result['test_classification_report'] = classification_report(Ytest, Ypred)
    
    print(f"Accuracy: {result['test_accuracy']}")
    print('Confusion Matrix')
    print(result['test_confusion_matrix'])
    print(result['test_classification_report'])
    
    ### Try to validate on another time cutoff
    
    objs_df_val = transform_filter_data(
        source_csv = source_csv,
        finish_date = finish_date_validate,
        horizon_date = finish_date_validate + horizon,
        span_threshold = span_threshold,
        history_len = history_len,
        accesses_threshold = accesses_threshold,
        merge_tids=True
    )
    
    Xval = np.stack(objs_df_val['history_ts'].to_numpy())
    yval = (objs_df_val['y'].to_numpy() > 0).astype('int')
    
    ypred_val = clf.predict(Xval)
    
    n_positive_answers = np.sum(ypred_val)

    result['val_accuracy'] = accuracy_score(yval, ypred_val)
    result['val_confusion_matrix'] = confusion_matrix(yval, ypred_val)
    if n_positive_answers:
        result['val_classification_report'] = classification_report(
            yval, ypred_val
        )
    else:
        result['val_classification_report'] = None
    
    print(f"Accuracy: {result['val_accuracy']}")
    print('Confusion Matrix')
    print(result['val_confusion_matrix'])
    print(result['val_classification_report'])
    
    plot_heatmap_accesses(
            np.stack(objs_df[objs_df['y'] > 0]['history_ts'].to_numpy()),
            axis=None,
            title=f'history {history_len.days} days, train positive dss',
            savefig_path=None
    )
    
    plot_heatmap_accesses(
            np.stack(
                objs_df_val[objs_df_val['y'] > 0]['history_ts'].to_numpy()
            ),
            axis=None,
            title=f'history {history_len.days} days, validation positive dss',
            savefig_path=None
    )

    return result
