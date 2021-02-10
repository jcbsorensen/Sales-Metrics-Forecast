import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
import scipy
from sklearn.preprocessing import PowerTransformer
import xgboost as xgb
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline


# customo
import DataPrep
import FeatureEngineering
import PipelineTools as pt
import Validations
import WalkForwardValidation as wfv

pd.set_option('display.max_row', 1000)
pd.set_option('display.max_columns', 50)

raw_data = pd.read_csv('data\Key Metrics_full.csv',
                       sep=";",
                       decimal=',',
                       index_col='Date',
                       parse_dates=True,
                       dayfirst=True,
                       usecols=['Date', 'Team Name', 'Product Category', 'Product Subcategory', 'Sales'])

# Clean & Prep Data
clean_df = DataPrep.DataPrepping(raw_data)
del (raw_data)


clean_df.reset_index(drop=True, inplace=True)
clean_df['Sales_X'] = clean_df.loc[:,'Sales']
X_train = clean_df.drop(['Sales'], axis=1)
Y_train = clean_df['Sales']

# Set indices for CV
custom_cv = []
steps = 7
for step in range(steps):
    max_date = clean_df['Salesdate'].max()
    start_test_date = max_date - pd.Timedelta(steps, unit='D')
    step_no = step + 1
    current_test_date = start_test_date + pd.Timedelta(step_no, unit='D')
    train_index = clean_df[clean_df['Salesdate'] < current_test_date].index.values.astype(int)
    validation_index = clean_df[clean_df['Salesdate'] == '2020-11-30'].index.values.astype(int)
    custom_cv.append((train_index, validation_index))


pipeline_val_steps = [
    ('lag features', pt.lag_feature(lags=[1, 6, 7, 14, 21, 28], col='Sales_X')),
    ('roll mean', pt.rolling_mean(lags=[7,8, 14,15,21,29], col='Sales_X')),
    ('roll ewm', pt.exponential_moving_average(lags=[4,6,8,9,29], col='Sales_X')),
    ('roll max', pt.rolling_max(lags=[4,7,8,14], col='Sales_X')),
    ('roll min', pt.rolling_min(lags=[4,6,10], col='Sales_X')),
    ('roll std', pt.rolling_std(lags=[4,15,22,29], col='Sales_X')),
    ('date features', pt.date_features()),
    ('roll wk mean', pt.rolling_weekday_mean(rolls=[2,3,4], col='Sales_X')),
    ('roll wk max', pt.rolling_weekday_max(rolls=[2,3,4], col='Sales_X')),
    ('roll wk min', pt.rolling_weekday_min(rolls=[2,4], col='Sales_X')),
    ('roll wk std', pt.rolling_weekday_std(rolls=[2,3,4], col='Sales_X')),
    ('final step', pt.final(['Sales_X', 'Salesdate', 'Product Subcategory', 'Team Name', 'Product Category', 'month', 'week', 'day']))
]


pipeline_val = Pipeline(steps=pipeline_val_steps)
X_train = pipeline_val.fit_transform(X_train)



xgb_params = {
    'max_depth':[2, 4, 6, 8],
    'colsample_bytree':[0.3, 0.6, 0.8, 1],
    'subsample':[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'eta':[0.0001, 0.001, 0.01, 0.1, 0.2, 0.3],
    'seed':[2017]
}

fit_params={"early_stopping_rounds":10,
            "eval_metric" : "rmse",
            "eval_set" : [[X_train, Y_train]]}

model = xgb.XGBRegressor()
gridsearch = GridSearchCV(model, xgb_params, verbose=1, scoring='neg_root_mean_squared_error', cv=custom_cv, n_jobs=-1)
gridsearch.fit(X_train,Y_train, **fit_params)

gridsearch.best_estimator_
gridsearch.best_params_
gridsearch.best_score_
#
# XGBRegressor(colsample_bytree=0.8, eta=0.0001, max_depth=8, seed=2017,
#              subsample=1.0)
# score 2.8086