import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import cross_val_score, cross_validate
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

traffic_data = pd.read_csv('data\Traffic.csv', sep=";",
                           index_col='Date',
                           parse_dates=True,
                           dayfirst=True,
                           usecols=['Date', 'Team Name', 'Visitor Amount'])
traffic_data.columns = ['Traffic', 'Team Name']
traffic_data['Salesdate'] = traffic_data.index


clean_df = pd.merge(clean_df, traffic_data, on=['Salesdate', 'Team Name'], how='left')
clean_df['Traffic'].fillna(0, inplace=True)
del (traffic_data)



# Cross Validation
clean_df.reset_index(drop=True, inplace=True)
clean_df['Sales_X'] = clean_df.loc[:,'Sales']
X_train = clean_df.drop(['Sales'], axis=1)
Y_train = clean_df['Sales']
Y_valid = clean_df['Sales']

# Set indices for CV
custom_cv = []
steps = 7
for step in range(steps):
    max_date = clean_df['Salesdate'].max()
    start_test_date = max_date - pd.Timedelta(steps, unit='D')
    step_no = step + 1
    current_test_date = start_test_date + pd.Timedelta(step_no, unit='D')
    train_index = clean_df[clean_df['Salesdate'] < current_test_date].index.values.astype(int)
    validation_index = clean_df[clean_df['Salesdate'] == '2020-10-31'].index.values.astype(int)
    custom_cv.append((train_index, validation_index))


X_train['TS_ID'] = X_train['Team Name'] + X_train['Product Subcategory']

pipeline_val_steps = [
    ('lag features', pt.lag_feature(lags=[1, 6, 7, 14, 21, 28], col='Sales_X')),
    #('lag traffic features', pt.lag_feature(lags=[1, 6, 7, 14, 21, 28], col='Traffic')),
    ('roll mean', pt.rolling_mean(lags=[7,8, 14,15,21,29], col='Sales_X')),
    ('roll ewm', pt.exponential_moving_average(lags=[4,6,8,9,29], col='Sales_X')),
    #('roll traffic ewm', pt.exponential_moving_average(lags=[3,6], col='Traffic')),
    ('roll max', pt.rolling_max(lags=[4,7,8,14], col='Sales_X')),
    ('roll min', pt.rolling_min(lags=[4,6,10], col='Sales_X')),
    ('roll std', pt.rolling_std(lags=[4,15,22,29], col='Sales_X')),
    ('date features', pt.date_features()),
    ('roll wk mean', pt.rolling_weekday_mean(rolls=[2,3,4], col='Sales_X')),
    ('roll wk max', pt.rolling_weekday_max(rolls=[2,3,4], col='Sales_X')),
    ('roll wk min', pt.rolling_weekday_min(rolls=[2,4], col='Sales_X')),
    ('roll wk std', pt.rolling_weekday_std(rolls=[2,3,4], col='Sales_X')),
    ('final step', pt.final(['Sales_X', 'Salesdate', 'Product Subcategory', 'Team Name', 'Product Category', 'month', 'week', 'day', 'Traffic']))
]

# pipeline_train_steps = [
#     ('lag features', pt.lag_feature(lags=[1, 6, 7, 14, 21, 28], col='Sales_X')),
#     ('date features', pt.date_features()),
#     ('final step', pt.final(['Sales_X', 'Salesdate'])),
#     ('xgb', XGBRegressor(max_depth=4,
#     n_estimators=120,
#     min_child_weight=300,
#     colsample_bytree=0.8,
#     subsample=0.9,
#     eta=0.3,
#     seed=2017))
# ]

pipeline_val = Pipeline(steps=pipeline_val_steps)
X_valid = pipeline_val.fit_transform(X_train)
#pipeline_train = Pipeline(steps=pipeline_train_steps)
#pipeline.fit(X_train, Y_train, xgb__eval_metric='rmse', xgb__eval_set=[(X_train, Y_train)], xgb__verbose=False, xgb__early_stopping_rounds=10)
#X_train = pipeline_train.fit_transform(X_train)
#test2 = test.loc[(test['Team Name'] == 0) & (test['Product Subcategory'] == 0)].sort_values(['Salesdate', 'Team Name', 'Product Subcategory'])

TS_Features = pd.read_csv('data\TSFresh Features.csv',
                       #sep=";",
                       #decimal=',',
                       #index_col='Date',
                       #parse_dates=True,
                       #dayfirst=True,
                       usecols=['Sales__abs_energy', 'Sales__mean_abs_change',
                                'Sales__mean_change', 'Sales__mean_second_derivative_central', 'Sales__median', 'TS_ID'])
#TS_Features.columns[30:60]
# X_valid = pd.merge(X_valid, TS_Features, on='TS_ID', how='left')
X_valid.drop(['TS_ID'], axis=1, inplace=True)


xgb_params = {
    'max_depth':8,
    'n_estimators':1000,
    'min_child_weight':300,
    'colsample_bytree':0.8,
    'subsample':1.0,
    'eta':0.0001,
    'seed':2017
}

xgb_eval_params = {
    'eval_metric':"rmse",
    'eval_set':[(X_valid, Y_valid)],
    'verbose':False,
    'early_stopping_rounds':10
}

model = XGBRegressor(max_depth=8,
    n_estimators=120,
    min_child_weight=300,
    colsample_bytree=0.8,
    subsample=1.0,
    eta=0.0001,
    seed=2017)

#scores = cross_validate(pipeline_train, X_train, Y_train, cv=custom_cv, scoring='neg_mean_squared_error', fit_params=xgb_eval_params)
scores = cross_validate(model, X_valid, Y_valid, cv=custom_cv, scoring='neg_mean_squared_error', fit_params=xgb_eval_params, n_jobs=-1)
print(np.mean(np.sqrt(np.abs(scores['test_score']))))





