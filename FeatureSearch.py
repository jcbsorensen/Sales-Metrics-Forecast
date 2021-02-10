import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
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
clean_df['Sales_X'] = clean_df.loc[:, 'Sales']
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
    validation_index = clean_df[clean_df['Salesdate'] == '2020-10-31'].index.values.astype(int)
    custom_cv.append((train_index, validation_index))

xgb_params = {
    'max_depth': 4,
    'n_estimators': 120,
    'min_child_weight': 300,
    'colsample_bytree': 0.8,
    'subsample': 0.9,
    'eta': 0.3,
    'seed': 2017
}

X_train['TS_ID'] = X_train['Team Name'] + X_train['Product Subcategory']

pipeline_val_steps = [
    ('lag features', pt.lag_feature(lags=[1, 6, 7, 14, 21, 28], col='Sales_X')),
    # ('roll mean', pt.rolling_mean(lags=[7,8, 14,15,21,29], col='Sales_X')),
    # ('roll ewm', pt.exponential_moving_average(lags=[4,6,8,9,29], col='Sales_X')),
    # ('roll max', pt.rolling_max(lags=[4,7,8,14], col='Sales_X')),
    # ('roll min', pt.rolling_min(lags=[4,6,10], col='Sales_X')),
    # ('roll std', pt.rolling_std(lags=[4,15,22,29], col='Sales_X')),
    ('date features', pt.date_features()),
    # ('roll wk mean', pt.rolling_weekday_mean(rolls=[2,3,4], col='Sales_X')),
    # ('roll wk max', pt.rolling_weekday_max(rolls=[2,3,4], col='Sales_X')),
    # ('roll wk min', pt.rolling_weekday_min(rolls=[2,4], col='Sales_X')),
    # ('roll wk std', pt.rolling_weekday_std(rolls=[2,3,4], col='Sales_X')),
    ('final step', pt.final(['Sales_X', 'Salesdate', 'Product Subcategory', 'Team Name', 'Product Category', 'month', 'week', 'day']))
]

pipeline_val = Pipeline(steps=pipeline_val_steps)
X_train = pipeline_val.fit_transform(X_train)

TS_Features = pd.read_csv('data\TSFresh Features.csv',
                          # sep=";",
                          # decimal=',',
                          # index_col='Date',
                          # parse_dates=True,
                          # dayfirst=True,
                          usecols=['Sales__sum_values',
                                   'Sales__abs_energy', 'Sales__mean_abs_change', 'Sales__mean_change',
                                   'Sales__mean_second_derivative_central', 'Sales__median',
                                   'Sales__mean',
                                   # 'Sales__standard_deviation', 'Sales__variance', 'Sales__skewness',
                                   # 'Sales__kurtosis', 'Sales__absolute_sum_of_changes',
                                   # 'Sales__last_location_of_minimum', 'Sales__sum_of_reoccurring_values',
                                   # 'Sales__sum_of_reoccurring_data_points',
                                   'TS_ID'])

# X_train = pd.merge(X_train, TS_Features, on='TS_ID', how='left')
X_train.drop(['TS_ID'], axis=1, inplace=True)
X_train.fillna(0, inplace=True)

model = XGBRegressor(
    max_depth=8,
    n_estimators=120,
    min_child_weight=300,
    colsample_bytree=0.8,
    subsample=1.0,
    eta=0.0001,
    seed=2017)


from sklearn.feature_selection import RFECV

selector = RFECV(model, step=1, min_features_to_select=1, cv=custom_cv, scoring='neg_root_mean_squared_error', n_jobs=-1)
# selector = selector.set_params(**fit_params)
selector = selector.fit(X_train, Y_train)

plt.figure()
plt.title('XGB CV score vs No of Features')
plt.xlabel("Number of features selected")
plt.ylabel("Neg Mean Squared Error")
plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
plt.show()

# get rank of X model features
rank = selector.ranking_

# Subset features to those selected by recursive feature elimination
fisk = selector.support_
hest = X_train.iloc[:, fisk]

hest.columns

# ['Sales__sum_values',
#        'Sales__abs_energy', 'Sales__mean_abs_change', 'Sales__mean_change',
#        'Sales__mean_second_derivative_central', 'Sales__median']

#
# ['Sales__mean',
#        'Sales__standard_deviation', 'Sales__variance', 'Sales__skewness',
#        'Sales__kurtosis', 'Sales__absolute_sum_of_changes',
#        'Sales__last_location_of_minimum', 'Sales__sum_of_reoccurring_values',
#        'Sales__sum_of_reoccurring_data_points']
