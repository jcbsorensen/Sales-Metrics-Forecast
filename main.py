import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn import preprocessing

# customo
import DataPrep
import PipelineTools as pt
import FeatureEngineering
import forecast
import Validations

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

# remove outliers for online
online_NMvoice_median = clean_df.loc[(clean_df['Team Name'] == 'Online') & (clean_df['Product Subcategory'] == 'NM Voice'), 'Sales'].median()
online_acces_median = clean_df.loc[(clean_df['Team Name'] == 'Online') & (clean_df['Product Subcategory'] == 'Accessories'), 'Sales'].median()
clean_df.loc[(clean_df['Sales']> 200) & (clean_df['Team Name'] == 'Online') & (clean_df['Product Subcategory'] == 'NM Voice'), 'Sales'] = online_NMvoice_median
clean_df.loc[(clean_df['Sales']> 200) & (clean_df['Team Name'] == 'Online') & (clean_df['Product Subcategory'] == 'Accessories'), 'Sales'] = online_acces_median
del online_NMvoice_median
del online_acces_median

clean_df, LE_TeamName, LE_Category, LE_Subcategory = FeatureEngineering.label_encoding(clean_df)


pipeline_val_steps = [
    ('lag features', pt.lag_feature(lags=[1, 6, 7, 14, 21, 28], col='Sales')),
    ('roll mean', pt.rolling_mean(lags=[7,8, 14,15,21,29], col='Sales')),
    ('roll ewm', pt.exponential_moving_average(lags=[4,6,8,9,29], col='Sales')),
    ('roll max', pt.rolling_max(lags=[4,7,8,14], col='Sales')),
    ('roll min', pt.rolling_min(lags=[4,6,10], col='Sales')),
    ('roll std', pt.rolling_std(lags=[4,15,22,29], col='Sales')),
    ('date features', pt.date_features()),
    ('roll wk mean', pt.rolling_weekday_mean(rolls=[2,3,4], col='Sales')),
    ('roll wk max', pt.rolling_weekday_max(rolls=[2,3,4], col='Sales')),
    ('roll wk min', pt.rolling_weekday_min(rolls=[2,4], col='Sales')),
    ('roll wk std', pt.rolling_weekday_std(rolls=[2,3,4], col='Sales')),
    ('final step', pt.final(['month', 'week', 'day', 'Product Category']))
]

pipeline_val = Pipeline(steps=pipeline_val_steps)
clean_df = pipeline_val.fit_transform(clean_df)

# Xgboost
#MAE, RMSE, predict_df = Validations.multi_step_validation(clean_df, 7, 'Salesdate', 'Sales')
MAE, RMSE, predict_df = forecast.multi_step_forecast(clean_df, 7, 'Salesdate', 'Sales')

predict_df['Team Name'] = LE_TeamName.inverse_transform(predict_df['Team Name'])
predict_df['Product Subcategory'] = LE_Subcategory.inverse_transform(predict_df['Product Subcategory'])
clean_df['Team Name'] = LE_TeamName.inverse_transform(clean_df['Team Name'])
clean_df['Product Subcategory'] = LE_Subcategory.inverse_transform(clean_df['Product Subcategory'])


backup = predict_df.copy()

complete = pd.merge(clean_df, predict_df[['Salesdate', 'Team Name', 'Product Subcategory', 'predict']], on=['Salesdate', 'Team Name', 'Product Subcategory'], how='left')
#

products = complete['Product Subcategory'].unique()

for product in products:
    complete.loc[(complete['Team Name'] == 'Online') & (complete['Product Subcategory'] == product) & (
                complete['Salesdate'] >= '2020-10-01')].plot(x='Salesdate', y=['Sales', 'predict'], title=product)
    plt.show()


teamNames = complete['Team Name'].unique()

for team in teamNames[:5]:
    aggr = complete.loc[(complete['Team Name'] == team)]
    aggr = aggr.groupby(['Salesdate']).sum()
    aggr['Salesdate'] = aggr.index
    aggr.loc[(aggr['Salesdate'] >= '2020-10-01')].plot(x='Salesdate', y=['Sales', 'predict'], title=team)
    plt.show()








