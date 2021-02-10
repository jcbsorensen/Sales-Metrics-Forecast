import six
import sys
import joblib
sys.modules['sklearn.externals.six'] = six
sys.modules['sklearn.externals.joblib'] = joblib
import numpy as np, pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.pipeline import Pipeline
import pmdarima as pm

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


clean_df = FeatureEngineering.date_feature(clean_df)


teams = clean_df.loc[(clean_df['Team Name'] == 'Online') & (clean_df['Product Subcategory'] == 'Voice GA HW')].copy()
teams.sort_values('Salesdate', inplace=True)
#teams.set_index('Salesdate_num', inplace=True)




#fig, ax = plt.subplots(figsize=(50, 80))
result = seasonal_decompose(teams['Sales'].iloc[:60], model='additive')
result.seasonal.plot(figsize=(80,30))
plt.show()


fig, ax = plt.subplots(figsize=(30, 10))
plot_acf(teams['Sales'].iloc[:].values, ax=ax)
plt.show()







fig, ax = plt.subplots(figsize=(30, 20))
plot_acf(teams['Sales'], ax=ax)
plt.show()







# Seasonal - fit stepwise auto-ARIMA
smodel = pm.auto_arima(teams['Sales'].iloc[:-24], start_p=1, start_q=1,
                         test='adf',
                         max_p=14, max_q=14,
                         max_P=5, max_Q=5, m=14,
                         start_P=0, seasonal=True,
                         d=None, D=None, trace=True,
                         error_action='ignore',
                         suppress_warnings=True,
                         stepwise=True)

print(smodel.summary())

# Forecast
n_periods = 7
Prediction = pd.DataFrame(smodel.predict(n_periods=n_periods), index=teams['Sales'].iloc[-n_periods:].index)
Prediction.columns = ['predicted_sales']

# Plot
plt.figure(figsize=(20,15))
plt.plot(teams['Sales'].iloc[:-n_periods], label='Training')
plt.plot(teams['Sales'].iloc[-n_periods:], label='Test')
plt.plot(Prediction, label='Predicted')
plt.legend()

plt.title("Final Forecast")
plt.show()

from sklearn.metrics import mean_squared_error
teams['predicted'] = Prediction
mean_squared_error(teams['Sales'].iloc[-n_periods:], teams['predicted'].iloc[-n_periods:], squared=False)