import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from tsfresh import extract_features
from tsfresh.utilities.distribution import LocalDaskDistributor

# customo
import DataPrep
import PipelineTools as pt

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

clean_df['TS_ID'] = clean_df['Team Name'] + clean_df['Product Subcategory']
TS_Ready = clean_df.drop(['Team Name', 'Product Category', 'Product Subcategory'], axis=1)

Distributor = LocalDaskDistributor(n_workers=3)

extracted_features = extract_features(timeseries_container=TS_Ready,
                                      column_id='TS_ID',
                                      column_sort="Salesdate",
                                      distributor=Distributor)

extracted_features['TS_ID'] = extracted_features.index
extracted_features.to_csv('data\TSFresh Features.csv', index=False)

extracted_features.columns


