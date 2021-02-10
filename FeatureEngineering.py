import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def lag_feature(df, lags, col):
    print('Creating lag features....')
    tmp = df[['Salesdate','Team Name','Product Category', 'Product Subcategory', col]]
    for i in lags:
        shifted = tmp.copy()
        shifted.columns = ['Salesdate','Team Name','Product Category', 'Product Subcategory', col+'_lag_'+str(i)]
        shifted['Salesdate'] = shifted['Salesdate'] + pd.Timedelta(i, unit='D')
        df = pd.merge(df, shifted, on=['Salesdate','Team Name','Product Category', 'Product Subcategory'], how='left')
    print('Creating lag features - Completed')
    return df

def rolling_feature(df, lags, col):
    print('Creating rolling features....')
    tmp = df[['Salesdate','Team Name', 'Product Subcategory', col]]
    for i in lags:
        print('Roll Range: ' + str(i))
        min_period = int(round(i/2,0))
        shifted = tmp.copy()
        shifted[str(i)+'D_roll_mean_'+col] = shifted.sort_values('Salesdate').groupby(['Team Name', 'Product Subcategory'])[col].apply(lambda x: x.rolling(i, min_periods=min_period).mean())
        shifted[str(i)+'D_roll_max_'+col] = shifted.sort_values('Salesdate').groupby(['Team Name', 'Product Subcategory'])[col].apply(lambda x: x.rolling(i, min_periods=min_period).max())
        shifted[str(i)+'D_roll_min_'+col] = shifted.sort_values('Salesdate').groupby(['Team Name', 'Product Subcategory'])[col].apply(lambda x: x.rolling(i, min_periods=min_period).min())
        shifted[str(i)+'D_roll_std_'+col] = shifted.sort_values('Salesdate').groupby(['Team Name', 'Product Subcategory'])[col].apply(lambda x: x.rolling(i, min_periods=min_period).std())
        del shifted[col]
        shifted['Salesdate'] = shifted['Salesdate'] + pd.Timedelta(1, unit='D')
        df = pd.merge(df, shifted, on=['Salesdate','Team Name','Product Subcategory'], how='left')
    print('Creating rolling features - Completed')
    return df

def date_feature(df, datenum=True):
    tmp = df.copy()
    min_date = tmp['Salesdate'].min()
    tmp['month'] = tmp['Salesdate'].dt.month
    tmp['week'] = tmp['Salesdate'].dt.week
    tmp['day'] = tmp['Salesdate'].dt.day
    tmp['weekday'] = tmp['Salesdate'].dt.dayofweek
    if datenum:
        tmp['Salesdate_num'] = (tmp['Salesdate'] - min_date).dt.days
    print('Creating date features - Completed')
    return tmp

def weekDay_feature(df, lags, col):
    print('Creating weekday features....')
    tmp = df[['Salesdate','Team Name', 'Product Subcategory', 'weekday', col]]
    for i in lags:
        min_period = int(round(i/2,0))
        shifted = tmp.copy()
        shifted[str(i)+'_wk_mean_'+col] = shifted.sort_values('Salesdate').groupby(['Team Name', 'Product Subcategory', 'weekday'])[col].apply(lambda x: x.rolling(i, min_periods=min_period).mean())
        shifted[str(i)+'_wk_max_'+col] = shifted.sort_values('Salesdate').groupby(['Team Name', 'Product Subcategory', 'weekday'])[col].apply(lambda x: x.rolling(i, min_periods=min_period).max())
        shifted[str(i)+'_wk_min_'+col] = shifted.sort_values('Salesdate').groupby(['Team Name', 'Product Subcategory', 'weekday'])[col].apply(lambda x: x.rolling(i, min_periods=min_period).min())
        shifted[str(i)+'_wk_std_'+col] = shifted.sort_values('Salesdate').groupby(['Team Name', 'Product Subcategory', 'weekday'])[col].apply(lambda x: x.rolling(i, min_periods=min_period).std())
        del shifted[col]
        shifted['Salesdate'] = shifted['Salesdate'] + pd.Timedelta(7, unit='D')
        df = pd.merge(df, shifted, on=['Salesdate','Team Name','Product Subcategory', 'weekday'], how='left')
    print('Creating weekday features - Completed')
    return df

def clean_data_engineering(df, max_lag):
    print('Cleaning features....')
    tmp = df.copy()

    # Cut-off the dataframe by the highest lag value (removes many nulls)
    min_date = (tmp['Salesdate'].min() + pd.Timedelta(max_lag, unit='D'))
    tmp = tmp.loc[tmp['Salesdate'] > min_date]

    # Replace NA with 0 for all lags
    for col in tmp.columns:
        if ('_lag_' in col) & (tmp[col].isnull().any()):
            if ('Sales' in col):
                tmp[col].fillna(0, inplace=True)
        if ('_roll_' in col) & (tmp[col].isnull().any()):
            if ('Sales' in col):
                tmp[col].fillna(0, inplace=True)
        if ('_wk_' in col) & (tmp[col].isnull().any()):
            if ('Sales' in col):
                tmp[col].fillna(0, inplace=True)

    print('Cleaning features - Completed')
    return tmp

def label_encoding(df):
    print('Encoding categorical features....')
    tmp = df.copy()

    LE_TeamName = LabelEncoder()
    LE_Category = LabelEncoder()
    LE_Subcategory = LabelEncoder()

    LE_TeamName.fit(tmp['Team Name'])
    tmp['Team Name'] = LE_TeamName.transform(tmp['Team Name'])

    LE_Category.fit(tmp['Product Category'])
    tmp['Product Category'] = LE_Category.transform(tmp['Product Category'])

    LE_Subcategory.fit(tmp['Product Subcategory'])
    tmp['Product Subcategory'] = LE_Subcategory.transform(tmp['Product Subcategory'])

    print('Encoding categorical features - Completed')
    return tmp, LE_TeamName, LE_Category, LE_Subcategory


def one_step_lag_feature(train, test, lags, col, time_feature, cutoff_date):
    print('Creating one-step lag features....')
    tmp = train.loc[train[time_feature] > cutoff_date,[time_feature,'Team Name','Product Subcategory', col]]
    for i in lags:
        shifted = tmp.copy()
        shifted.columns = [time_feature,'Team Name','Product Subcategory', col+'_lag_'+str(i)]
        shifted[time_feature] = shifted[time_feature] + pd.Timedelta(i, unit='D')
        test = pd.merge(test, shifted, on=[time_feature,'Team Name','Product Subcategory'], how='left')
    print('Creating one-step lag features - Completed')
    return test

def one_step_rolling_feature(train, test, lags, col, time_feature, cutoff_date):
    print('Creating one-step rolling features....')
    tmp = train.loc[train[time_feature] > cutoff_date, [time_feature, 'Team Name', 'Product Subcategory', col]]
    # check duplicates in index
    #print(tmp[tmp.index.duplicated()].head())
    for i in lags:
        min_period = int(round(i/2,0))
        shifted = tmp.copy()
        shifted[str(i)+'D_roll_mean_'+col] = shifted.sort_values(time_feature).groupby(['Team Name', 'Product Subcategory'])[col].apply(lambda x: x.rolling(i, min_periods=min_period).mean())
        shifted[str(i)+'D_roll_max_'+col] = shifted.sort_values(time_feature).groupby(['Team Name', 'Product Subcategory'])[col].apply(lambda x: x.rolling(i, min_periods=min_period).max())
        shifted[str(i)+'D_roll_min_'+col] = shifted.sort_values(time_feature).groupby(['Team Name', 'Product Subcategory'])[col].apply(lambda x: x.rolling(i, min_periods=min_period).min())
        shifted[str(i)+'D_roll_std_'+col] = shifted.sort_values(time_feature).groupby(['Team Name', 'Product Subcategory'])[col].apply(lambda x: x.rolling(i, min_periods=min_period).std())
        del shifted[col]
        shifted[time_feature] = shifted[time_feature] + pd.Timedelta(1, unit='D')
        test = pd.merge(test, shifted, on=[time_feature,'Team Name','Product Subcategory'], how='left')
    print('Creating one-step rolling features - Completed')
    return test

def one_step_weekDay_feature(train, test, lags, col, time_feature, cutoff_date):
    print('Creating weekday features....')
    tmp = train.loc[train[time_feature] > cutoff_date, [time_feature, 'Team Name', 'Product Subcategory', 'weekday', col]]
    for i in lags:
        min_period = int(round(i/2,0))
        shifted = tmp.copy()
        shifted[str(i)+'_wk_mean_'+col] = shifted.sort_values(time_feature).groupby(['Team Name', 'Product Subcategory', 'weekday'])[col].apply(lambda x: x.rolling(i, min_periods=min_period).mean())
        shifted[str(i)+'_wk_max_'+col] = shifted.sort_values(time_feature).groupby(['Team Name', 'Product Subcategory', 'weekday'])[col].apply(lambda x: x.rolling(i, min_periods=min_period).max())
        shifted[str(i)+'_wk_min_'+col] = shifted.sort_values(time_feature).groupby(['Team Name', 'Product Subcategory', 'weekday'])[col].apply(lambda x: x.rolling(i, min_periods=min_period).min())
        shifted[str(i)+'_wk_std_'+col] = shifted.sort_values(time_feature).groupby(['Team Name', 'Product Subcategory', 'weekday'])[col].apply(lambda x: x.rolling(i, min_periods=min_period).std())
        del shifted[col]
        shifted['Salesdate'] = shifted['Salesdate'] + pd.Timedelta(7, unit='D')
        test = pd.merge(test, shifted, on=[time_feature,'Team Name','Product Subcategory', 'weekday'], how='left')
    print('Creating weekday features - Completed')
    return test
