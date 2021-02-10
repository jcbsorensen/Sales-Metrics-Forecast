import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBRegressor
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder




class lag_feature(BaseEstimator, TransformerMixin):
    def __init__(self, lags=[1], col='Sales'):
        print('Lags Init Called')
        self.lags = lags
        self.col = col

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print('Lags transform called')
        tmp = X[['Salesdate', 'Team Name', 'Product Subcategory', self.col]]
        for i in self.lags:
            shifted = tmp.copy()
            shifted.columns = ['Salesdate', 'Team Name', 'Product Subcategory', self.col + '_lag_' + str(i)]
            #shifted.fillna(0, inplace=True)
            shifted['Salesdate'] = shifted['Salesdate'] + pd.Timedelta(i, unit='D')
            X = pd.merge(X, shifted, on=['Salesdate', 'Team Name', 'Product Subcategory'], how='left')
        print('Creating lag features - Completed')
        return X.fillna(0)

class rolling_mean(BaseEstimator, TransformerMixin):
    def __init__(self, lags=[1], col='Sales'):
        print('Rolling Mean Initialized')
        self.lags = lags
        self.col = col

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print('Rolling Mean transform called')
        tmp = X[['Salesdate','Team Name', 'Product Subcategory', self.col]]
        for i in self.lags:
            print('Roll Range: ' + str(i))
            min_period = int(round(i / 2, 0))
            shifted = tmp.copy()
            shifted[str(i)+'D_roll_mean_'+ self.col] = shifted.sort_values('Salesdate').groupby(['Team Name', 'Product Subcategory'])[self.col].apply(lambda x: x.rolling(i, min_periods=min_period).mean())
            del shifted[self.col]
            shifted['Salesdate'] = shifted['Salesdate'] + pd.Timedelta(1, unit='D')
            X = pd.merge(X, shifted, on=['Salesdate','Team Name','Product Subcategory'], how='left')
        print('Creating rolling mean features - Completed')
        return X.fillna(0)

class rolling_max(BaseEstimator, TransformerMixin):
    def __init__(self, lags=[1], col='Sales'):
        print('Rolling Max Initialized')
        self.lags = lags
        self.col = col

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print('Rolling Max transform called')
        tmp = X[['Salesdate','Team Name', 'Product Subcategory', self.col]]
        for i in self.lags:
            print('Roll Range: ' + str(i))
            min_period = int(round(i / 2, 0))
            shifted = tmp.copy()
            shifted[str(i)+'D_roll_Max_'+ self.col] = shifted.sort_values('Salesdate').groupby(['Team Name', 'Product Subcategory'])[self.col].apply(lambda x: x.rolling(i, min_periods=min_period).max())
            del shifted[self.col]
            shifted['Salesdate'] = shifted['Salesdate'] + pd.Timedelta(1, unit='D')
            X = pd.merge(X, shifted, on=['Salesdate','Team Name','Product Subcategory'], how='left')
        print('Creating rolling Max features - Completed')
        return X.fillna(0)

class rolling_min(BaseEstimator, TransformerMixin):
    def __init__(self, lags=[1], col='Sales'):
        print('Rolling min Initialized')
        self.lags = lags
        self.col = col

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print('Rolling min transform called')
        tmp = X[['Salesdate','Team Name', 'Product Subcategory', self.col]]
        for i in self.lags:
            print('Roll Range: ' + str(i))
            min_period = int(round(i / 2, 0))
            shifted = tmp.copy()
            shifted[str(i)+'D_roll_min_'+ self.col] = shifted.sort_values('Salesdate').groupby(['Team Name', 'Product Subcategory'])[self.col].apply(lambda x: x.rolling(i, min_periods=min_period).min())
            del shifted[self.col]
            shifted['Salesdate'] = shifted['Salesdate'] + pd.Timedelta(1, unit='D')
            X = pd.merge(X, shifted, on=['Salesdate','Team Name','Product Subcategory'], how='left')
        print('Creating rolling min features - Completed')
        return X.fillna(0)

class rolling_std(BaseEstimator, TransformerMixin):
    def __init__(self, lags=[1], col='Sales'):
        print('Rolling std Initialized')
        self.lags = lags
        self.col = col

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print('Rolling std transform called')
        tmp = X[['Salesdate','Team Name', 'Product Subcategory', self.col]]
        for i in self.lags:
            print('Roll Range: ' + str(i))
            min_period = int(round(i / 2, 0))
            shifted = tmp.copy()
            shifted[str(i)+'D_roll_std_'+ self.col] = shifted.sort_values('Salesdate').groupby(['Team Name', 'Product Subcategory'])[self.col].apply(lambda x: x.rolling(i, min_periods=min_period).std())
            del shifted[self.col]
            shifted['Salesdate'] = shifted['Salesdate'] + pd.Timedelta(1, unit='D')
            X = pd.merge(X, shifted, on=['Salesdate','Team Name','Product Subcategory'], how='left')
        print('Creating rolling std features - Completed')
        return X.fillna(0)

class rolling_weekday_mean(BaseEstimator, TransformerMixin):
    def __init__(self, rolls=[1], col='Sales'):
        print('Rolling Weekday Initialized')
        self.rolls = rolls
        self.col = col

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print('Rolling Weekday transform called')
        tmp = X[['Salesdate','Team Name', 'Product Subcategory', 'weekday', self.col]]
        for i in self.rolls:
            print('Roll Weekday Range: ' + str(i))
            min_period = int(round(i / 2, 0))
            shifted = tmp.copy()
            shifted[str(i) + '_wk_mean_' + self.col] = shifted.sort_values('Salesdate').groupby(['Team Name', 'Product Subcategory', 'weekday'])[self.col].apply(lambda x: x.rolling(i, min_periods=min_period).mean())
            del shifted[self.col]
            shifted['Salesdate'] = shifted['Salesdate'] + pd.Timedelta(7, unit='D')
            X = pd.merge(X, shifted, on=['Salesdate','Team Name','Product Subcategory', 'weekday'], how='left')
        print('Creating rolling Weekday features - Completed')
        return X.fillna(0)

class rolling_weekday_max(BaseEstimator, TransformerMixin):
    def __init__(self, rolls=[1], col='Sales'):
        print('Rolling Weekday Initialized')
        self.rolls = rolls
        self.col = col

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print('Rolling Weekday transform called')
        tmp = X[['Salesdate','Team Name', 'Product Subcategory', 'weekday', self.col]]
        for i in self.rolls:
            print('Roll Weekday Range: ' + str(i))
            min_period = int(round(i / 2, 0))
            shifted = tmp.copy()
            shifted[str(i) + '_wk_max_' + self.col] = shifted.sort_values('Salesdate').groupby(['Team Name', 'Product Subcategory', 'weekday'])[self.col].apply(lambda x: x.rolling(i, min_periods=min_period).max())
            del shifted[self.col]
            shifted['Salesdate'] = shifted['Salesdate'] + pd.Timedelta(7, unit='D')
            X = pd.merge(X, shifted, on=['Salesdate','Team Name','Product Subcategory', 'weekday'], how='left')
        print('Creating rolling Weekday features - Completed')
        return X.fillna(0)

class rolling_weekday_min(BaseEstimator, TransformerMixin):
    def __init__(self, rolls=[1], col='Sales'):
        print('Rolling Weekday Initialized')
        self.rolls = rolls
        self.col = col

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print('Rolling Weekday transform called')
        tmp = X[['Salesdate','Team Name', 'Product Subcategory', 'weekday', self.col]]
        for i in self.rolls:
            print('Roll Weekday Range: ' + str(i))
            min_period = int(round(i / 2, 0))
            shifted = tmp.copy()
            shifted[str(i) + '_wk_min_' + self.col] = shifted.sort_values('Salesdate').groupby(['Team Name', 'Product Subcategory', 'weekday'])[self.col].apply(lambda x: x.rolling(i, min_periods=min_period).min())
            del shifted[self.col]
            shifted['Salesdate'] = shifted['Salesdate'] + pd.Timedelta(7, unit='D')
            X = pd.merge(X, shifted, on=['Salesdate','Team Name','Product Subcategory', 'weekday'], how='left')
        print('Creating rolling Weekday features - Completed')
        return X.fillna(0)

class rolling_weekday_std(BaseEstimator, TransformerMixin):
    def __init__(self, rolls=[1], col='Sales'):
        print('Rolling Weekday Initialized')
        self.rolls = rolls
        self.col = col

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print('Rolling Weekday transform called')
        tmp = X[['Salesdate','Team Name', 'Product Subcategory', 'weekday', self.col]]
        for i in self.rolls:
            print('Roll Weekday Range: ' + str(i))
            min_period = int(round(i / 2, 0))
            shifted = tmp.copy()
            shifted[str(i) + '_wk_std_' + self.col] = shifted.sort_values('Salesdate').groupby(['Team Name', 'Product Subcategory', 'weekday'])[self.col].apply(lambda x: x.rolling(i, min_periods=min_period).std())
            del shifted[self.col]
            shifted['Salesdate'] = shifted['Salesdate'] + pd.Timedelta(7, unit='D')
            X = pd.merge(X, shifted, on=['Salesdate','Team Name','Product Subcategory', 'weekday'], how='left')
        print('Creating rolling Weekday features - Completed')
        return X.fillna(0)

class exponential_moving_average(BaseEstimator, TransformerMixin):
    def __init__(self, lags=[7], col='Sales', group_by=['Team Name', 'Product Subcategory'], time_col='Salesdate'):
        print('EMA Initialized')
        self.lags = lags
        self.col = col
        self.group_by = group_by
        self.time_col = time_col

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print('EMA transform called')
        tmp = X[['Salesdate','Team Name', 'Product Subcategory', self.col]]
        for i in self.lags:
            print('Roll Range: ' + str(i))
            min_period = int(round(i / 2, 0))
            shifted = tmp.copy()
            shifted[str(i)+'D_ema_'+ self.col] = shifted.sort_values(self.time_col).groupby(self.group_by)[self.col].apply(lambda x: x.ewm(i, min_periods=min_period).mean())
            del shifted[self.col]
            shifted[self.time_col] = shifted[self.time_col] + pd.Timedelta(1, unit='D')
            X = pd.merge(X, shifted, on=['Salesdate','Team Name','Product Subcategory'], how='left')
        print('EMA features - Completed')
        return X.fillna(0)

class trend_features(BaseEstimator, TransformerMixin):
    def __init__(self, lags=[180], col='Sales', group_by=['Team Name', 'Product Subcategory'], time_col='Salesdate'):
        print('EMA Initialized')
        self.lags = lags
        self.col = col
        self.group_by = group_by
        self.time_col = time_col

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print('EMA transform called')
        tmp = X[['Salesdate','Team Name', 'Product Subcategory', self.col]]
        for i in self.lags:
            print('Trend Range: ' + str(i))
            shifted = tmp.copy()
            shifted['ema'] = shifted.sort_values(self.time_col).groupby(self.group_by)[self.col].apply(lambda x: x.ewm(i).mean())
            shifted['3d_mean'] = shifted.sort_values(self.time_col).groupby(self.group_by)[self.col].apply(lambda x: x.rolling(3).mean())
            shifted['mean_trend'] = shifted['3d_mean'] - shifted['ema']
            shifted['diff'] = (shifted[self.col] / shifted['ema']) - 1
            shifted['signal'] = 0
            shifted['signal'] = np.where(shifted['diff'] > 0, 1, 0)
            shifted['signal_diff'] = shifted.sort_values(self.time_col).groupby(self.group_by)['signal'].apply(lambda x: x.diff(2))
            shifted['trend_prep'] = shifted.sort_values(self.time_col).groupby(self.group_by)['signal'].apply(lambda x: x.rolling(3).sum())
            # if there has been 3 or more positive signals then upwards trend and visa versa with 0
            shifted['trend'] = 1
            shifted['trend'] = np.where(shifted['trend_prep'] == 3, 2, 1)
            shifted['trend'] = np.where(shifted['trend_prep'] <= 0, 0, shifted['trend'])
            #shifted['signal'] = np.where(shifted['diff'] < -0, 1, shifted['signal'])
            del shifted[self.col]
            del shifted['diff']
            del shifted['signal_diff']
            del shifted['trend_prep']
            del shifted['signal']
            del shifted['3d_mean']
            shifted[self.time_col] = shifted[self.time_col] + pd.Timedelta(1, unit='D')
            X = pd.merge(X, shifted, on=['Salesdate','Team Name','Product Subcategory'], how='left')
        print('EMA features - Completed')
        return X.fillna(0)

class date_features(BaseEstimator, TransformerMixin):
    def __init__(self, datenum=False):
        print('Date Feature Initialized')
        self.datenum = datenum

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print('Date Feature Initialized')
        tmp = X.copy()
        tmp['month'] = tmp['Salesdate'].dt.month
        tmp['week'] = tmp['Salesdate'].dt.week
        tmp['day'] = tmp['Salesdate'].dt.day
        tmp['weekday'] = tmp['Salesdate'].dt.dayofweek
        if self.datenum:
            min_date = tmp['Salesdate'].min()
            tmp['Salesdate_num'] = (tmp['Salesdate'] - min_date).dt.days
        print('Creating date features - Completed')
        return tmp

class feature_encodings(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_encode):
        print('Date Feature Initialized')
        self.cols = columns_to_encode

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print('Date Feature Initialized')
        tmp = X.copy()
        tmp = pd.get_dummies(tmp, columns=self.cols, drop_first=True)
        return tmp

class feature_power_transform(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_transform):
        print('Date Feature Initialized')
        self.cols = columns_to_transform

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print('Power Transform Initialized')
        transformer = PowerTransformer(method='yeo-johnson')
        tmp = X.copy()
        tmp[self.cols] = transformer.fit_transform(tmp[[self.cols]])
        return X

class final(BaseEstimator, TransformerMixin):
    def __init__(self, remove):
        print('Final step Initialized')
        self.remove = remove

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        tmp = X.copy()
        tmp.drop(self.remove, axis=1, inplace=True)
        return tmp