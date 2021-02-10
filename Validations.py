import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import FeatureEngineering
from sklearn.pipeline import Pipeline
import PipelineTools as pt


def multi_step_validation(df, n_test, time_feature, target):
    predictions = []
    # split dataset
    train, test = timeseries_train_test_split(df, n_test, time_feature)
    # set the start date of the forecast range
    start_forecast_date = test[time_feature].min()
    # create history dataframe
    history = df.loc[df[time_feature] <= start_forecast_date]
    # step over each day in the forecast range
    for day_no in range(n_test):
        # set the current forecast date
        forecast_date = start_forecast_date + pd.Timedelta(day_no, unit='D')
        # split forecast data
        test_X_before_engineering = test.loc[test[time_feature] == forecast_date, [time_feature, 'Team Name', 'Product Category', 'Product Subcategory']]
        test_X = one_step_engineering_split(train, test_X_before_engineering, time_feature, 60, forecast_date)
        # fit model to history and make a prediction
        predicted_values = xgboost_forecast(train, test_X, time_feature, target, forecast_date)
        # save predictions
        save_predictions = predicted_values.copy()
        save_predictions[time_feature] = forecast_date
        predictions.append(save_predictions)
        # add predictions to the train set for the next day forecast
        predicted_values.rename(columns={'predict':target}, inplace=True)
        predicted_values[time_feature] = forecast_date
        train = pd.concat([train, predicted_values], ignore_index=True)
        # Report progress
        print('Step {step_no} out of {total} completed'.format(step_no=(day_no+1), total=n_test))
    # concat each predictions per day into one df
    predictions_df = pd.concat(predictions, ignore_index=True)
    predictions_df = pd.merge(predictions_df, test[[time_feature, 'Team Name', 'Product Subcategory', target]], on=[time_feature, 'Team Name', 'Product Subcategory'], how='left')
    MAE = mean_absolute_error(predictions_df[target], predictions_df['predict'])
    RMSE = mean_squared_error(predictions_df[target], predictions_df['predict'], squared=False)
    return MAE, RMSE, predictions_df

def one_step_engineering_split(train, test, time_feature, max_lag, forecast_date):
    cutoff_date = forecast_date - pd.Timedelta(max_lag, unit='D')
    # engineering
    # test = FeatureEngineering.date_feature(test, datenum=False)
    # test = FeatureEngineering.one_step_lag_feature(train, test, [1, 9, 10, 11, 14, 22, 27, 28], 'Sales', time_feature, cutoff_date)
    # test = FeatureEngineering.one_step_rolling_feature(train, test, [7], 'Sales', time_feature, cutoff_date)
    # test = FeatureEngineering.one_step_weekDay_feature(train, test, [4,5,6], 'Sales', time_feature, cutoff_date)

    pipeline_val_steps = [
        ('lag features', pt.lag_feature(lags=[1, 6, 7, 14, 21, 28], col='Sales')),
        ('roll mean', pt.rolling_mean(lags=[7, 8, 14, 15, 21, 29], col='Sales')),
        ('roll ewm', pt.exponential_moving_average(lags=[4, 6, 8, 9, 29], col='Sales')),
        ('roll max', pt.rolling_max(lags=[4, 7, 8, 14], col='Sales')),
        ('roll min', pt.rolling_min(lags=[4, 6, 10], col='Sales')),
        ('roll std', pt.rolling_std(lags=[4, 15, 22, 29], col='Sales')),
        ('date features', pt.date_features()),
        ('roll wk mean', pt.rolling_weekday_mean(rolls=[2, 3, 4], col='Sales')),
        ('roll wk max', pt.rolling_weekday_max(rolls=[2, 3, 4], col='Sales')),
        ('roll wk min', pt.rolling_weekday_min(rolls=[2, 4], col='Sales')),
        ('roll wk std', pt.rolling_weekday_std(rolls=[2, 3, 4], col='Sales')),
        ('final step', pt.final(['month', 'week', 'day']))
    ]

    pipeline_val = Pipeline(steps=pipeline_val_steps)
    train = pipeline_val.fit_transform(train)

    test = pd.merge(test, train, on=[time_feature, 'Team Name', 'Product Subcategory'], how='left')

    return test.drop([time_feature], axis=1)



def timeseries_train_test_split(df, n_test, time_feature):
    tmp = df.copy()
    max_date = tmp['Salesdate'].max()
    cutoff_date = max_date - pd.Timedelta(n_test, unit='D')
    return tmp.loc[tmp[time_feature] <= cutoff_date], tmp.loc[tmp[time_feature] > cutoff_date]


def xgboost_forecast(train, test, time_feature, target, forecast_day):
    # split training data
    validation_day = forecast_day - pd.Timedelta(1, unit='D')
    print('forecast day: {forc} and validation day: {val}'.format(forc=forecast_day, val=validation_day))
    train_X = train.loc[train[time_feature] < validation_day].drop([target, time_feature], axis=1)
    train_Y = train.loc[train[time_feature] < validation_day, target]
    valid_X = train.loc[train[time_feature] == validation_day].drop([target, time_feature], axis=1)
    valid_Y = train.loc[train[time_feature] == validation_day, target]
    # save the order of columns to ensure match between train and test df
    column_order = train_X.columns.tolist()
    # create model
    model = XGBRegressor(
        max_depth=4,
        n_estimators=1000,
        min_child_weight=300,
        colsample_bytree=0.8,
        subsample=0.9,
        eta=0.3,
        seed=2017)
    # fit model
    model.fit(
        train_X,
        train_Y,
        eval_metric="rmse",
        eval_set=[(train_X, train_Y), (valid_X, valid_Y)],
        verbose=False,
        early_stopping_rounds=10
    )
    # make a one-step prediction
    predictions = model.predict(test.loc[:,column_order])

    test.loc[:, 'predict'] = predictions
    return test
