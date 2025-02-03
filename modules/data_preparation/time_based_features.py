import traceback
import numpy as np
import pandas as pd
from modules.data_preparation.custom_feature_creation import create_custom_features
from modules.data_preparation.missing_value_imputation import impute_missing_values


def create_time_based_features(data):
    date_column = data['at']

    # Rename the column to datetime to be more contextually appropriate
    data = data.rename(columns={'at': 'datetime'})

    # Convert the DateTime column to a pandas datetime object
    data['datetime'] = pd.to_datetime(data['datetime'])

    # Extract date-time features from the date-time column
    data['date'] = data['datetime'].dt.date
    data['time'] = data['datetime'].dt.time
    data['year'] = data['datetime'].dt.year
    data['month'] = data['datetime'].dt.month
    data['day'] = data['datetime'].dt.day
    data['dayofweek'] = data['datetime'].dt.dayofweek
    data['hour'] = data['datetime'].dt.hour
    data['is_weekend'] = data['datetime'].dt.dayofweek.isin([5, 6])  # 5 and 6 represent Saturday and Sunday
    data['month_day'] = data['datetime'].dt.strftime('%m-%d')

    # Extract the start of the week (Monday) and the end of the week (Sunday), then combine the two dates into a week
    # column
    data['week_start'] = (data['datetime'] - pd.to_timedelta(data['datetime'].dt.weekday, unit='d')).dt.strftime(
        '%m/%d')

    # Extract the end date of the week in "month/day" format
    data['week_end'] = (data['datetime'] + pd.to_timedelta(6 - data['datetime'].dt.weekday, unit='d')).dt.strftime(
        '%m/%d')

    # Combine the week start and end into a single "week" column
    data['week'] = data['week_start'] + ' - ' + data['week_end']

    # Cyclical encoding for periodic features. This ensures that numeric values that are far apart but have
    # intrinsic correlations have their relationships made clear.
    data[f'{'datetime'}_month_sin'] = np.sin(2 * np.pi * data['datetime'].dt.month / 12)
    data[f'{'datetime'}_month_cos'] = np.cos(2 * np.pi * data['datetime'].dt.month / 12)

    return data


if __name__ == '__main__':
    try:
        df = pd.DataFrame(pd.read_csv('../../data/uber_reviews_without_reviewid.csv'))
        df = impute_missing_values(df)
        df = create_custom_features(df)
        df = create_time_based_features(df)
        print(df)
    except Exception as e:
        print("An error has occurred:")
        print(e)
        traceback.print_exc(limit=2)
