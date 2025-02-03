import traceback
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, KBinsDiscretizer
from modules.data_preparation.missing_value_imputation import impute_missing_values
from modules.data_preparation.custom_feature_creation import create_custom_features
from modules.data_preparation.text_data_preprocessing import preprocess_text_data


def preprocess_numerical_data(df):
    # Select only the columns with numeric values.
    numeric_cols = df.select_dtypes(include=["float", "int"]).columns

    # Apply standard scaling with the z-score. The z-score is the measure of how many standard deviations the datapoint
    # is from the mean.
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[numeric_cols])

    # Apply min-max scaling, a technique focused on concentrating data within a certain range, otherwise known as
    # normalization.
    minmax = MinMaxScaler()
    df_normalized = minmax.fit_transform(df[numeric_cols])

    # Apply robust scaling, in which the data has its median removed, and data is instead scaled on the interquartile
    # range. This method handles outliers better.
    robust = RobustScaler()
    df_robust = robust.fit_transform(df[numeric_cols])

    # -- Handling Skewness --

    # Apply log transformation, to handle extreme outliers.
    df_log = np.log1p(df[numeric_cols])  # log1p handles zero values

    # Apply square root transformation to handle moderately skewed data.
    df_sqrt = np.sqrt(df[numeric_cols])

    # To stabilize the variance and make the data more approximate to a normal distribution, apply a box-cox
    # transformation
    df_boxcox, lambda_param = stats.boxcox(df['score'])

    # Apply binning to separate the data into different 'bins', the default is equal-width binning.
    kbd = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
    df_binned = kbd.fit_transform(df[numeric_cols])

    # Custom binning is also available, if the bins need to be a certain size
    def custom_bins(x, bins=4):
        return pd.cut(x, bins=bins, labels=False)

    df_custom_binned = custom_bins(df['score'])

    return df_scaled, df_normalized, df_robust, df_log, df_sqrt, df_boxcox, df_binned, df_custom_binned


if __name__ == '__main__':
    try:
        df = pd.DataFrame(pd.read_csv('../../data/uber_reviews_without_reviewid.csv'))
        df = impute_missing_values(df)
        df = create_custom_features(df)
        print(preprocess_numerical_data(df)[5])
    except Exception as e:
        print("An error occurred:")
        print(e)
        traceback.print_exc(limit=2)
