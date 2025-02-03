import traceback

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from modules.data_preparation.custom_feature_creation import create_custom_features
from modules.data_preparation.missing_value_imputation import impute_missing_values
from modules.data_preparation.text_data_preprocessing import preprocess_text_data
from modules.data_preparation.time_based_features import create_time_based_features


def validate_features(data):
    # Check for correlation
    numerical_df = data.select_dtypes(include=['float64', 'int64'])
    corr_matrix = numerical_df.corr()
    high_corr = np.where((np.abs(corr_matrix) > 0.95) & (np.eye(corr_matrix.shape[0]) == 0))

    # Exclude the diagonals
    unique_high_corr_pairs = [(corr_matrix.index[i], corr_matrix.columns[j]) for i, j in zip(*high_corr)]

    # Check for low variance
    low_var = numerical_df.var() < 0.01

    # Check for feature importance
    rf = RandomForestClassifier()
    x = data[['word_count', 'negative_word_count', 'positive_word_count', 'thumbsUpCount']]
    y = data['score']
    rf.fit(x, y)
    importances = pd.DataFrame({
        'feature': x.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    return unique_high_corr_pairs, low_var, importances


if __name__ == '__main__':
    try:
        df = pd.DataFrame(pd.read_csv('../../data/uber_reviews_without_reviewid.csv'))
        df = impute_missing_values(df)
        df = create_custom_features(df)
        df = create_time_based_features(df)
        df = preprocess_text_data(df)[0]
        print(validate_features(df)[0])
    except Exception as e:
        print("An error has occurred:")
        print(e)
        traceback.print_exc(limit=2)
