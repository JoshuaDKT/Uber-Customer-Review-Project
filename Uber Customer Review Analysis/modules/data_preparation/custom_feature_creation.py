import traceback
import numpy as np
import pandas as pd
from modules.data_preparation.missing_value_imputation import impute_missing_values


def create_custom_features(df):
    # Remove the userImage column since it is empty
    df = df.drop('userImage', axis=1)

    # Create a polarity column, which indicates the polarity of the score (a score of 1 or 2 is negative, while 4 or 5
    # is positive, and a score of 3 has polarity 0).
    conditions = [
        df['score'] <= 2,
        df['score'] >= 4,
    ]
    choices = [1, 2]
    df['polarity'] = np.select(conditions, choices, default=0)

    # Reverse the data to start with the earliest date
    df = df[::-1].reset_index(drop=True)

    return df


if __name__ == '__main__':
    try:
        df = pd.read_csv('../../data/uber_reviews_without_reviewid.csv')
        df = impute_missing_values(df)
        df = create_custom_features(df)
        print(df.columns)
    except Exception as e:
        print("An error has occurred:")
        print(e)
        traceback.print_exc(limit=2)
