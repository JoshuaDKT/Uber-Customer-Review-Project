import traceback

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from modules.data_preparation.custom_feature_creation import create_custom_features
from modules.data_preparation.missing_value_imputation import impute_missing_values


def preprocess_categorical_data(df):
    # --Encoding Techniques--

    # Apply label encoding to convert categorical data into numerical data. This is useful for data which has a natural
    # order (small, medium, large, etc.).
    le = LabelEncoder()
    df_label = le.fit_transform(df['polarity'])

    # Convert categorical data into binary matrices. This is useful for unordered data.
    df_onehot = pd.get_dummies(df, columns=['score'])

    # One hot encoding can also be applied with sklearn
    onehot = OneHotEncoder()
    # df_onehot = onehot.fit_transform(df)

    # Apply target encoding, a technique used to transform categorical data into numerical data by encoding each
    # category as the function of the target variable. This is useful for data with a lot of unique categories.
    def target_encode(train, test, column, target):
        mean_target = train.groupby(column)[target].mean()
        return train[column].map(mean_target), test[column].map(mean_target)

    # -- Feature Interactions --

    # Feature interactions help with data processing speed and improve overall model performance.
    # --This area was left blank due to a lack of feature interactions in the dataset.

    return df_label, df_onehot


if __name__ == '__main__':
    try:
        df = pd.DataFrame(pd.read_csv('../../data/uber_reviews_without_reviewid.csv'))
        df = impute_missing_values(df)
        df = create_custom_features(df)
        print(preprocess_categorical_data(df)[1].columns)
    except Exception as e:
        print("An error occurred:")
        print(e)
        traceback.print_exc(limit=2)
