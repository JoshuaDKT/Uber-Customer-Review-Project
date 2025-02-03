import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer


def impute_missing_values(df, n_neighbors=5):
    # Select only the numeric columns.
    df_numeric = df.select_dtypes(include=[np.number])
    print("Original numeric DataFrame shape:", df_numeric.shape)

    # Identify completely empty columns (all NaN), since they interfere with KNN imputation.
    empty_columns = df_numeric.columns[df_numeric.isnull().all()]
    print("Empty numeric columns:", empty_columns)

    # Drop empty columns temporarily for imputation
    df_numeric_non_empty = df_numeric.drop(columns=empty_columns)
    print("Numeric DataFrame (non-empty) shape:", df_numeric_non_empty.shape)

    # Apply KNN imputation for the non-empty columns
    knn_imputer = KNNImputer(n_neighbors=n_neighbors)
    df_knn = pd.DataFrame(
        knn_imputer.fit_transform(df_numeric_non_empty),
        columns=df_numeric_non_empty.columns,
        index=df_numeric_non_empty.index,
    )
    print("Imputed numeric DataFrame shape:", df_knn.shape)

    # Add back empty columns as filled with 0 or another default value
    for col in empty_columns:
        df_knn[col] = 0  # Replace with other default values if needed

    # Ensure all columns align with the original numeric DataFrame
    df_knn = df_knn.reindex(columns=df_numeric.columns)

    # Replace the numeric columns in the original DataFrame
    df[df_numeric.columns] = df_knn

    return df


if __name__ == '__main__':
    df = pd.DataFrame(pd.read_csv('../../data/uber_reviews_without_reviewid.csv'))

    # Apply the KNN imputation
    try:
        imputed_df = impute_missing_values(df)
        print(imputed_df)
    except ValueError as e:
        print(f"Error: {e}")