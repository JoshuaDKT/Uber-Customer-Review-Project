import traceback
import pandas as pd
from modules.data_preparation.text_data_preprocessing import preprocess_text_data
from modules.data_preparation.categorical_data_preprocessing import preprocess_categorical_data
from modules.data_preparation.custom_feature_creation import create_custom_features
from modules.data_preparation.dimensionality_reduction import reduce_dimensionality
from modules.data_preparation.feature_quality_checks import validate_features
from modules.data_preparation.missing_value_imputation import impute_missing_values
from modules.data_preparation.numerical_data_preprocessing import preprocess_numerical_data
from modules.data_preparation.time_based_features import create_time_based_features


def prepare_data(file_path):
    # Store the dataset to a variable and convert to a pandas dataframe
    df = pd.DataFrame(pd.read_csv(file_path))

    # --- Missing Value Imputation ---
    df = impute_missing_values(df)

    # --- Custom Feature Creation ---
    df = create_custom_features(df)

    # --- Numerical Data Preprocessing ---
    '''
    Index
    0: Return Standard Scaled Data
    1: ReturnMin-Max Scaled Data
    2: Return Robust Scaled Data
    3: Return Log Scaled Data
    4: Return Square root scaled data
    5: Return Box cox transformed data
    6: Return Binned data
    7: Return Custom binned data
    '''
    ndp_df = preprocess_numerical_data(df)

    # --- Categorical Data Preprocessing ---
    '''
    Index 
    0: Label Encoded df (Not recommended since label encoding turns categories into integers, 
    and the categorical column--score--is a series of integers anyway) 
    1: One-hot encoded df (Preferred)
    '''
    cdp_df = preprocess_categorical_data(df)[1]

    # --- Text Data Preprocessing ---
    '''
    Index:
    0: Return original data, with new text features
    1: Return bag of words dataframe
    2: Return the vocabulary for the TF-IDF method
    3: Return the TF-IDF matrix, a list of lists, each representing a document
    '''
    df = preprocess_text_data(df)[0]

    # --- Time Based Features ---
    df = create_time_based_features(df)

    # --- Dimensionality Reduction ---
    dr_df = reduce_dimensionality(df[['negative_word_count', 'positive_word_count', 'word_count']], df['score'])

    # --- 8. Feature Validation and Testing ---
    '''
    Index
    0: unique_high_corr_pairs - Lists with columns with high correlations
    1: low_var - Checks the data for low variances
    2: importances - Rates numerical columns for importance
    '''
    validate_df = validate_features(df)

    return df


if __name__ == '__main__':
    try:
        df = prepare_data('../../data/uber_reviews_without_reviewid.csv')
        df.to_csv(r'D:\Projects\PyCharm\Uber Customer Review Analysis\data\prepared_uber_data.csv')
    except Exception as e:
        print("An error occurred:")
        print(e)
        traceback.print_exc(limit=3)
