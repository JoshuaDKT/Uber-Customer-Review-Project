import traceback
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.manifold import TSNE
from modules.data_preparation.custom_feature_creation import create_custom_features
from modules.data_preparation.missing_value_imputation import impute_missing_values
from modules.data_preparation.text_data_preprocessing import preprocess_text_data
from modules.data_preparation.time_based_features import create_time_based_features


def reduce_dimensionality(x, y):
    # Statistical feature selection, y represents the dependent variables
    selector = SelectKBest(score_func=f_classif, k=3)
    df_selected = selector.fit_transform(x, y)

    # Recursive Feature Elimination
    rf = RandomForestClassifier()
    rfe = RFE(estimator=rf, n_features_to_select=3)
    df_rfe = rfe.fit_transform(x, y)

    # PCA
    pca = PCA(n_components=0.95)  # Keep 95% of variance
    df_pca = pca.fit_transform(x)

    # t-SNE
    tsne = TSNE(n_components=2)
    df_tsne = tsne.fit_transform(x)

    return df_selected, df_rfe, df_pca, df_tsne


if __name__ == '__main__':
    try:
        df = pd.DataFrame(pd.read_csv('../../data/uber_reviews_without_reviewid.csv'))
        df = impute_missing_values(df)
        df = create_custom_features(df)
        df = create_time_based_features(df)
        df = preprocess_text_data(df)[0]
        print(reduce_dimensionality(df[['negative_word_count', 'positive_word_count', 'word_count']], df['score'])[1])
    except Exception as e:
        print("An error has occurred:")
        print(e)
        traceback.print_exc(limit=2)