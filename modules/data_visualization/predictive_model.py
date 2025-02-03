import traceback

from modules.data_preparation.data_preparation import prepare_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report


def create_predictive_model(data):
    # Split the numerical data by their independent and dependent values.
    x = data[['thumbsUpCount', 'negative_word_count', 'positive_word_count', 'word_count']]
    y = data['score']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Initialize the model.
    model = RandomForestClassifier(random_state=42)

    # Train the model.
    model.fit(x_train, y_train)

    # Create the prediction.
    y_pred = model.predict(x_test)

    # Evaluation Metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)  # for current data: explains 0.5009893917098205% of variance
    print(f"MSE: {mse}, R^2: {r2}")

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print(classification_report(y_test, y_pred))


if __name__ == '__main__':
    try:
        data = prepare_data('../../data/uber_reviews_without_reviewid.csv')
    except Exception as e:
        print("An error has occurred:")
        print(e)
        traceback.print_exc(limit=2)
