from modules.data_preparation import data_preparation
from modules.data_visualization import data_visualization


def main(file_path):
    df = data_preparation.prepare_data(file_path)
    data_visualization.visualize_data(df)


if __name__ == '__main__':
    main(r'D:\Projects\PyCharm\Uber Customer Review Analysis\data\uber_reviews_without_reviewid.csv')
