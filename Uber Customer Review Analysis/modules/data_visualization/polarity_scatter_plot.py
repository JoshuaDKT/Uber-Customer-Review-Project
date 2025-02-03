from modules.data_preparation.data_preparation import prepare_data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''
For a scatter plot we want two sets of values.
In this case, we'll count the number of negative keywords, and positive keywords
'''


def create_polarity_scatter_plot(data):
    # Create the scatter plot, with the negative and positive keyword counts as the values
    plt.scatter(data['negative_word_count'], data['positive_word_count'])
    plt.title('Keyword Count by Polarity')
    plt.xlabel('Negative Keywords Total')
    plt.ylabel('Positive Keywords Total')

    # Show the plot
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    create_polarity_scatter_plot(prepare_data('../../data/uber_reviews_without_reviewid.csv'))
