from modules.data_preparation.data_preparation import prepare_data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def create_review_polarity_plot(data):
    fig, ax = plt.subplots()

    polarity_review_count = data.groupby('week').agg(
        negative_polarity=('polarity', lambda x: (x == 1).sum()),
        positive_polarity=('polarity', lambda x: (x == 2).sum())
    )

    # Set the number of categories
    x = np.arange(len(data['week'].unique()))

    # Set the bar width
    bar_width = 0.4

    # Create the plot
    ax.bar(x - bar_width / 2, polarity_review_count['negative_polarity'], width=bar_width,
           label='Negative Review Count', color='red')
    ax.bar(x + bar_width / 2, polarity_review_count['positive_polarity'], width=bar_width,
           label='Positive Review Count', color='green')
    ax.set_xlabel('Week of 2024')
    ax.set_ylabel('Count')
    ax.set_title('Review Polarities per Week')
    ax.set_xticks(x, data['week'].unique(), rotation=45)
    ax.set_ylim(0, max(max(polarity_review_count['negative_polarity']),
                       max(polarity_review_count['positive_polarity'])) * 1.1)
    ax.grid()
    ax.legend()

    # Add data points to the bar plot
    for i, value in enumerate(polarity_review_count['negative_polarity']):
        plt.text(i - bar_width / 2, value + 0.5, str(value), ha='center', va='bottom')
    for i, value in enumerate(polarity_review_count['positive_polarity']):
        plt.text(i + bar_width / 2, value + 0.5, str(value), ha='center', va='bottom')

    # Adjust layout to prevent overlapping
    fig.tight_layout()

    # Show the combined visualization
    plt.show()


if __name__ == '__main__':
    create_review_polarity_plot(prepare_data('../../data/uber_reviews_without_reviewid.csv'))
