from modules.data_preparation import prepare_data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''
Parse through the datetime column
Filter by week, and get a sum of positive scores and negative scores
Create the stacked bar plot
'''


def weekly_review_count(data):
    # Group by week, then count how many entries are in the score column.
    weekly_review_counts = data.groupby('week')['score'].count().reset_index()
    weekly_review_counts.rename(columns={'score':'review_count'}, inplace=True)

    # Convert the 'Week' column to a categorical type
    weekly_review_counts['week'] = weekly_review_counts['week'].astype(str)

    # Plot the data
    plt.bar(weekly_review_counts['week'], weekly_review_counts['review_count'], color='skyblue')
    plt.xlabel('Week')
    plt.ylabel('Review Count')
    plt.title('Review Count per Week')

    # Set y-axis limits to ensure the bars do not extend past the top
    plt.ylim(0, max(weekly_review_counts['review_count']) * 1.1)  # Adding 10% padding to the top

    # Show the plot
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    weekly_review_count(prepare_data('../data/uber_reviews_without_reviewid.csv'))