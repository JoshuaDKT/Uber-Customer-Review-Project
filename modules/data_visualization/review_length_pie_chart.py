from modules.data_preparation.data_preparation import prepare_data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''
Get the total number of words for reviews with a positive review polarity (rating of 4 or 5), then do the same with
negative reviews. The values will be used for the pie chart.

To do this:
1. Group by polarity
2. Sum using the word count column
3. Visualize using the plot function

The values are as follows:
label = Polarity
values = Number of reviews for each polarity
'''


def create_review_length_pie_chart(data):
    fig, ax = plt.subplots()

    values = [(data['polarity'] == 1).sum(),
              (data['polarity'] == 2).sum(),
              (data['polarity'] == 0).sum()]
    labels = ['Negative', 'Positive', 'Neutral']

    # Create a function to format the percentage of the values, with the actual value underneath it
    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))
            return f'{pct:.1f}%\n({val:,})'

        return my_autopct

    # Create the plot
    ax.pie(values, labels=labels, autopct=make_autopct(values), colors=['red', 'green', 'gray'])
    ax.set_title("Percentages of Reviews by Polarity")
    ax.axis('equal')


if __name__ == '__main__':
    create_review_length_pie_chart(prepare_data('../../data/uber_reviews_without_reviewid.csv'))
