from modules.data_preparation.data_preparation import prepare_data
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import LogLocator, LogFormatter


def create_review_length_box_and_whisker(data):
    fig, ax = plt.subplots()

    # Create the plot
    ax.boxplot(data['word_count'])
    ax.margins(y=0.1)
    ax.set_ylim(0, 45)
    ax.set_yticks(range(0, 45, 5))
    ax.grid()
    ax.margins(y=0.1)


if __name__ == '__main__':
    create_review_length_box_and_whisker(prepare_data('../../data/uber_reviews_without_reviewid.csv'))
