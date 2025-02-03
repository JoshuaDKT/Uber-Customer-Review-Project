import traceback
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline

from modules.data_preparation.data_preparation import prepare_data


def visualize_data(data):
    # Combine the graphs into one figure.
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # --- Review length box and whisker subplot ---
    axes[0, 0].boxplot(data['word_count'])
    axes[0, 0].margins(y=0.1)
    axes[0, 0].set_ylim(0, 45)
    axes[0, 0].set_yticks(range(0, 45, 5))
    axes[0, 0].grid()
    axes[0, 0].margins(y=0.1)
    axes[0, 0].set_title('Review Length')

    # --- Review Length Pie Chart ---
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
    axes[0, 1].pie(values, labels=labels, autopct=make_autopct(values), colors=['red', 'green', 'gray'])
    axes[0, 1].set_title("Percentages of Reviews by Polarity")
    axes[0, 1].axis('equal')
    axes[0, 1].text(0.5, -1, 'Positive polarity: score of 4 or 5')
    axes[0, 1].text(0.5, -1.2, 'Negative polarity: score of 1 or 2')

    # --- Review Polarity Bar Plot ---
    polarity_review_count = data.groupby('week').agg(
        negative_polarity=('polarity', lambda x: (x == 1).sum()),
        positive_polarity=('polarity', lambda x: (x == 2).sum())
    )

    # Set the number of categories
    x = np.arange(len(data['week'].unique()))

    # Set the bar width
    bar_width = 0.4

    # Create the plot
    axes[1, 0].bar(x - bar_width / 2, polarity_review_count['negative_polarity'], width=bar_width,
           label='Negative Review Count', color='red')
    axes[1, 0].bar(x + bar_width / 2, polarity_review_count['positive_polarity'], width=bar_width,
           label='Positive Review Count', color='green')
    axes[1, 0].set_xlabel('Week of 2024')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Review Polarities per Week')
    axes[1, 0].set_xticks(x, data['week'].unique(), rotation=45)
    axes[1, 0].set_ylim(0, max(max(polarity_review_count['negative_polarity']),
                       max(polarity_review_count['positive_polarity'])) * 1.1)
    axes[1, 0].grid()
    axes[1, 0].tick_params(axis='x', rotation=45)

    # Add data points to the bar plot
    for i, value in enumerate(polarity_review_count['negative_polarity']):
        axes[1, 0].text(i - bar_width / 2, value + 0.5, str(value), ha='center', va='bottom')
    for i, value in enumerate(polarity_review_count['positive_polarity']):
        axes[1, 0].text(i + bar_width / 2, value + 0.5, str(value), ha='center', va='bottom')

    # --- Score Bar Plot ---
    # Count occurrences of each rating
    rating_counts = data['score'].value_counts().sort_index()

    # Create the bar plot
    plt.bar(rating_counts.index, rating_counts.values, color=['red', 'orange', 'yellow', 'green', 'blue'])

    # Add title and labels
    plt.xlabel('Score')
    plt.ylabel('Count')
    plt.title('Score Distribution')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add data points to the bar plot
    for i, value in enumerate(rating_counts.values):
        plt.text(rating_counts.index[i], value + 50, str(value), ha='center')

    # Add ONE legend for all subplots
    # fig.legend(['Negative', 'Positive'], loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2)

    # Create a tight layout to prevent overlapping
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Add an overall title
    fig.suptitle("Analyzing Uber Customer Reviews: Insights & Trends")

    # Save the figure
    plt.savefig(r"D:\Projects\PyCharm\Uber Customer Review Analysis\figures\Figure_1.pdf", dpi=300, bbox_inches='tight')

    # Show the combined visualization
    plt.show()


if __name__ == '__main__':
    try:
        visualize_data(pd.read_csv('../../data/prepared_uber_data.csv'))
    except Exception as e:
        print("An error has occurred:")
        print(e)
        traceback.print_exc(limit=3)
