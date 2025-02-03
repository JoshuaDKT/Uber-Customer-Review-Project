for i, value in enumerate(polarity_review_count['negative_polarity']):
    axes[1, 0].text(i - bar_width / 2, value + 0.5, str(value), ha='center', va='bottom')
for i, value in enumerate(polarity_review_count['positive_polarity']):
    axes[1, 0].text(i + bar_width / 2, value + 0.5, str(value), ha='center', va='bottom')