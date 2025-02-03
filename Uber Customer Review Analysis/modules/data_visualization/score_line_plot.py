import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline, make_interp_spline
from modules.data_preparation.data_preparation import prepare_data


def create_score_line_plot(data):
    fig, ax = plt.subplots()

    # Convert dates to numerical values for interpolation
    x = np.arange(len(data['datetime']))
    y = data['score']

    # Fit a linear regression line
    coeffs = np.polyfit(x, y, 1)  # Degree 1 for a linear fit
    trendline = np.polyval(coeffs, x)

    # Create cubic spline interpolation
    x_smooth = np.linspace(x.min(), x.max(), 20)
    y_smooth = make_interp_spline(x, y)(x_smooth)

    # Plot the smooth curve and trendline
    ax.plot(data['datetime'][x_smooth.astype(int)], y_smooth, color='red', label='Smoothed Review Score')
    ax.plot(data['datetime'], trendline, color='blue', label='Trendline')

    # Customize the plot
    ax.set_title('Smoothed Review Scores Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Review Score')
    ax.set_ylim(1, 5)
    ax.grid(True)
    ax.legend()

    # Automatically format the x-axis dates
    fig.autofmt_xdate()


if __name__ == '__main__':
    create_score_line_plot(prepare_data('../../data/uber_reviews_without_reviewid.csv'))