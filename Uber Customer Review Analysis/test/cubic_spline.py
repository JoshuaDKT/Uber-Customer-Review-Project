import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Sample data: Dates and corresponding values
dates = pd.date_range('2024-01-01', periods=5, freq='D')  # 5 days starting from Jan 1, 2024
values = np.array([2, 4, 9, 16, 25])  # A quadratic-like relationship

# Convert dates to numerical values (number of days since the first date)
x_numeric = (dates - dates.min()).days  # Days since the first date

# Fit a quadratic curve (degree 2)
coefficients = np.polyfit(x_numeric, values, 2)

# Generate the trend line (quadratic curve)
trend_line = np.polyval(coefficients, x_numeric)

# Plot the original data
plt.plot(dates, values, 'o', label='Data')

# Plot the curved line (polynomial)
plt.plot(dates, trend_line, label='Curved Trend Line', color='red')

# Adding labels and title
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Line Plot with Curved Trend Line (Polynomial)')

# Formatting the x-axis to show the date
plt.xticks(rotation=45)  # Rotate date labels for readability

plt.legend()
plt.tight_layout()  # Ensure everything fits in the plot
plt.show()