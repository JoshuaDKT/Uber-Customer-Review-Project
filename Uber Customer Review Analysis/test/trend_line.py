import numpy as np
import matplotlib.pyplot as plt

# Sample data
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 9, 16, 25])  # A quadratic relationship

# Plot the original data
plt.plot(x, y, label='Data', marker='o')

# Fit a quadratic curve (degree 2)
coefficients = np.polyfit(x, y, 2)

# Generate the trend line (quadratic curve)
trend_line = np.polyval(coefficients, x)

# Plot the trend line
plt.plot(x, trend_line, label='Quadratic Trend Line', linestyle='--', color='red')

# Adding labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Line Plot with Curved Trend Line')
plt.legend()

# Show the plot
plt.show()