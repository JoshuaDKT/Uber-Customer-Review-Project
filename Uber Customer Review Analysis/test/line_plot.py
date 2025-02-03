import matplotlib.pyplot as plt

# Sample data
x = [1, 2, 3, 4, 5]  # X-axis values
y = [2, 3, 5, 7, 11]  # Y-axis values

# Create the line plot
plt.plot(x, y, marker='o', color='blue', label='Line Plot')

# Add labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Simple Line Plot')

# Add a legend
plt.legend()

# Show the plot
plt.grid(True)
plt.show()
