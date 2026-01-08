import numpy as np
import matplotlib.pyplot as plt

# Define means and standard deviations
means = [36.9, 41.9, 83.5]  # Replace mu1, mu2, mu3 with your actual mean values
std_devs = [65, 68, 64]  # Replace sigma1, sigma2, sigma3 with your actual standard deviation values

# Generate data points for the x-axis
x = np.linspace(min(means) - 3 * max(std_devs), max(means) + 3 * max(std_devs), 1000)

# Plot Gaussian distributions with different colors
colors = ['red', 'blue', 'green']
for mean, std_dev, color in zip(means, std_devs, colors):
    y = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)
    plt.plot(x, y, label=f'Mean={mean}, Std Dev={std_dev}', color=color)



# Add labels and legend
plt.title('Gaussian Distributions')
plt.xlabel('X-axis')
plt.ylabel('Probability Density')
plt.legend()
plt.show()