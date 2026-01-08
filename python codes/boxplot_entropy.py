import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

# Function to calculate RGB histogram
def calculate_rgb_histogram(image):
    histogram = []
    for channel in range(3):  # RGB channels
        hist_channel = cv2.calcHist([image], [channel], None, [256], [0, 256])
        histogram.append(hist_channel)
    return histogram

# Function to calculate mean entropy across channels
def calculate_mean_entropy(image):
    hist = calculate_rgb_histogram(image)
    total_pixels = image.shape[0] * image.shape[1]

    # Normalize histograms
    hist_normalized = [channel / total_pixels for channel in hist]

    # Flatten and calculate entropy for each channel
    entropies = [entropy(np.concatenate(channel)) for channel in hist_normalized]

    # Calculate mean entropy
    mean_entropy = np.mean(entropies)

    return mean_entropy

# Process images in three folders
folder_paths = [r'D:\NUST\assign2\class_1\only_fg_1', r'D:\NUST\assign2\class_2\only_fg_2', r'D:\NUST\assign2\class_3\only_fg_3']

all_mean_entropy_values = []

for folder_path in folder_paths:
    mean_entropy_values = []

    # Loop through all files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(('.bmp', '.jpeg', '.png')):  # Filter only image files
            # Read image
            image_path = os.path.join(folder_path, file_name)
            img = cv2.imread(image_path)

            # Calculate mean entropy
            mean_entropy_val = calculate_mean_entropy(img)
            mean_entropy_values.append(mean_entropy_val)

    all_mean_entropy_values.append(mean_entropy_values)

# Plot boxplots for each folder
plt.boxplot(all_mean_entropy_values, labels=['Class 1', 'Class 2', 'Class 3'])
plt.title('Mean Entropy of RGB Histogram for Images in Each Class')
plt.xlabel('Class')
plt.ylabel('Mean Entropy')
plt.show()
