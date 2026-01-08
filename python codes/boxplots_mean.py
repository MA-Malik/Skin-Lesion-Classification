import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_mean_green_channel(folder_path):
    mean_green_values = []

    # Get a list of all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    # Loop through each image and calculate mean values for the green channel
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        img = cv2.imread(image_path)

        # Calculate mean value for the green channel
        mean_green_values.append(np.mean(img[:, :, 2]))  # Green channel

    return np.array(mean_green_values)

def plot_green_channel_box_plot(data, folder_names):
    plt.figure(figsize=(8, 6))
    plt.boxplot(data, labels=folder_names)
    plt.title('Box Plot of Mean Blue Channel Across Classes')
    plt.ylabel('Mean Blue Value')
    plt.show()

# Paths to the folders containing images
folder1_path = r"D:\NUST\assign2\class_1\only_fg_1"
folder2_path = r"D:\NUST\assign2\class_2\only_fg_2"
folder3_path = r"D:\NUST\assign2\class_3\only_fg_3"

# Calculate mean green values for each folder
mean_green_values1 = calculate_mean_green_channel(folder1_path)
mean_green_values2 = calculate_mean_green_channel(folder2_path)
mean_green_values3 = calculate_mean_green_channel(folder3_path)

# Extract mean green values for each folder
mean_green_values = [mean_green_values1, mean_green_values2, mean_green_values3]

# Plot box plot for mean green values
plot_green_channel_box_plot(mean_green_values, ['Class 1', 'Class 2', 'Class 3'])
