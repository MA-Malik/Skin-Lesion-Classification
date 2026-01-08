import cv2
import os
import pandas as pd
import numpy as np

# Specify the input and output folders
input_folder = r"D:\NUST\assign2\class_1\only_fg_1"
output_folder = r"D:\NUST\assign2\class_1\grey_1"
new_csv = r"D:\NUST\assign2\class_1\rgb_hist_1.csv"

# List all the files in the input folder
files = os.listdir(input_folder)

# Initialize a list to store the new results
results = []

# Loop through the files, convert them to grayscale and RGB, and calculate statistics
for file in files:
    if file.endswith(('.jpg', '.png', '.jpeg', '.bmp')):
        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, file)

        # Open the image using OpenCV
        image = cv2.imread(input_path)

        # Calculate the histogram mean, standard deviation, and variance for each RGB channel
        histogram_means_rgb = np.mean(image, axis=(0, 1))
        histogram_stddevs_rgb = np.std(image, axis=(0, 1))
        histogram_variances_rgb = np.var(image, axis=(0, 1))

        # Store the new results in a list
        results.append({
            'File': file,
            'Hist_Mean_R': histogram_means_rgb[2],
            'Hist_StdDev_R': histogram_stddevs_rgb[2],
            'Hist_Mean_G': histogram_means_rgb[1],
            'Hist_StdDev_G': histogram_stddevs_rgb[1],
            'Hist_Mean_B': histogram_means_rgb[0],
            'Hist_StdDev_B': histogram_stddevs_rgb[0],
        })

# Load the existing CSV file if it exists, or create a new DataFrame if it doesn't
df = pd.DataFrame(results)
df.to_csv(new_csv, index=False)

print(f"Conversion and calculation completed. Results saved to {new_csv}")
