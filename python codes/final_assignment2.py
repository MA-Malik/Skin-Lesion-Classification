import cv2
import numpy as np
import os

# Folder paths containing images for each class
class_folders = {
    'Class1': r"D:\NUST\assign2\class_1\masked_test_1",
    'Class2': r"D:\NUST\assign2\class_2\masked_test_2",
    'Class3': r"D:\NUST\assign2\class_3\masked_test_3"
}

# Mean values and user-defined threshold values for each class (replace these with your actual values)
class_info = {
    'Class1': {'mean': [36.96, 23.3, 15.12], 'entropy': 2.08, 'lower_mean_threshold': [2.5, 3.3, 6], 'upper_mean_threshold': [39, 45, 80],
               'lower_entropy_threshold': 0.6, 'upper_entropy_threshold': 2.9},
    'Class2': {'mean': [42, 27.4, 20], 'entropy': 2.53, 'lower_mean_threshold': [3.3, 5, 10], 'upper_mean_threshold': [45, 47, 80],
               'lower_entropy_threshold': 0.9, 'upper_entropy_threshold': 3.3},
    'Class3': {'mean': [83.5, 55.8, 47], 'entropy': 4.9, 'lower_mean_threshold': [12, 15, 20], 'upper_mean_threshold': [100, 99, 150],
               'lower_entropy_threshold': 1.6, 'upper_entropy_threshold': 5}
}

# Initialize variables for accuracy calculation
total_images = 0
correct_assignments = 0
class_correct_assignments = {class_name: 0 for class_name in class_info}
unclassified_images = 0

# Loop through each class folder
for class_name, folder_path in class_folders.items():
    # List all files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.bmp', '.png', '.jpeg'))]

    # Loop through each image in the folder
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        img = cv2.imread(image_path)

        # Calculate mean for each channel
        mean_values = [np.mean(img[:, :, i]) for i in range(3)]

        # Calculate entropy for each channel
        hist_channels = [cv2.calcHist([img], [i], None, [256], [0, 256]) for i in range(3)]
        entropy_values = [0] * 3
        for i in range(3):
            hist_normalized = hist_channels[i].ravel() / hist_channels[i].sum()
            log_hist = np.log2(hist_normalized + 1e-10)
            entropy_values[i] = -np.sum(hist_normalized * log_hist)

        # Use Euclidean distance approach for unclassified images
        if not (class_info[class_name]['lower_mean_threshold'][0] <= mean_values[0] <= class_info[class_name]['upper_mean_threshold'][0] and
                class_info[class_name]['lower_mean_threshold'][1] <= mean_values[1] <= class_info[class_name]['upper_mean_threshold'][1] and
                class_info[class_name]['lower_mean_threshold'][2] <= mean_values[2] <= class_info[class_name]['upper_mean_threshold'][2] and
                class_info[class_name]['lower_entropy_threshold'] <= entropy_values[0] <= class_info[class_name]['upper_entropy_threshold'] and
                class_info[class_name]['lower_entropy_threshold'] <= entropy_values[1] <= class_info[class_name]['upper_entropy_threshold'] and
                class_info[class_name]['lower_entropy_threshold'] <= entropy_values[2] <= class_info[class_name]['upper_entropy_threshold']):

            distances = [np.linalg.norm(np.array(mean_values) - np.array(class_info[class_name]['mean'])) for class_name in class_info]
            assigned_class = list(class_info.keys())[np.argmin(distances)]
        else:
            assigned_class = class_name

        # Check if the assigned class matches the actual class
        if assigned_class == class_name:
            correct_assignments += 1
            class_correct_assignments[class_name] += 1

        total_images += 1

# Calculate accuracy for each class
class_accuracies = {class_name: (class_correct_assignments[class_name] / len(os.listdir(class_folders[class_name]))) * 100
                    for class_name in class_correct_assignments}

# Calculate overall accuracy if total_images is not zero
overall_accuracy = (correct_assignments / total_images) * 100 if total_images != 0 else 0

# Print results
print(f'Total Images: {total_images}, Correct Assignments: {correct_assignments}, Overall Accuracy: {overall_accuracy:.2f}%')
print(f'Unclassified Images: {unclassified_images}')

for class_name in class_accuracies:
    print(f'Class: {class_name}, Correct Assignments: {class_correct_assignments[class_name]}, Total Images: {len(os.listdir(class_folders[class_name]))}, Accuracy: {class_accuracies[class_name]:.2f}%')
