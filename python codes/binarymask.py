import cv2
import numpy as np
import os

def remove_background(rgb_path, binary_mask_path, output_path):
    # Read RGB image
    rgb_image = cv2.imread(rgb_path)

    # Read binary mask
    binary_mask = cv2.imread(binary_mask_path, cv2.IMREAD_GRAYSCALE)

    # Create a 3-channel mask
    mask = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

    # Invert the binary mask (make the foreground white)
    inverted_mask = cv2.bitwise_not(mask)

    # Bitwise AND operation to keep the background
    result_image = cv2.bitwise_and(rgb_image, mask)

    # Save the result
    cv2.imwrite(output_path, result_image)

def process_images_in_folder(folder_path):
    # Get a list of files in the folder
    files = os.listdir(folder_path)

    # Iterate through each file in the folder
    for file in files:
        # Check if the file is an RGB image
        if file.lower().endswith(('.png', '.jpg', '.jpeg','.bmp')):
            rgb_path = os.path.join(folder_path, file)

            # Form the corresponding binary mask file path
            mask_filename = f"{os.path.splitext(file)[0]}_lesion.bmp"
            mask_path = os.path.join(folder_path, mask_filename)

            # Check if the binary mask file exists
            if os.path.exists(mask_path):
                # Create an output file path
                output_filename = f"result_{os.path.splitext(file)[0]}.bmp"
                output_path = os.path.join(folder_path, output_filename)

                # Remove background and save the result
                remove_background(rgb_path, mask_path, output_path)


if __name__ == "__main__":
    # Replace these paths with the actual paths to your images
    folder_path = r'D:\NUST\assign2\class_3\foreground_3'


    process_images_in_folder(folder_path)
