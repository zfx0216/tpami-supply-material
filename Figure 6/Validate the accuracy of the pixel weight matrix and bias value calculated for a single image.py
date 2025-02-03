"""
The main function of this file is to validate the accuracy of the computed pixel weight matrices and bias values for a single image.
First, you need to set "image_absolute_path" to the absolute path of the image to be calculated.
Second, you need to set "pixel_weight_matrix_R_path", "pixel_weight_matrix_G_path", and "pixel_weight_matrix_B_path" to the absolute paths of the pixel weight matrix files corresponding to the RGB channels of this image.
Third, you need to set "bias" to the bias value corresponding to this image.
Finally, executing this code file will output the computed overall contribution value. If the overall contribution value matches or is close to the classification value computed by AlexNet for the image (close due to computer calculation precision, but the difference will not exceed 0.1), it indicates that the computation is correct.
"""


from PIL import Image
import numpy as np

# # Read the matrix from the document
def read_matrix(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        matrix = [list(map(float, line.split())) for line in lines]
    return np.array(matrix)

# Open the image and perform preprocessing
image_absolute_path = "..."
# Open the image file
image = Image.open(image_absolute_path)
# Resize the image to the specified size
image = image.resize((224, 224))
# Convert the image to a NumPy array
image_array = np.array(image)

# Based on the RGB display mode, extract the data of each channel
actual_image_R_channel = image_array[:, :, 0]
actual_image_G_channel = image_array[:, :, 1]
actual_image_B_channel = image_array[:, :, 2]

# Read the pixel weight matrix of the RGB three-channel of the actual image
pixel_weight_matrix_R_path = r"..."
pixel_weight_matrix_G_path = r"..."
pixel_weight_matrix_B_path = r"..."

pixel_weight_matrix_R = read_matrix(pixel_weight_matrix_R_path)
pixel_weight_matrix_G = read_matrix(pixel_weight_matrix_G_path)
pixel_weight_matrix_B = read_matrix(pixel_weight_matrix_B_path)

# Calculate the pixel classification contribution matrix for the RGB three channels
pixel_classification_contribution_matrix_R = ((actual_image_R_channel / 255) - 0.485) / 0.229 * pixel_weight_matrix_R
pixel_classification_contribution_matrix_G = ((actual_image_G_channel / 255) - 0.456) / 0.224 * pixel_weight_matrix_G
pixel_classification_contribution_matrix_B = ((actual_image_B_channel / 255) - 0.406) / 0.225 * pixel_weight_matrix_B

# Import bias value.
bias = 16.50979995727539

# Calculate total contribution value.
total_contribution_value = np.sum(pixel_classification_contribution_matrix_R) + np.sum(pixel_classification_contribution_matrix_G) + np.sum(pixel_classification_contribution_matrix_B) + bias

print("Calculate total contribution value.: ", total_contribution_value)