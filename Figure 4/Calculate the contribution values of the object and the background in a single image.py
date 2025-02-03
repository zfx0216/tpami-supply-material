"""
The main function of this code file is to calculate the contribution values of the marked object regions and the background regions in an image.
First, you need to set "actual_image_absolute_path" to the absolute path of the image to be calculated.
Second, you need to set "read_file_path_mark" to the absolute path of the image used for region marking. In this case, we use red to mark the regions of dogs and blue to mark the regions of cats, while the regions not marked are considered background.
Third, you need to set "pixel_weight_matrix_R_path", "pixel_weight_matrix_G_path", and "pixel_weight_matrix_B_path" to the absolute paths of the pixel weight matrix files corresponding to the RGB channels of the image to be calculated.
Finally, executing this code file will output the contribution values of different regions.
"""


from PIL import Image
import numpy as np

# Read the matrix from the document
def read_matrix(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        matrix = [list(map(float, line.split())) for line in lines]
    return np.array(matrix)

# Import the absolute path of the actual image
actual_image_absolute_path = "..."
image = Image.open(actual_image_absolute_path)
image = image.resize((224, 224))
image_array = np.array(image)

# RGB three-channel matrix of the actual image
actual_image_R_channel = image_array[:, :, 0]
actual_image_G_channel = image_array[:, :, 1]
actual_image_B_channel = image_array[:, :, 2]

# Import an image with specific regions marked
read_file_path_mark = "..."
image = Image.open(read_file_path_mark)
image = image.resize((224, 224))
image_array = np.array(image)

# RGB three-channel matrix of the image with specific regions marked
image_R_channel_mark = image_array[:, :, 0]
image_G_channel_mark = image_array[:, :, 1]
image_B_channel_mark = image_array[:, :, 2]

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

# Calculate the total pixel classification contribution matrix
pixel_classification_contribution_matrix_total = pixel_classification_contribution_matrix_R + pixel_classification_contribution_matrix_G + pixel_classification_contribution_matrix_B

# Calculate the forward and backward contribution values for specific regions
pixel_classification_contribution_red_forward = 0
pixel_classification_contribution_red_reverse = 0

pixel_classification_contribution_blue_forward = 0
pixel_classification_contribution_blue_reverse = 0

pixel_classification_contribution_bacground_forward = 0
pixel_classification_contribution_bacground_reverse = 0


"""
Calculate the pixel classification contribution values for the red-marked region, 
where the region is defined as having R channel pixel values greater than or equal to 150, 
G channel pixel values less than or equal to 50, 
and B channel pixel values less than or equal to 50
"""
for i in range(224):
    for j in range(224):
        if image_R_channel_mark[i][j] >= 150 and image_G_channel_mark[i][j] <= 50 and image_B_channel_mark[i][j] <= 50:
            if pixel_classification_contribution_matrix_total[i][j] >= 0:
                pixel_classification_contribution_red_forward = pixel_classification_contribution_red_forward + pixel_classification_contribution_matrix_total[i][j]
            if pixel_classification_contribution_matrix_total[i][j] < 0:
                pixel_classification_contribution_red_reverse = pixel_classification_contribution_red_reverse + pixel_classification_contribution_matrix_total[i][j]

"""
Calculate the pixel classification contribution values for the blue-marked region, 
where the region is defined as having R channel pixel values less than or equal to 50, 
G channel pixel values less than or equal to 50, 
and B channel pixel values greater than or equal to 150
"""
for i in range(224):
    for j in range(224):
        if image_R_channel_mark[i][j] <= 50 and image_G_channel_mark[i][j] <= 50 and image_B_channel_mark[i][j] >= 150:
            if pixel_classification_contribution_matrix_total[i][j] >= 0:
                pixel_classification_contribution_blue_forward = pixel_classification_contribution_blue_forward + pixel_classification_contribution_matrix_total[i][j]
            if pixel_classification_contribution_matrix_total[i][j] < 0:
                pixel_classification_contribution_blue_reverse = pixel_classification_contribution_blue_reverse + pixel_classification_contribution_matrix_total[i][j]

"""
Calculate the pixel classification contribution values for the background region, 
defined as the region not marked by either red or blue
"""
for i in range(224):
    for j in range(224):
        if ~(image_R_channel_mark[i][j] >= 150 and image_G_channel_mark[i][j] <= 50 and image_B_channel_mark[i][j] <= 50):
            if ~(image_R_channel_mark[i][j] <= 50 and image_G_channel_mark[i][j] <= 50 and image_B_channel_mark[i][j] >= 150):
                if pixel_classification_contribution_matrix_total[i][j] >= 0:
                    pixel_classification_contribution_bacground_forward = pixel_classification_contribution_bacground_forward + pixel_classification_contribution_matrix_total[i][j]
                if pixel_classification_contribution_matrix_total[i][j] < 0:
                    pixel_classification_contribution_bacground_reverse = pixel_classification_contribution_bacground_reverse + pixel_classification_contribution_matrix_total[i][j]

print("Total forward contribution value of the red-marked region：")
print(pixel_classification_contribution_red_forward)
print("Total reverse contribution value of the red-marked region：")
print(pixel_classification_contribution_red_reverse)

print("Total forward contribution value of the blue-marked region：")
print(pixel_classification_contribution_blue_forward)
print("Total reverse contribution value of the blue-marked region：")
print(pixel_classification_contribution_blue_reverse)

print("Total forward contribution value of the bacground region：")
print(pixel_classification_contribution_bacground_forward)
print("Total reverse contribution value of the bacground region：")
print(pixel_classification_contribution_bacground_reverse)

"""
Validate that the sum of contribution values and bias for each region equals the actual classification value of the original image in the model, 
where the bias value for the original image is 36.26948928833008
"""
bias = "..."
print(pixel_classification_contribution_red_forward + pixel_classification_contribution_red_reverse + pixel_classification_contribution_blue_forward + pixel_classification_contribution_blue_reverse + pixel_classification_contribution_bacground_forward + pixel_classification_contribution_bacground_reverse + bias)