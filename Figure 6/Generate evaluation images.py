"""
The main function of this code file is to generate evaluation images for the "guid-Xmage" method's C1~C8 evaluations.
First, you need to set "actual_image_absolute_path" to the absolute path of the original image to be analyzed.
Second, you need to set "output_directory" to the absolute path of the folder where the final series of evaluation images will be saved.
Third, you need to set "pixel_weight_matrix_R_path", "pixel_weight_matrix_G_path", and "pixel_weight_matrix_B_path" to the absolute paths of the pixel weight matrices of the RGB channels corresponding to the original image to be analyzed.
Finally, execute this code file to obtain a series of evaluation images.
"""


from PIL import Image
import numpy as np
import os
import math

# Read the matrix from the document
def read_matrix(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        matrix = [list(map(float, line.split())) for line in lines]
    return np.array(matrix)

###########################################################################################################
# Import the absolute path of the actual image
actual_image_absolute_path = "..."
image = Image.open(actual_image_absolute_path)
image = image.resize((224, 224))
image_array = np.array(image)

# RGB three-channel matrix of the actual image
actual_image_R_channel = image_array[:, :, 0]
actual_image_G_channel = image_array[:, :, 1]
actual_image_B_channel = image_array[:, :, 2]

# the absolute path of the folder to save experimental images
output_directory = "..."

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

# Perform preprocessing operations on the RGB three-channel matrix of the original image
bin_actual_image_R_channel = ((actual_image_R_channel / 255) - 0.485) / 0.229
bin_actual_image_G_channel = ((actual_image_G_channel / 255) - 0.456) / 0.224
bin_actual_image_B_channel = ((actual_image_B_channel / 255) - 0.406) / 0.225

# adjustment ratio
ratio = 0.01
###########################################################################################################
# slightly increase the value of x
bin_actual_image_R_channel_increase = bin_actual_image_R_channel.copy()
bin_actual_image_G_channel_increase = bin_actual_image_G_channel.copy()
bin_actual_image_B_channel_increase = bin_actual_image_B_channel.copy()

addend = 0
for i in range(224):
    for j in range(224):

        addend = abs(bin_actual_image_R_channel[i][j] * ratio)
        bin_actual_image_R_channel_increase[i][j] = bin_actual_image_R_channel_increase[i][j] + addend

        addend = abs(bin_actual_image_G_channel[i][j] * ratio)
        bin_actual_image_G_channel_increase[i][j] = bin_actual_image_G_channel_increase[i][j] + addend

        addend = abs(bin_actual_image_B_channel[i][j] * ratio)
        bin_actual_image_B_channel_increase[i][j] = bin_actual_image_B_channel_increase[i][j] + addend
###########################################################################################################
# slightly decrease the value of x
bin_actual_image_R_channel_decrease = bin_actual_image_R_channel.copy()
bin_actual_image_G_channel_decrease = bin_actual_image_G_channel.copy()
bin_actual_image_B_channel_decrease = bin_actual_image_B_channel.copy()

addend = 0
for i in range(224):
    for j in range(224):

        addend = abs(bin_actual_image_R_channel[i][j] * ratio)
        bin_actual_image_R_channel_decrease[i][j] = bin_actual_image_R_channel_decrease[i][j] - addend

        addend = abs(bin_actual_image_G_channel[i][j] * ratio)
        bin_actual_image_G_channel_decrease[i][j] = bin_actual_image_G_channel_decrease[i][j] - addend

        addend = abs(bin_actual_image_B_channel[i][j] * ratio)
        bin_actual_image_B_channel_decrease[i][j] = bin_actual_image_B_channel_decrease[i][j] - addend

###########################################################################################################
# C1 x>=0,k>=0
# slightly increase the value of x, leading to an increase in the classification value
change_matrix_R = actual_image_R_channel.copy()
change_matrix_G = actual_image_G_channel.copy()
change_matrix_B = actual_image_B_channel.copy()

for i in range(224):
    for j in range(224):

        if bin_actual_image_R_channel[i][j] >= 0 and pixel_weight_matrix_R[i][j] >= 0:
            if (bin_actual_image_R_channel_increase[i][j] * 0.229 + 0.485) > 1:
                change_matrix_R[i][j] = 255
            elif (bin_actual_image_R_channel_increase[i][j] * 0.229 + 0.485) <= 1:
                change_matrix_R[i][j] = math.ceil((bin_actual_image_R_channel_increase[i][j] * 0.229 + 0.485) * 255)

        if bin_actual_image_G_channel[i][j] >= 0 and pixel_weight_matrix_G[i][j] >= 0:
            if (bin_actual_image_G_channel_increase[i][j] * 0.224 + 0.456) > 1:
                change_matrix_G[i][j] = 255
            elif (bin_actual_image_G_channel_increase[i][j] * 0.224 + 0.456) <= 1:
                change_matrix_G[i][j] = math.ceil((bin_actual_image_G_channel_increase[i][j] * 0.224 + 0.456) * 255)

        if bin_actual_image_B_channel[i][j] >= 0 and pixel_weight_matrix_B[i][j] >= 0:
            if (bin_actual_image_B_channel_increase[i][j] * 0.225 + 0.406) > 1:
                change_matrix_B[i][j] = 255
            elif (bin_actual_image_B_channel_increase[i][j] * 0.225 + 0.406) <= 1:
                change_matrix_B[i][j] = math.ceil((bin_actual_image_B_channel_increase[i][j] * 0.225 + 0.406) * 255)

# Combine three channels into an RGB image
image_rgb = np.stack([change_matrix_R, change_matrix_G, change_matrix_B], axis=-1)
image_rgb = image_rgb.astype(np.uint8)
image_pil = Image.fromarray(image_rgb)
os.makedirs(output_directory, exist_ok=True)
image_pil.save(os.path.join(output_directory, f"forward_and_forward_increase_C1.png"))
###########################################################################################################
# C2 x>=0,k>=0
# slightly decrease the value of x, leading to an decrease in the classification value
change_matrix_R = actual_image_R_channel.copy()
change_matrix_G = actual_image_G_channel.copy()
change_matrix_B = actual_image_B_channel.copy()

for i in range(224):
    for j in range(224):

        if bin_actual_image_R_channel[i][j] >= 0 and pixel_weight_matrix_R[i][j] >= 0:
            if (bin_actual_image_R_channel_decrease[i][j] * 0.229 + 0.485) < 0:
                change_matrix_R[i][j] = 0
            elif (bin_actual_image_R_channel_decrease[i][j] * 0.229 + 0.485) >= 0:
                change_matrix_R[i][j] = int((bin_actual_image_R_channel_decrease[i][j] * 0.229 + 0.485) * 255)

        if bin_actual_image_G_channel[i][j] >= 0 and pixel_weight_matrix_G[i][j] >= 0:
            if (bin_actual_image_G_channel_decrease[i][j] * 0.224 + 0.456) < 0:
                change_matrix_G[i][j] = 0
            elif (bin_actual_image_G_channel_decrease[i][j] * 0.224 + 0.456) >= 0:
                change_matrix_G[i][j] = int((bin_actual_image_G_channel_decrease[i][j] * 0.224 + 0.456) * 255)

        if bin_actual_image_B_channel[i][j] >= 0 and pixel_weight_matrix_B[i][j] >= 0:
            if (bin_actual_image_B_channel_decrease[i][j] * 0.225 + 0.406) < 0:
                change_matrix_B[i][j] = 0
            elif (bin_actual_image_B_channel_decrease[i][j] * 0.225 + 0.406) >= 0:
                change_matrix_B[i][j] = int((bin_actual_image_B_channel_decrease[i][j] * 0.225 + 0.406) * 255)

# Combine three channels into an RGB image
image_rgb = np.stack([change_matrix_R, change_matrix_G, change_matrix_B], axis=-1)
image_rgb = image_rgb.astype(np.uint8)
image_pil = Image.fromarray(image_rgb)
os.makedirs(output_directory, exist_ok=True)
image_pil.save(os.path.join(output_directory, f"forward_and_forward_decrease_C2.png"))

###########################################################################################################
# C3 x<=0,k<=0
# slightly decrease the value of x, leading to an increase in the classification value
change_matrix_R = actual_image_R_channel.copy()
change_matrix_G = actual_image_G_channel.copy()
change_matrix_B = actual_image_B_channel.copy()
for i in range(224):
    for j in range(224):

        if bin_actual_image_R_channel[i][j] <= 0 and pixel_weight_matrix_R[i][j] <= 0:
            if (bin_actual_image_R_channel_decrease[i][j] * 0.229 + 0.485) < 0:
                change_matrix_R[i][j] = 0
            elif (bin_actual_image_R_channel_decrease[i][j] * 0.229 + 0.485) >= 0:
                change_matrix_R[i][j] = int((bin_actual_image_R_channel_decrease[i][j] * 0.229 + 0.485) * 255)

        if bin_actual_image_G_channel[i][j] <= 0 and pixel_weight_matrix_G[i][j] <= 0:
            if (bin_actual_image_G_channel_decrease[i][j] * 0.224 + 0.456) < 0:
                change_matrix_G[i][j] = 0
            elif (bin_actual_image_G_channel_decrease[i][j] * 0.224 + 0.456) >= 0:
                change_matrix_G[i][j] = int((bin_actual_image_G_channel_decrease[i][j] * 0.224 + 0.456) * 255)

        if bin_actual_image_B_channel[i][j] <= 0 and pixel_weight_matrix_B[i][j] <= 0:
            if (bin_actual_image_B_channel_decrease[i][j] * 0.225 + 0.406) < 0:
                change_matrix_B[i][j] = 0
            elif (bin_actual_image_B_channel_decrease[i][j] * 0.225 + 0.406) >= 0:
                change_matrix_B[i][j] = int((bin_actual_image_B_channel_decrease[i][j] * 0.225 + 0.406) * 255)

# Combine three channels into an RGB image
image_rgb = np.stack([change_matrix_R, change_matrix_G, change_matrix_B], axis=-1)
image_rgb = image_rgb.astype(np.uint8)
image_pil = Image.fromarray(image_rgb)
os.makedirs(output_directory, exist_ok=True)
image_pil.save(os.path.join(output_directory, f"reverse_and_reverse_increase_C3.png"))
###########################################################################################################
# C4 x<=0,k<=0
# slightly increase the value of x, leading to an decrease in the classification value
change_matrix_R = actual_image_R_channel.copy()
change_matrix_G = actual_image_G_channel.copy()
change_matrix_B = actual_image_B_channel.copy()
for i in range(224):
    for j in range(224):

        if bin_actual_image_R_channel[i][j] <= 0 and pixel_weight_matrix_R[i][j] <= 0:
            if (bin_actual_image_R_channel_increase[i][j] * 0.229 + 0.485) > 1:
                change_matrix_R[i][j] = 255
            elif (bin_actual_image_R_channel_increase[i][j] * 0.229 + 0.485) <= 1:
                change_matrix_R[i][j] = math.ceil((bin_actual_image_R_channel_increase[i][j] * 0.229 + 0.485) * 255)

        if bin_actual_image_G_channel[i][j] <= 0 and pixel_weight_matrix_G[i][j] <= 0:
            if (bin_actual_image_G_channel_increase[i][j] * 0.224 + 0.456) > 1:
                change_matrix_G[i][j] = 255
            elif (bin_actual_image_G_channel_increase[i][j] * 0.224 + 0.456) <= 1:
                change_matrix_G[i][j] = math.ceil((bin_actual_image_G_channel_increase[i][j] * 0.224 + 0.456) * 255)

        if bin_actual_image_B_channel[i][j] <= 0 and pixel_weight_matrix_B[i][j] <= 0:
            if (bin_actual_image_B_channel_increase[i][j] * 0.225 + 0.406) > 1:
                change_matrix_B[i][j] = 255
            elif (bin_actual_image_B_channel_increase[i][j] * 0.225 + 0.406) <= 1:
                change_matrix_B[i][j] = math.ceil((bin_actual_image_B_channel_increase[i][j] * 0.225 + 0.406) * 255)

# Combine three channels into an RGB image
image_rgb = np.stack([change_matrix_R, change_matrix_G, change_matrix_B], axis=-1)
image_rgb = image_rgb.astype(np.uint8)
image_pil = Image.fromarray(image_rgb)
os.makedirs(output_directory, exist_ok=True)
image_pil.save(os.path.join(output_directory, f"reverse_and_reverse_decrease_C4.png"))
###########################################################################################################
# C5 x>=0,k<=0
# slightly increase the value of x, leading to an decrease in the classification value
change_matrix_R = actual_image_R_channel.copy()
change_matrix_G = actual_image_G_channel.copy()
change_matrix_B = actual_image_B_channel.copy()
for i in range(224):
    for j in range(224):

        if bin_actual_image_R_channel[i][j] >= 0 and pixel_weight_matrix_R[i][j] <= 0:
            if (bin_actual_image_R_channel_increase[i][j] * 0.229 + 0.485) > 1:
                change_matrix_R[i][j] = 255
            elif (bin_actual_image_R_channel_increase[i][j] * 0.229 + 0.485) <= 1:
                change_matrix_R[i][j] = math.ceil((bin_actual_image_R_channel_increase[i][j] * 0.229 + 0.485) * 255)

        if bin_actual_image_G_channel[i][j] >= 0 and pixel_weight_matrix_G[i][j] <= 0:
            if (bin_actual_image_G_channel_increase[i][j] * 0.224 + 0.456) > 1:
                change_matrix_G[i][j] = 255
            elif (bin_actual_image_G_channel_increase[i][j] * 0.224 + 0.456) <= 1:
                change_matrix_G[i][j] = math.ceil((bin_actual_image_G_channel_increase[i][j] * 0.224 + 0.456) * 255)

        if bin_actual_image_B_channel[i][j] >= 0 and pixel_weight_matrix_B[i][j] <= 0:
            if (bin_actual_image_B_channel_increase[i][j] * 0.225 + 0.406) > 1:
                change_matrix_B[i][j] = 255
            elif (bin_actual_image_B_channel_increase[i][j] * 0.225 + 0.406) <= 1:
                change_matrix_B[i][j] = math.ceil((bin_actual_image_B_channel_increase[i][j] * 0.225 + 0.406) * 255)

# Combine three channels into an RGB image
image_rgb = np.stack([change_matrix_R, change_matrix_G, change_matrix_B], axis=-1)
image_rgb = image_rgb.astype(np.uint8)
image_pil = Image.fromarray(image_rgb)
os.makedirs(output_directory, exist_ok=True)
image_pil.save(os.path.join(output_directory, f"forward_and_reverse_decrease_C5.png"))
###########################################################################################################
# C6 x>=0,k<=0
# slightly decrease the value of x, leading to an increase in the classification value
change_matrix_R = actual_image_R_channel.copy()
change_matrix_G = actual_image_G_channel.copy()
change_matrix_B = actual_image_B_channel.copy()
for i in range(224):
    for j in range(224):

        if bin_actual_image_R_channel[i][j] >= 0 and pixel_weight_matrix_R[i][j] <= 0:
            if (bin_actual_image_R_channel_decrease[i][j] * 0.229 + 0.485) < 0:
                change_matrix_R[i][j] = 0
            elif (bin_actual_image_R_channel_decrease[i][j] * 0.229 + 0.485) >= 0:
                change_matrix_R[i][j] = int((bin_actual_image_R_channel_decrease[i][j] * 0.229 + 0.485) * 255)

        if bin_actual_image_G_channel[i][j] >= 0 and pixel_weight_matrix_G[i][j] <= 0:
            if (bin_actual_image_G_channel_decrease[i][j] * 0.224 + 0.456) < 0:
                change_matrix_G[i][j] = 0
            elif (bin_actual_image_G_channel_decrease[i][j] * 0.224 + 0.456) >= 0:
                change_matrix_G[i][j] = int((bin_actual_image_G_channel_decrease[i][j] * 0.224 + 0.456) * 255)

        if bin_actual_image_B_channel[i][j] >= 0 and pixel_weight_matrix_B[i][j] <= 0:
            if (bin_actual_image_B_channel_decrease[i][j] * 0.225 + 0.406) < 0:
                change_matrix_B[i][j] = 0
            elif (bin_actual_image_B_channel_decrease[i][j] * 0.225 + 0.406) >= 0:
                change_matrix_B[i][j] = int((bin_actual_image_B_channel_decrease[i][j] * 0.225 + 0.406) * 255)

# Combine three channels into an RGB image
image_rgb = np.stack([change_matrix_R, change_matrix_G, change_matrix_B], axis=-1)
image_rgb = image_rgb.astype(np.uint8)
image_pil = Image.fromarray(image_rgb)
os.makedirs(output_directory, exist_ok=True)
image_pil.save(os.path.join(output_directory, f"forward_and_reverse_increase_C6.png"))
###########################################################################################################
# C7 x<=0,k>=0
# slightly decrease the value of x, leading to an decrease in the classification value
change_matrix_R = actual_image_R_channel.copy()
change_matrix_G = actual_image_G_channel.copy()
change_matrix_B = actual_image_B_channel.copy()
for i in range(224):
    for j in range(224):

        if bin_actual_image_R_channel[i][j] <= 0 and pixel_weight_matrix_R[i][j] >= 0:
            if (bin_actual_image_R_channel_decrease[i][j] * 0.229 + 0.485) < 0:
                change_matrix_R[i][j] = 0
            elif (bin_actual_image_R_channel_decrease[i][j] * 0.229 + 0.485) >= 0:
                change_matrix_R[i][j] = int((bin_actual_image_R_channel_decrease[i][j] * 0.229 + 0.485) * 255)

        if bin_actual_image_G_channel[i][j] <= 0 and pixel_weight_matrix_G[i][j] >= 0:
            if (bin_actual_image_G_channel_decrease[i][j] * 0.224 + 0.456) < 0:
                change_matrix_G[i][j] = 0
            elif (bin_actual_image_G_channel_decrease[i][j] * 0.224 + 0.456) >= 0:
                change_matrix_G[i][j] = int((bin_actual_image_G_channel_decrease[i][j] * 0.224 + 0.456) * 255)

        if bin_actual_image_B_channel[i][j] <= 0 and pixel_weight_matrix_B[i][j] >= 0:
            if (bin_actual_image_B_channel_decrease[i][j] * 0.225 + 0.406) < 0:
                change_matrix_B[i][j] = 0
            elif (bin_actual_image_B_channel_decrease[i][j] * 0.225 + 0.406) >= 0:
                change_matrix_B[i][j] = int((bin_actual_image_B_channel_decrease[i][j] * 0.225 + 0.406) * 255)


# Combine three channels into an RGB image
image_rgb = np.stack([change_matrix_R, change_matrix_G, change_matrix_B], axis=-1)
image_rgb = image_rgb.astype(np.uint8)
image_pil = Image.fromarray(image_rgb)
os.makedirs(output_directory, exist_ok=True)
image_pil.save(os.path.join(output_directory, f"reverse_and_forward_decrease_C7.png"))
###########################################################################################################
# C8 x<=0,k>=0
# slightly increase the value of x, leading to an increase in the classification value
change_matrix_R = actual_image_R_channel.copy()
change_matrix_G = actual_image_G_channel.copy()
change_matrix_B = actual_image_B_channel.copy()
for i in range(224):
    for j in range(224):

        if bin_actual_image_R_channel[i][j] <= 0 and pixel_weight_matrix_R[i][j] >= 0:
            if (bin_actual_image_R_channel_increase[i][j] * 0.229 + 0.485) > 1:
                change_matrix_R[i][j] = 255
            elif (bin_actual_image_R_channel_increase[i][j] * 0.229 + 0.485) <= 1:
                change_matrix_R[i][j] = math.ceil((bin_actual_image_R_channel_increase[i][j] * 0.229 + 0.485) * 255)

        if bin_actual_image_G_channel[i][j] <= 0 and pixel_weight_matrix_G[i][j] >= 0:
            if (bin_actual_image_G_channel_increase[i][j] * 0.224 + 0.456) > 1:
                change_matrix_G[i][j] = 255
            elif (bin_actual_image_G_channel_increase[i][j] * 0.224 + 0.456) <= 1:
                change_matrix_G[i][j] = math.ceil((bin_actual_image_G_channel_increase[i][j] * 0.224 + 0.456) * 255)

        if bin_actual_image_B_channel[i][j] <= 0 and pixel_weight_matrix_B[i][j] >= 0:
            if (bin_actual_image_B_channel_increase[i][j] * 0.225 + 0.406) > 1:
                change_matrix_B[i][j] = 255
            elif (bin_actual_image_B_channel_increase[i][j] * 0.225 + 0.406) <= 1:
                change_matrix_B[i][j] = math.ceil((bin_actual_image_B_channel_increase[i][j] * 0.225 + 0.406) * 255)

# Combine three channels into an RGB image
image_rgb = np.stack([change_matrix_R, change_matrix_G, change_matrix_B], axis=-1)
image_rgb = image_rgb.astype(np.uint8)
image_pil = Image.fromarray(image_rgb)
os.makedirs(output_directory, exist_ok=True)
image_pil.save(os.path.join(output_directory, f"reverse_and_forward_increase_C8.png"))