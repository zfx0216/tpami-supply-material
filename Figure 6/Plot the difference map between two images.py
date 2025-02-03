"""
The main function of this code file is to generate a "difference_map" between two images.
First, you need to set "actual_image_absolute_path" to the absolute path of the original image.
Second, you need to set "tampered_image_absolute_path" to the absolute path of another tampered image.
Third, you need to set "save_file_path" to the specific location and name where you want to save the generated "difference_map".
Finally, execute this code file to obtain the "difference_map".
"""


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

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

actual_image_R_channel = actual_image_R_channel.astype(np.float64)
actual_image_G_channel = actual_image_G_channel.astype(np.float64)
actual_image_B_channel = actual_image_B_channel.astype(np.float64)

# absolute path of the tampered image
tampered_image_absolute_path = "..."
# Open the image file
image = Image.open(tampered_image_absolute_path)
# Resize the image to the specified size
image = image.resize((224, 224))
# Convert the image to a NumPy array
image_array = np.array(image)

# Based on the RGB display mode, extract the data of each channel
tampered_image_R_channel = image_array[:, :, 0]
tampered_image_G_channel = image_array[:, :, 1]
tampered_image_B_channel = image_array[:, :, 2]

tampered_image_R_channel = tampered_image_R_channel.astype(np.float64)
tampered_image_G_channel = tampered_image_G_channel.astype(np.float64)
tampered_image_B_channel = tampered_image_B_channel.astype(np.float64)

diff_red_channel = np.abs(tampered_image_R_channel - actual_image_R_channel)

diff_green_channel = np.abs(tampered_image_G_channel - actual_image_G_channel)

diff_blue_channel = np.abs(tampered_image_B_channel - actual_image_B_channel)

for i in range(224):
    for j in range(224):
        if diff_red_channel[i][j] == 0 and diff_green_channel[i][j] == 0 and diff_blue_channel[i][j] == 0:
           diff_red_channel[i][j] = 255 - diff_red_channel[i][j]
           diff_green_channel[i][j] = 255 - diff_green_channel[i][j]
           diff_blue_channel[i][j] = 255 - diff_blue_channel[i][j]

rgb_image = np.stack((diff_red_channel, diff_green_channel, diff_blue_channel), axis=-1)

plt.imshow(rgb_image.astype(np.uint8))
plt.axis('off')
save_file_path = "..."
plt.savefig(save_file_path, bbox_inches='tight', pad_inches=0)  # Save the image without extra whitespace
plt.show()
