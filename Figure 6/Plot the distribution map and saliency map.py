"""
The main function of this code file is to draw the distribution_map and saliency_map for a single image.
First, you need to set "actual_image_absolute_path" to the absolute path of the original image to be drawn.
Second, you need to set "pixel_weight_matrix_R_path", "pixel_weight_matrix_G_path", and "pixel_weight_matrix_B_path" to the absolute paths of the pixel weight matrix files corresponding to the RGB channels of the original image.
Finally, executing this code file will complete the drawing, and the distribution_map and saliency_map will be saved in the current directory.
"""

from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import matplotlib.pyplot as plt

# Read the matrix from the document
def read_matrix(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        matrix = [list(map(float, line.split())) for line in lines]
    return np.array(matrix)

# Define color mapping
def custom_colormap(value , zero):
    blue = np.array([0, 0, 255])
    white = np.array([255, 255, 255])
    red = np.array([255, 0, 0])

    if value < zero:
        color = white - ((zero - value) / zero) * (white - blue)
        color[0] = 0
    elif value > zero:
        color = white - (value - zero) / (255 - zero) * (white - red)
        color[2] = 0
    elif value == zero:
        color = np.array([0, 255, 0])

    return color.astype(np.uint8)

###########################################################################################################
# Import the absolute path of the actual image
actual_image_absolute_path = "..."
image = Image.open(actual_image_absolute_path)
image = image.resize((224, 224))
image_array_actual = np.array(image)

# RGB three-channel matrix of the actual image
actual_image_R_channel = image_array_actual[:, :, 0]
actual_image_G_channel = image_array_actual[:, :, 1]
actual_image_B_channel = image_array_actual[:, :, 2]

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

# ###################################################################################################
matrix = pixel_classification_contribution_matrix_total

sorted_indices = np.argsort(matrix, axis=None)

sorted_values = np.sort(matrix, axis=None)

label_one = "{:.4f}".format(sorted_values[0])
label_two = "{:.4f}".format(sorted_values[10035])
label_three = "{:.4f}".format(sorted_values[20070])
label_four = "{:.4f}".format(sorted_values[30105])
label_five = "{:.4f}".format(sorted_values[40140])
label_six = "{:.4f}".format(sorted_values[50175])

# find the zero point
level = 100
for i in range(224):
    for j in range(224):
        if abs(pixel_classification_contribution_matrix_total[i][j] - 0) < level:
            level = pixel_classification_contribution_matrix_total[i][j]

# min-max normalization
min_value = np.min(pixel_classification_contribution_matrix_total)
pixel_classification_contribution_matrix_total = pixel_classification_contribution_matrix_total - min_value
level = level - min_value

max_value = np.max(pixel_classification_contribution_matrix_total)
pixel_classification_contribution_matrix_total = pixel_classification_contribution_matrix_total / max_value
level = level / max_value

#  map back to [0,255]
pixel_classification_contribution_matrix_total = pixel_classification_contribution_matrix_total * 255
level = level * 255

####################################################################################################
# pixel classification contribution value distribution map
heatmap = pixel_classification_contribution_matrix_total

heatmap_color = np.zeros((224, 224, 3), dtype=np.uint8)
for i in range(224):
    for j in range(224):
        value = heatmap[i, j]
        heatmap_color[i, j] = custom_colormap(value, level)
#################################################################################
cmap_name = 'custom_colormap'
heatmap_colorbar = pixel_classification_contribution_matrix_total

sorted_values = np.sort(heatmap_colorbar, axis=None)
colors = [custom_colormap(i, zero=128) for i in range(256)]
colors = np.array(colors) / 255.0
custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors)

fig, ax = plt.subplots()
cbar = ax.imshow(heatmap_color, cmap=custom_cmap)
cb = fig.colorbar(cbar, ax=ax, ticks=[0, 128, 255])

tick_labels = [label_one, '0', label_six]
cb.set_ticklabels(tick_labels)
plt.xticks([])
plt.yticks([])
plt.savefig("distribution_map.png", bbox_inches='tight', pad_inches=0)
plt.show()

############################################################################
# saliency map
sorted_array = np.sort(pixel_classification_contribution_matrix_total.flatten())

# mark the position of the zero point
zero_locatin = 0
for i in range(50176):
    if sorted_array[i] == level:
        zero_locatin = i

red_location = int((zero_locatin + 50176) / 2)
green_location = int((zero_locatin + 0) / 2)

heatmap_shape = (224, 224, 3)
zero_array = np.zeros(heatmap_shape)

for i in range(224):
    for j in range(224):
        # blue
        if pixel_classification_contribution_matrix_total[i][j] < sorted_array[green_location]:
            zero_array[i][j][0] = 0
            zero_array[i][j][1] = 0
            zero_array[i][j][2] = 1
        # turquoise
        elif pixel_classification_contribution_matrix_total[i][j] >= sorted_array[green_location] and pixel_classification_contribution_matrix_total[i][j] < sorted_array[zero_locatin]:
            zero_array[i][j][0] = 0
            zero_array[i][j][1] = 1
            zero_array[i][j][2] = 1
        # yellow
        elif pixel_classification_contribution_matrix_total[i][j] >= sorted_array[zero_locatin] and pixel_classification_contribution_matrix_total[i][j] < sorted_array[red_location]:
            zero_array[i][j][0] = 1
            zero_array[i][j][1] = 1
            zero_array[i][j][2] = 0
        # red
        elif pixel_classification_contribution_matrix_total[i][j] >= sorted_array[red_location]:
            zero_array[i][j][0] = 1
            zero_array[i][j][1] = 0
            zero_array[i][j][2] = 0

red_channel = zero_array[:, :, 0]
green_channel = zero_array[:, :, 1]
blue_channel = zero_array[:, :, 2]

rgb_image = np.stack([red_channel, green_channel, blue_channel], axis=-1)

heatmap_image = Image.fromarray((rgb_image * 255).astype(np.uint8))

save_path = "saliency_map.png"
heatmap_image.save(save_path)

heatmap_image = Image.open(save_path)

colors = [(0, 0, 1), (0, 1, 1), (1, 1, 0), (1, 0, 0)]
custom_cmap = LinearSegmentedColormap.from_list("custom_colormap", colors, N=4)

fig, ax = plt.subplots()
cb = fig.colorbar(plt.cm.ScalarMappable(cmap=custom_cmap), ax=ax, ticks=[0,0.5,1])

tick_labels = [label_one, '0',label_six]
cb.set_ticklabels(tick_labels)

plt.xticks([])
plt.yticks([])

ax.imshow(np.array(heatmap_image))
plt.savefig("saliency_map.png", bbox_inches='tight', pad_inches=0)
plt.show()