"""
The main function of this code file is to generate adversarial samples with the ability to weaken the contribution value of background classification
1. You need to set "actual_image_folder_path" to the absolute path of the folder where the original images are stored.
2. You need to set "mark_object_image_folder_path" to the absolute path of the folder where the images with marked object regions are stored.
3. You need to set "save_path_for_adversarial_samples" to the absolute path of the folder where the final generated adversarial samples will be saved.
4. You need to set "the_absolute_path_of_the_weight_file" to the absolute path of the model weight file.
5. You need to set "the_absolute_path_of_the_index_file" to the absolute path of the index file.
"""


import json
import shutil
import torch
from PIL import Image
import os
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from model import AlexNet


# Set the device based on CUDA availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Perform initialization operations on the images
data_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


def predict(image_path, index_path, weight_path):
    # Load image
    img = Image.open(image_path)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)

    with open(index_path, "r") as f:
        class_indict = json.load(f)

    # Create model
    model = AlexNet(num_classes=3).to(device)

    # Load model weights
    model.load_state_dict(torch.load(weight_path))

    # Set the model to evaluation mode
    model.eval()
    with torch.no_grad():
        # Predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        classification_probability = torch.softmax(output, dim=0)
    # Get the index of the class with the highest probability
    predicted_class_index = torch.argmax(classification_probability).item()

    return(predicted_class_index)


def pixel_weight_matrix(image_path, weight_path):
    img = Image.open(image_path)
    plt.imshow(img)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)

    # Create model
    model = AlexNet(num_classes=3).to(device)

    model.load_state_dict(torch.load(weight_path))

    # Set the model to evaluation mode
    model.eval()
    output = torch.squeeze(model(img.to(device))).cpu()
    classification_probability = torch.softmax(output, dim=0)

    top_probs, top_indices = torch.topk(classification_probability, 3)
    img = img.to(device)
    model.eval()
    img.requires_grad_()
    output = model(img)
    pred_score = output[0, top_indices[0]]
    pred_score.backward(retain_graph=True)
    gradients = img.grad

    channel_r = gradients[0, 0, :, :].cpu().detach().numpy()
    channel_g = gradients[0, 1, :, :].cpu().detach().numpy()
    channel_b = gradients[0, 2, :, :].cpu().detach().numpy()

    np.savetxt("iterative_sample_pixel_weight_matrix_top1_r_33_66.txt", channel_r, fmt="%.10f", delimiter=" ")
    np.savetxt("iterative_sample_pixel_weight_matrix_top1_g_33_66.txt", channel_g, fmt="%.10f", delimiter=" ")
    np.savetxt("iterative_sample_pixel_weight_matrix_top1_b_33_66.txt", channel_b, fmt="%.10f", delimiter=" ")

    return channel_r, channel_g, channel_b

def read_matrix(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        matrix = [list(map(float, line.split())) for line in lines]
    return np.array(matrix)

# Define the sorting function
def sort_func(file_name):
    # Extract the numerical part of the file name
    return int(''.join(filter(str.isdigit, file_name)))


# actual image folder path
actual_image_folder_path = "..."
# Mark object image folder path
mark_object_image_folder_path = "..."
# Save the folder path for the generated adversarial samples
save_path_for_adversarial_samples = "..."
# Load model weights
the_absolute_path_of_the_weight_file = "..."
# Read class_indict
the_absolute_path_of_the_index_file = "..."

image_files = sorted([f for f in os.listdir(actual_image_folder_path) if f.endswith('.png')], key=sort_func)
fail_images = []
for image_file in image_files:
    actual_image_absolute_path = os.path.join(actual_image_folder_path, image_file)
    mark_object_image_path = os.path.join(mark_object_image_folder_path, image_file)

    img = Image.open(actual_image_absolute_path)
    plt.imshow(img)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)

    image = Image.open(actual_image_absolute_path)
    image = image.resize((224, 224))
    iterative_image = np.array(image)

    actual_image_channel_R = iterative_image[:, :, 0]
    actual_image_channel_R = actual_image_channel_R.astype(np.float64)
    actual_image_channel_G = iterative_image[:, :, 1]
    actual_image_channel_G = actual_image_channel_G.astype(np.float64)
    actual_image_channel_B = iterative_image[:, :, 2]
    actual_image_channel_B = actual_image_channel_B.astype(np.float64)

    image = Image.open(mark_object_image_path)
    # Resize the image to the specified size
    image = image.resize((224, 224))
    # Convert the image to a NumPy array
    mark_object_image = np.array(image)
    # Based on the RGB display mode, extract the data of each channel
    mark_object_image_channel_R = mark_object_image[:, :, 0]
    mark_object_image_channel_G = mark_object_image[:, :, 1]
    mark_object_image_channel_B = mark_object_image[:, :, 2]

    with open(the_absolute_path_of_the_index_file, "r") as f:
        class_indict = json.load(f)

    actual_image_index = predict(actual_image_absolute_path, the_absolute_path_of_the_index_file, the_absolute_path_of_the_weight_file)
    ####################################################################################################################################
    flag = 0
    iterative_image_path = actual_image_absolute_path
    num_iterative = 0

    while flag == 0:

        num_iterative = num_iterative + 1
        pixel_weight_matrix(iterative_image_path, the_absolute_path_of_the_weight_file)

        image = Image.open(iterative_image_path)
        image = image.resize((224, 224))
        iterative_image = np.array(image)

        iterative_image_channel_R = iterative_image[:, :, 0]
        iterative_image_channel_R = iterative_image_channel_R.astype(np.float64)
        iterative_image_channel_G = iterative_image[:, :, 1]
        iterative_image_channel_G = iterative_image_channel_G.astype(np.float64)
        iterative_image_channel_B = iterative_image[:, :, 2]
        iterative_image_channel_B = iterative_image_channel_B.astype(np.float64)

        pixel_weight_matrix_R_iterative = "iterative_sample_pixel_weight_matrix_top1_r_33_66.txt"
        pixel_weight_matrix_G_iterative = "iterative_sample_pixel_weight_matrix_top1_g_33_66.txt"
        pixel_weight_matrix_B_iterative = "iterative_sample_pixel_weight_matrix_top1_b_33_66.txt"

        pixel_weight_matrix_R_iterative = read_matrix(pixel_weight_matrix_R_iterative)
        pixel_weight_matrix_G_iterative = read_matrix(pixel_weight_matrix_G_iterative)
        pixel_weight_matrix_B_iterative = read_matrix(pixel_weight_matrix_B_iterative)

        matrix_R_data_transform = ((iterative_image_channel_R / 255) - 0.485) / 0.229
        matrix_G_data_transform = ((iterative_image_channel_G / 255) - 0.456) / 0.224
        matrix_B_data_transform = ((iterative_image_channel_B / 255) - 0.406) / 0.225

        ratio = 0.01
        #############################################################################################
        matrix_R_increase = iterative_image_channel_R.copy()
        matrix_R_increase = matrix_R_increase.astype(np.float64)
        matrix_G_increase = iterative_image_channel_G.copy()
        matrix_G_increase = matrix_G_increase.astype(np.float64)
        matrix_B_increase = iterative_image_channel_B.copy()
        matrix_B_increase = matrix_B_increase.astype(np.float64)

        addend = 0
        for i in range(224):
            for j in range(224):
                addend = abs(matrix_R_increase[i][j] * ratio)
                matrix_R_increase[i][j] = matrix_R_increase[i][j] + addend
                np.clip(matrix_R_increase[i][j], 0, 255)

                addend = abs(matrix_G_increase[i][j] * ratio)
                matrix_G_increase[i][j] = matrix_G_increase[i][j] + addend
                np.clip(matrix_G_increase[i][j], 0, 255)

                addend = abs(matrix_B_increase[i][j] * ratio)
                matrix_B_increase[i][j] = matrix_B_increase[i][j] + addend
                np.clip(matrix_B_increase[i][j], 0, 255)
        ##############################################################################################
        matrix_R_decrease = iterative_image_channel_R.copy()
        matrix_R_decrease = matrix_R_decrease.astype(np.float64)
        matrix_G_decrease = iterative_image_channel_G.copy()
        matrix_G_decrease = matrix_G_decrease.astype(np.float64)
        matrix_B_decrease = iterative_image_channel_B.copy()
        matrix_B_decrease = matrix_B_decrease.astype(np.float64)

        addend = 0
        for i in range(224):
            for j in range(224):
                addend = abs(matrix_R_decrease[i][j] * ratio)
                matrix_R_decrease[i][j] = matrix_R_decrease[i][j] - addend
                np.clip(matrix_R_decrease[i][j], 0, 255)

                addend = abs(matrix_G_decrease[i][j] * ratio)
                matrix_G_decrease[i][j] = matrix_G_decrease[i][j] - addend
                np.clip(matrix_G_decrease[i][j], 0, 255)

                addend = abs(matrix_B_decrease[i][j] * ratio)
                matrix_B_decrease[i][j] = matrix_B_decrease[i][j] - addend
                np.clip(matrix_B_decrease[i][j], 0, 255)

        ########################################################################################################################
        step = 0.33
        change_matrix_R = iterative_image_channel_R.copy()
        change_matrix_G = iterative_image_channel_G.copy()
        change_matrix_B = iterative_image_channel_B.copy()

        for i in range(224):
            for j in range(224):
                ############################################################################################################
                if matrix_R_data_transform[i][j] < 0 and pixel_weight_matrix_R_iterative[i][j] < 0 and ~(
                        mark_object_image_channel_R[i][j] >= 150 and mark_object_image_channel_G[i][j] <= 50 and
                        mark_object_image_channel_B[i][j] <= 50):
                    if matrix_R_increase[i][j] > 255:
                        change_matrix_R[i][j] = 255
                    elif matrix_R_increase[i][j] <= 255:
                        change_matrix_R[i][j] = matrix_R_increase[i][j]
                    change_matrix_R[i][j] = np.clip(change_matrix_R[i][j], actual_image_channel_R[i][j] + 255 * step,
                                                    actual_image_channel_R[i][j] + 255 * step * 2).astype(np.uint8)
                    change_matrix_R[i][j] = np.clip(change_matrix_R[i][j], 0, 255).astype(np.uint8)

                if matrix_G_data_transform[i][j] < 0 and pixel_weight_matrix_G_iterative[i][j] < 0 and ~(
                        mark_object_image_channel_R[i][j] >= 150 and mark_object_image_channel_G[i][j] <= 50 and
                        mark_object_image_channel_B[i][j] <= 50):
                    if matrix_G_increase[i][j] > 255:
                        change_matrix_G[i][j] = 255
                    elif matrix_G_increase[i][j] <= 255:
                        change_matrix_G[i][j] = matrix_G_increase[i][j]
                    change_matrix_G[i][j] = np.clip(change_matrix_G[i][j], actual_image_channel_G[i][j] + 255 * step,
                                                    actual_image_channel_G[i][j] + 255 * step * 2).astype(np.uint8)
                    change_matrix_G[i][j] = np.clip(change_matrix_G[i][j], 0, 255).astype(np.uint8)

                if matrix_B_data_transform[i][j] < 0 and pixel_weight_matrix_B_iterative[i][j] < 0 and ~(
                        mark_object_image_channel_R[i][j] >= 150 and mark_object_image_channel_G[i][j] <= 50 and
                        mark_object_image_channel_B[i][j] <= 50):
                    if matrix_B_increase[i][j] > 255:
                        change_matrix_B[i][j] = 255
                    elif matrix_B_increase[i][j] <= 255:
                        change_matrix_B[i][j] = matrix_B_increase[i][j]
                    change_matrix_B[i][j] = np.clip(change_matrix_B[i][j], actual_image_channel_B[i][j] + 255 * step,
                                                    actual_image_channel_B[i][j] + 255 * step * 2).astype(np.uint8)
                    change_matrix_B[i][j] = np.clip(change_matrix_B[i][j], 0, 255).astype(np.uint8)

                #########################################################################################################
                if matrix_R_data_transform[i][j] > 0 and pixel_weight_matrix_R_iterative[i][j] > 0 and ~(
                        mark_object_image_channel_R[i][j] >= 150 and mark_object_image_channel_G[i][j] <= 50 and
                        mark_object_image_channel_B[i][j] <= 50):
                    if matrix_R_decrease[i][j] < 0:
                        change_matrix_R[i][j] = 0
                    elif matrix_R_decrease[i][j] >= 0:
                        change_matrix_R[i][j] = matrix_R_decrease[i][j]
                    change_matrix_R[i][j] = np.clip(change_matrix_R[i][j],
                                                    actual_image_channel_R[i][j] - 255 * step * 2,
                                                    actual_image_channel_R[i][j] - 255 * step).astype(np.uint8)
                    change_matrix_R[i][j] = np.clip(change_matrix_R[i][j], 0, 255).astype(np.uint8)

                if matrix_G_data_transform[i][j] > 0 and pixel_weight_matrix_G_iterative[i][j] > 0 and ~(
                        mark_object_image_channel_R[i][j] >= 150 and mark_object_image_channel_G[i][j] <= 50 and
                        mark_object_image_channel_B[i][j] <= 50):
                    if matrix_G_decrease[i][j] < 0:
                        change_matrix_G[i][j] = 0
                    elif matrix_G_decrease[i][j] >= 0:
                        change_matrix_G[i][j] = matrix_G_decrease[i][j]
                    change_matrix_G[i][j] = np.clip(change_matrix_G[i][j],
                                                    actual_image_channel_G[i][j] - 255 * step * 2,
                                                    actual_image_channel_G[i][j] - 255 * step).astype(np.uint8)
                    change_matrix_G[i][j] = np.clip(change_matrix_G[i][j], 0, 255).astype(np.uint8)

                if matrix_B_data_transform[i][j] > 0 and pixel_weight_matrix_B_iterative[i][j] > 0 and ~(
                        mark_object_image_channel_R[i][j] >= 150 and mark_object_image_channel_G[i][j] <= 50 and
                        mark_object_image_channel_B[i][j] <= 50):
                    if matrix_B_decrease[i][j] < 0:
                        change_matrix_B[i][j] = 0
                    elif matrix_B_decrease[i][j] >= 0:
                        change_matrix_B[i][j] = matrix_B_decrease[i][j]
                    change_matrix_B[i][j] = np.clip(change_matrix_B[i][j],
                                                    actual_image_channel_B[i][j] - 255 * step * 2,
                                                    actual_image_channel_B[i][j] - 255 * step).astype(np.uint8)
                    change_matrix_B[i][j] = np.clip(change_matrix_B[i][j], 0, 255).astype(np.uint8)

                ########################################################################################################
        # Combine three channels into an RGB image
        image_rgb = np.stack([change_matrix_R, change_matrix_G, change_matrix_B], axis=-1)
        # Convert data type to 8-bit unsigned integer
        image_rgb = image_rgb.astype(np.uint8)
        # Create PIL image object
        image_pil = Image.fromarray(image_rgb)
        image_pil.save("adversarial_samples_with_reduced_background_classification_contribution_value_33_66.png")

        # Create model
        model = AlexNet(num_classes=3).to(device)
        model.load_state_dict(torch.load(the_absolute_path_of_the_weight_file))

        iterative_image_path = "adversarial_samples_with_reduced_background_classification_contribution_value_33_66.png"

        iterative_image_index = predict(iterative_image_path, the_absolute_path_of_the_index_file, the_absolute_path_of_the_weight_file)

        if iterative_image_index != actual_image_index:
            print("success")
            current_image_path = "adversarial_samples_with_reduced_background_classification_contribution_value_33_66.png"
            new_image_name = str(image_file)
            new_image_path = os.path.join(save_path_for_adversarial_samples, new_image_name)
            shutil.copy(current_image_path, new_image_path)
            flag = 1

        elif num_iterative == 15:
            print("Final failure")
            fail_images.append(image_file)
            print(fail_images)
            file_path = "fail_images_33_66.txt"
            with open(file_path, 'w') as file:
                for image in fail_images:
                    file.write(f"{image}\n")
            flag = 1
        else:
            print("fail！")
            flag = 0
