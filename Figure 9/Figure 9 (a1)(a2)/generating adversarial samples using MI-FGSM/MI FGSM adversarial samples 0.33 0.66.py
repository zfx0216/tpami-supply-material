"""
The main function of this code file is to generate adversarial samples of the original image based on MI-FGSM.
Firstly, you need to set the "weighted path" parameter to the absolute path of the AlexNet model's weight file.
Secondly, you need to set the "folder_path" parameter to the absolute path of the folder where the original image is stored.
Thirdly, you need to set the "save_path_for-adversarial_samples" parameter to the absolute path of the folder where the generated adversarial samples are stored.
Fourthly, you need to set the category label of "label" according to the prompts in the code comments
"""


import torch
from torchvision import transforms
from PIL import Image
from model import AlexNet
import os
from MI_FGSM_CLASS import MIFGSM
import json
import numpy as np

def sort_func(file_name):
    return int(''.join(filter(str.isdigit, file_name)))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ratio = 0.33
# Create and load model
model = AlexNet(num_classes=3).to(device)

# Absolute path to import AlexNet weight file
weights_path = "..."
assert os.path.exists(weights_path), "file: '{}' does not exist.".format(weights_path)
model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# The absolute path to store the original image folder
folder_path = r"..."
# Save the folder path for the generated adversarial samples
save_path_for_adversarial_samples = "..."

file_list = os.listdir(folder_path)
file_list = sorted(file_list, key=sort_func)

for file_name in file_list:
    print(file_name)
    image_path = os.path.join(folder_path, file_name)

    image_actual = Image.open(image_path)
    image_actual = image_actual.resize((224, 224))
    image_actual = np.array(image_actual)

    actual_image_channel_R = image_actual[:, :, 0]
    actual_image_channel_R = actual_image_channel_R.astype(np.float64)
    actual_image_channel_G = image_actual[:, :, 1]
    actual_image_channel_G = actual_image_channel_G.astype(np.float64)
    actual_image_channel_B = image_actual[:, :, 2]
    actual_image_channel_B = actual_image_channel_B.astype(np.float64)

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).cuda()

    # # When generating adversarial samples, it is a no target attack, so the true label of the image needs to be set in "[]" here
    label = torch.tensor([0]).cuda()

    atk = MIFGSM(model, eps=8 / 255, decay=1, steps=10, alpha=2 / 255)
    adv_image = atk(image, label)

    adv_image = adv_image.squeeze(0).cpu()
    adv_image = transforms.Normalize(mean=[-2.12, -2.04, -1.80], std=[4.37, 4.46, 4.44])(adv_image)
    adv_image = transforms.ToPILImage()(adv_image)

    new_image_name = str(file_name)
    new_image_path = os.path.join(save_path_for_adversarial_samples, new_image_name)
    adv_image.save(new_image_path)

    adv_image = Image.open(new_image_path)
    adv_image = adv_image.resize((224, 224))
    adv_image = np.array(adv_image)
    adv_image_channel_R = adv_image[:, :, 0]
    adv_image_channel_R = adv_image_channel_R.astype(np.float64)
    adv_image_channel_G = adv_image[:, :, 1]
    adv_image_channel_G = adv_image_channel_G.astype(np.float64)
    adv_image_channel_B = adv_image[:, :, 2]
    adv_image_channel_B = adv_image_channel_B.astype(np.float64)

    for i in range(224):
        for j in range(224):
            if adv_image_channel_R[i][j] > actual_image_channel_R[i][j]:
                adv_image_channel_R[i][j] = np.clip(adv_image_channel_R[i][j],
                                                    actual_image_channel_R[i][j] + 255 * ratio,
                                                    actual_image_channel_R[i][j] + 255 * ratio * 2).astype(np.uint8)
            else:
                adv_image_channel_R[i][j] = np.clip(adv_image_channel_R[i][j],
                                                    actual_image_channel_R[i][j] - 255 * ratio * 2,
                                                    actual_image_channel_R[i][j] - 255 * ratio).astype(np.uint8)

            if adv_image_channel_G[i][j] > actual_image_channel_G[i][j]:
                adv_image_channel_G[i][j] = np.clip(adv_image_channel_G[i][j],
                                                    actual_image_channel_G[i][j] + 255 * ratio,
                                                    actual_image_channel_G[i][j] + 255 * ratio * 2).astype(np.uint8)
            else:
                adv_image_channel_G[i][j] = np.clip(adv_image_channel_G[i][j],
                                                    actual_image_channel_G[i][j] - 255 * ratio * 2,
                                                    actual_image_channel_G[i][j] - 255 * ratio).astype(np.uint8)

            if adv_image_channel_B[i][j] > actual_image_channel_B[i][j]:
                adv_image_channel_B[i][j] = np.clip(adv_image_channel_B[i][j],
                                                    actual_image_channel_B[i][j] + 255 * ratio,
                                                    actual_image_channel_B[i][j] + 255 * ratio * 2).astype(np.uint8)
            else:
                adv_image_channel_B[i][j] = np.clip(adv_image_channel_B[i][j],
                                                    actual_image_channel_B[i][j] - 255 * ratio * 2,
                                                    actual_image_channel_B[i][j] - 255 * ratio).astype(np.uint8)

            adv_image_channel_R[i][j] = np.clip(adv_image_channel_R[i][j], 0, 255).astype(np.uint8)
            adv_image_channel_G[i][j] = np.clip(adv_image_channel_G[i][j], 0, 255).astype(np.uint8)
            adv_image_channel_B[i][j] = np.clip(adv_image_channel_B[i][j], 0, 255).astype(np.uint8)

    # Combine three channels into an RGB image
    image_rgb = np.stack([adv_image_channel_R, adv_image_channel_G, adv_image_channel_B], axis=-1)
    # Convert data type to 8-bit unsigned integer
    image_rgb = image_rgb.astype(np.uint8)
    # Create PIL image object
    image_pil = Image.fromarray(image_rgb)
    image_pil.save(new_image_path)




