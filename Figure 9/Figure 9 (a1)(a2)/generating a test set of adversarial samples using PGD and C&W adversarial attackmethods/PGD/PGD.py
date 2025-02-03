"""
The main function of this code file is to use PGD to attack the AlexNet model, generating a series of attack samples.
Firstly, you need to set the parameter "weights_path" to the absolute path of the weight file.
Secondly, you need to set the parameter "folder_path" to the absolute path of the folder where the original image is stored.
Thirdly, you need to set the parameter "save_path_for-adversarial_samples" to the absolute path of the folder where the generated attack samples are saved.
Fourthly, you need to set the parameter "label" to an appropriate label based on the code comments
"""

"""
The code for generating robustness test sets for the convolutional neural network models ResNet and DenseNet is very similar to this code, and you only need to modify the model.
"""

import torch
from torchvision import transforms
from PIL import Image
from model import AlexNet
import os
from PGD_CLASS import PGD


def sort_func(file_name):
    return int(''.join(filter(str.isdigit, file_name)))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create and load model
model = AlexNet(num_classes=3).to(device)
# The absolute path of the weight file
weights_path = "..."
assert os.path.exists(weights_path), "file: '{}' does not exist.".format(weights_path)
model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# The absolute path to the folder where the original images are stored
folder_path = r"..."
# Save the folder path for the generated attack samples
save_path_for_adversarial_samples = "..."

file_list = os.listdir(folder_path)
file_list = sorted(file_list, key=sort_func)

for file_name in file_list:
    print(file_name)
    image_path = os.path.join(folder_path, file_name)

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).cuda()

    # When generating adversarial samples for aimless attacks, it is necessary to set the label in "[]" to the true label of the image
    label = torch.tensor(["..."]).cuda()

    atk = PGD(model, eps=8 / 255, alpha=1 / 255, steps=10)
    adv_image = atk(image, label)

    adv_image = adv_image.squeeze(0).cpu()
    adv_image = transforms.Normalize(mean=[-2.12, -2.04, -1.80], std=[4.37, 4.46, 4.44])(adv_image)
    adv_image = transforms.ToPILImage()(adv_image)


    new_image_name = str(file_name)
    new_image_path = os.path.join(save_path_for_adversarial_samples, new_image_name)
    adv_image.save(new_image_path)



