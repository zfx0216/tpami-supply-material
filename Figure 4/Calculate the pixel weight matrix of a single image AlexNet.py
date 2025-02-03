"""
The main function of this code file is to calculate the pixel weight matrix of a single image
The first step is to set 'imd_path' as the absolute path of this image.
Step two, you need to set "json path" as the absolute path of the index file, which is located in the current folder named "output class indicators. json".
Step three, you need to set "weights_path" as the absolute path of the weight file.
Step four, you need to modify the category index number in the "pred_store" calculation to your target category index number
"""


import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision.models as models
import numpy as np


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    # Load image
    img_path = "..."
    assert os.path.exists(img_path), "file: '{}' does not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)

    # Read class_indict
    json_path = "..."
    assert os.path.exists(json_path), "file: '{}' does not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    model = models.alexnet(pretrained=True)
    model.to(device)

    img = img.to(device)
    # Load model weights
    weights_path = "..."
    assert os.path.exists(weights_path), "file: '{}' does not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path))

    model.eval()
    img.requires_grad_()
    output = model(img)

    # Target category index number
    pred_score = output[0, 208]
    pred_score.backward()
    gradients = img.grad

    pixel_weight_matrix_R = gradients[0, 0, :, :].cpu().detach().numpy()
    pixel_weight_matrix_G = gradients[0, 1, :, :].cpu().detach().numpy()
    pixel_weight_matrix_B = gradients[0, 2, :, :].cpu().detach().numpy()

    # save
    np.savetxt("the_image_pixel_weight_matrix_R.txt", pixel_weight_matrix_R, fmt="%f", delimiter=" ")
    np.savetxt("the_image_pixel_weight_matrix_G.txt", pixel_weight_matrix_G, fmt="%f", delimiter=" ")
    np.savetxt("the_image_pixel_weight_matrix_B.txt", pixel_weight_matrix_B, fmt="%f", delimiter=" ")


if __name__ == '__main__':
    main()