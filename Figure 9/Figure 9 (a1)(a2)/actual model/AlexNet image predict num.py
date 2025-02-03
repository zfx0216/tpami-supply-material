"""
The main function of this code file is to use the trained AlexNet model to batch classify and predict multiple images, and you can adjust the code to count the specific number of images and classification information for each category.
Firstly, you need to set the parameter "folder_path" to the absolute path of the folder where the classified images are stored.
Secondly, you need to set the parameter "json_path" to the absolute path of the index file.
Thirdly, you need to set the parameter "weights.path" to the absolute path of the weight file.
Fourthly, you can choose to count the classification information of a certain class by setting the label value of "class_indict"
"""


import os
import json
import torch
from PIL import Image
from torchvision import transforms
from model import AlexNet

def sort_func(file_name):
    return int(''.join(filter(str.isdigit, file_name)))

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # The absolute path to the folder where images are stored
    folder_path = r"..."

    # The absolute path of the index file
    json_path = "..."
    assert os.path.exists(json_path), "file: '{}' does not exist.".format(json_path)
    with open(json_path, "r") as f:
        class_indict = json.load(f)

    model = AlexNet(num_classes=3).to(device)

    # The absolute path of the weight file
    weights_path = "..."
    assert os.path.exists(weights_path), "file: '{}' does not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path))

    model.eval()

    file_list = os.listdir(folder_path)
    file_list = sorted(file_list, key=sort_func)

    num = 0
    name = []
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)

        img = Image.open(file_path)

        # Check if the image is single-channel
        if len(img.getbands()) == 1:
            print(f"Warning: The image {file_name} is a single-channel image. Skipping.")
            continue

        img = data_transform(img)
        img = torch.unsqueeze(img, dim=0)

        with torch.no_grad():
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            class_id = torch.argmax(predict).item()
            class_name = class_indict[str(class_id)]

        top_prob, top_index = torch.max(predict, 0)

        class_prob = top_prob.item()
        class_value = output[class_id].item()
        print("File: {}   Top 1: index: {}  class: {:10}    Classification value: {:.10f}  Classification probability: {:.10f}".format(file_name,
                                                                                                class_id,
                                                                                                class_name,
                                                                                                class_value,
                                                                                                class_prob))

        if class_id == 0:
            num = num + 1

    print(num)
    print(name)

if __name__ == '__main__':
    main()
