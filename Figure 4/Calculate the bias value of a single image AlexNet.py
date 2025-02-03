"""
The main function of this code file is to calculate the bias values of a single image in the specified category of the officially trained AlexNet model.
First, you need to set "weights_absolute_path" to the absolute path of the weights file of the officially trained AlexNet model.
Second, you need to set "image_absolute_path" to the absolute path of the image to be calculated.
Third, you need to set "level" to the index of the category for which you want to obtain the bias values.
Finally, executing this code file will give you the bias values corresponding to this image, saved in the "bias.txt" file format in the current directory.
"""


import torch
import torch.nn as nn
import torchvision.transforms as transforms
import time
from PIL import Image
import os

from torchvision.models import alexnet

b_weight = torch.zeros((1, 1000), dtype=torch.float32)
relu_map = []
maxpool_map = []
modules = []
start_time = time.time()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

alexnet = alexnet(num_classes=1000).to(device)

# Load model weights
weights_absolute_path = "..."
assert os.path.exists(weights_absolute_path), "file: '{}' not.".format(weights_absolute_path)
alexnet.load_state_dict(torch.load(weights_absolute_path, map_location=device))

# Open the image and preprocess it
image_absolute_path = "..."
image = Image.open(image_absolute_path)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)
input_batch = input_batch.to(device)

# Specifying the category of the pixel weight matrix that may be required
level = "..."


def maxpool_specify_trajectory(input, map):
    map_shape = map.shape
    input = input.view(input.shape[0], input.shape[1], -1)
    map = map.view(map.shape[0], map.shape[1], -1)
    result = torch.gather(input, dim=2, index=map)
    result = result.view(map_shape)
    return result

def relu_specify_trajectory(input, map):
    return input * map


alexnet.eval()
with torch.no_grad():

    for name, module in alexnet.features.named_children():
        modules.append(module)

        if isinstance(module, nn.ReLU):
            relumap = (input_batch > 0)
            relu_map.append(relumap.clone())

        if isinstance(module, nn.MaxPool2d):
            module.return_indices = True
            input_batch, indices = module(input_batch)
            maxpool_map.append(indices.clone())
        else:
            input_batch = module(input_batch)

    avgpool_layer = alexnet.avgpool
    input_batch = avgpool_layer(input_batch)
    modules.append(avgpool_layer)

    for name, module in alexnet.classifier.named_children():

        if isinstance(module, nn.Dropout):
            continue
        if isinstance(module, nn.ReLU):
            relumap = (input_batch >= 0)
            relu_map.append(relumap.clone())
        if isinstance(module, nn.Linear):
            input_batch = input_batch.view(input_batch.size(0), -1)
        input_batch = module(input_batch)
        modules.append(module)

    cnt = 0
    for i in range(1):
        for j in range(1):
            for k in range(1):
                for l in range(1):
                    if cnt % 500 == 0:
                        print("!!!")
                        print("Time:{}".format(int((time.time()-start_time)/60))+" mins")
                        print("!!!")
                    cnt += 1
                    img = torch.zeros((1, 3, 224, 224))
                    img[i, j, k, l] = 0
                    img = img.to(device)
                    # layer1-conv2d
                    img = modules[0](img)
                    # layer2-relu
                    img = relu_specify_trajectory(img, relu_map[0])
                    # layer3-maxpool2d
                    img = maxpool_specify_trajectory(img, maxpool_map[0])
                    # layer4-conv2d
                    img = modules[3](img)
                    # layer5-relu
                    img = relu_specify_trajectory(img, relu_map[1])
                    # layer6-maxpool2d
                    img = maxpool_specify_trajectory(img, maxpool_map[1])
                    # layer7-conv2d
                    img = modules[6](img)
                    # layer8-relu
                    img = relu_specify_trajectory(img, relu_map[2])
                    # layer9-conv2d
                    img = modules[8](img)
                    # layer10-relu
                    img = relu_specify_trajectory(img, relu_map[3])
                    # layer11-conv2d
                    img = modules[10](img)
                    # layer12-relu
                    img = relu_specify_trajectory(img, relu_map[4])
                    # layer13-maxpool2d
                    img = maxpool_specify_trajectory(img, maxpool_map[2])
                    # layer14-adaptiveAvgPool2d
                    img = modules[13](img)
                    img = img.view(img.size(0), -1)
                    # layer15-linear
                    img = modules[14](img)
                    # layer16-relu
                    img = relu_specify_trajectory(img, relu_map[5])
                    # layer17-linear
                    img = modules[16](img)
                    # layer18-relu
                    img = relu_specify_trajectory(img, relu_map[6])
                    # layer19-linear
                    img = modules[18](img)

                    b_weight[i] = img

torch.save(b_weight, "b_weight.pt")

# Load x_weight.pt
b_weight = torch.load("b_weight.pt")
biad = b_weight[0, level]
with open("bias.txt", 'w') as file:
    file.write(str(biad.item()))