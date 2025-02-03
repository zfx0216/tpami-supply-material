"""
The main function of this code file is to calculate the bias value of a single image belonging to a specified class in the officially trained ResNet model.
First, you need to download the official ResNet weight file, the download address of which is given in the paper.
Second, you need to import the absolute path of the weight file of the officially trained ResNet in "resnet.load_state_dict(torch.load("..."))".
Third, you need to set "image_absolute_path" to the absolute path of the image to be calculated.
Fourth, you need to set "level" to the index number of the class for which you want to obtain the bias value.
Finally, execute this code file to get the bias value corresponding to this image, saved in the current directory in "bias.txt" file format.
"""


import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import torch.nn as nn

# Create a matrix of the same shape as the output, used to store the bias of each level
b_weight = torch.zeros((1, 1000), dtype=torch.float32)

# record the execution trajectory of ReLU
relu_map = []

# record the execution trajectory of max pooling
maxpool_map = []

# model that specifies the execution trajectory
modules = []

# load a pre-trained ResNet model
resnet = models.resnet18(pretrained=False)
resnet.load_state_dict(torch.load("..."))

# Set the model to evaluation mode and move it to the GPU (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet.eval()
resnet.to(device)

relu = nn.ReLU(inplace=True)

# Open the image and perform preprocessing
image_absolute_path = "..."
image = Image.open(image_absolute_path)
preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)
input_batch = input_batch.to(device)

# Specifying the category of the pixel weight matrix that may be required,For example, specify it as class 258
level = 258

# Specifying the trajectory for Maxpooling
def maxpool_specify_trajectory(input, map):
    map_shape = map.shape
    input = input.view(input.shape[0], input.shape[1], -1)
    map = map.view(map.shape[0], map.shape[1], -1)
    result = torch.gather(input, dim=2, index=map)
    result = result.view(map_shape)
    return result

# Specifying the trajectory for ReLU
def relu_specify_trajectory(input, map):
    return input * map

first_relu_found = False

block_1_0 = resnet.layer1[0]
block_1_1 = resnet.layer1[1]

block_2_0 = resnet.layer2[0]
block_2_1 = resnet.layer2[1]

block_3_0 = resnet.layer3[0]
block_3_1 = resnet.layer3[1]

block_4_0 = resnet.layer4[0]
block_4_1 = resnet.layer4[1]

resnet.eval()
with torch.no_grad():

    for name, module in resnet.named_children():
        modules.append(module)
        if isinstance(module, nn.ReLU) and not first_relu_found:
            relumap = (input_batch > 0)
            relu_map.append(relumap.clone())
            first_relu_found = True  # Set the flag to True to indicate that the first ReLU layer has been found
            input_batch = module(input_batch)

        elif isinstance(module, nn.MaxPool2d):
            module.return_indices = True
            input_batch, indices = module(input_batch)
            maxpool_map.append(indices.clone())
            break
        else:
            input_batch = module(input_batch)

    actual_value = input_batch

    for name, module in block_1_0.named_children():
        modules.append(module)
        if isinstance(module, nn.ReLU):
            relumap = (input_batch > 0)
            relu_map.append(relumap.clone())
            input_batch = module(input_batch)
        else:
            input_batch = module(input_batch)

    actual_value = input_batch + actual_value
    modules.append(nn.ReLU(inplace=True))
    relumap = (actual_value > 0)
    relu_map.append(relumap.clone())
    actual_value = relu(actual_value)

    input_batch = actual_value
    for name, module in block_1_1.named_children():
        modules.append(module)
        if isinstance(module, nn.ReLU):
            relumap = (input_batch > 0)
            relu_map.append(relumap.clone())
            input_batch = module(input_batch)
        else:
            input_batch = module(input_batch)

    actual_value = input_batch + actual_value

    modules.append(nn.ReLU(inplace=True))
    relumap = (actual_value > 0)
    relu_map.append(relumap.clone())
    actual_value = relu(actual_value)
##############################################################################################
    input_batch = actual_value
    for name, module in block_2_0.named_children():
        modules.append(module)
        if isinstance(module, nn.ReLU):
            relumap = (input_batch > 0)
            relu_map.append(relumap.clone())
            input_batch = module(input_batch)
        if isinstance(module, nn.Sequential) and len(module) == 2:
            downsample_conv, downsample_bn = module
            actual_value = downsample_bn(downsample_conv(actual_value))
        else:
            input_batch = module(input_batch)
    actual_value = input_batch + actual_value

    modules.append(nn.ReLU(inplace=True))
    relumap = (actual_value > 0)
    relu_map.append(relumap.clone())
    actual_value = relu(actual_value)

    input_batch = actual_value
    for name, module in block_2_1.named_children():
        modules.append(module)
        if isinstance(module, nn.ReLU):
            relumap = (input_batch > 0)
            relu_map.append(relumap.clone())
            input_batch = module(input_batch)
        else:
            input_batch = module(input_batch)

    actual_value = actual_value + input_batch

    modules.append(nn.ReLU(inplace=True))
    relumap = (actual_value > 0)
    relu_map.append(relumap.clone())
    actual_value = relu(actual_value)

    ##############################################################################################
    input_batch = actual_value
    for name, module in block_3_0.named_children():
        modules.append(module)
        if isinstance(module, nn.ReLU):
            relumap = (input_batch > 0)
            relu_map.append(relumap.clone())
            input_batch = module(input_batch)
        if isinstance(module, nn.Sequential) and len(module) == 2:
            downsample_conv, downsample_bn = module
            actual_value = downsample_bn(downsample_conv(actual_value))
        else:
            input_batch = module(input_batch)
    actual_value = input_batch + actual_value

    modules.append(nn.ReLU(inplace=True))
    relumap = (actual_value > 0)
    relu_map.append(relumap.clone())
    actual_value = relu(actual_value)

    input_batch = actual_value
    for name, module in block_3_1.named_children():
        modules.append(module)
        if isinstance(module, nn.ReLU):
            relumap = (input_batch > 0)
            relu_map.append(relumap.clone())
            input_batch = module(input_batch)
        else:
            input_batch = module(input_batch)

    actual_value = actual_value + input_batch

    modules.append(nn.ReLU(inplace=True))
    relumap = (actual_value > 0)
    relu_map.append(relumap.clone())
    actual_value = relu(actual_value)

    ##############################################################################################
    input_batch = actual_value
    for name, module in block_4_0.named_children():
        modules.append(module)
        if isinstance(module, nn.ReLU):
            relumap = (input_batch > 0)
            relu_map.append(relumap.clone())
            input_batch = module(input_batch)
        if isinstance(module, nn.Sequential) and len(module) == 2:
            downsample_conv, downsample_bn = module
            actual_value = downsample_bn(downsample_conv(actual_value))
        else:
            input_batch = module(input_batch)
    actual_value = input_batch + actual_value

    modules.append(nn.ReLU(inplace=True))
    relumap = (actual_value > 0)
    relu_map.append(relumap.clone())
    actual_value = relu(actual_value)

    input_batch = actual_value
    for name, module in block_4_1.named_children():
        modules.append(module)
        if isinstance(module, nn.ReLU):
            relumap = (input_batch > 0)
            relu_map.append(relumap.clone())
            input_batch = module(input_batch)
        else:
            input_batch = module(input_batch)

    actual_value = actual_value + input_batch

    modules.append(nn.ReLU(inplace=True))
    relumap = (actual_value > 0)
    relu_map.append(relumap.clone())
    actual_value = relu(actual_value)

    avgpool_layer = resnet.avgpool
    actual_value = avgpool_layer(actual_value)
    modules.append(avgpool_layer)

    actual_value = actual_value.view(actual_value.size(0), -1)
    fc = resnet.fc
    actual_value = fc(actual_value)
    modules.append(fc)

    ##############################################################################################
    cnt = 0
    for i in range(1):
        for j in range(1):
            for k in range(1):
                for l in range(1):
                    if cnt % 500 == 0:
                        print("!!!")
                    cnt += 1
                    img = torch.zeros((1, 3, 224, 224))
                    img[i, j, k, l] = 0
                    img = img.to(device)

                    # Execute the model with planned trajectories
                    # the first convolution
                    img = modules[0](img)
                    # the first bn1
                    img = modules[1](img)
                    # the first ReLU
                    img = relu_specify_trajectory(img, relu_map[0])
                    # the first maxpooling
                    img = maxpool_specify_trajectory(img, maxpool_map[0])

                    ##############################################################################################
                    # the first residual block
                    temp = img
                    # layer1_0_convolution1
                    img = modules[4](img)
                    # layer1_0_bn1
                    img = modules[5](img)
                    # layer1_0_ReLU
                    img = relu_specify_trajectory(img, relu_map[1])
                    # layer1_0_convolution2
                    img = modules[7](img)
                    # layer1_0_bn2
                    img = modules[8](img)
                    # simulate the addition operation in a residual block
                    img = img + temp
                    # ReLU
                    img = relu_specify_trajectory(img, relu_map[2])

                    temp = img
                    # layer1_1_convolution1
                    img = modules[10](img)
                    # layer1_1_bn1
                    img = modules[11](img)
                    # layer1_1_ReLU
                    img = relu_specify_trajectory(img, relu_map[3])
                    # layer1_1_convolution2
                    img = modules[13](img)
                    # layer1_1_bn2
                    img = modules[14](img)
                    # simulate the addition operation in a residual block
                    img = img + temp
                    # ReLU
                    img = relu_specify_trajectory(img, relu_map[4])

                    ##############################################################################################
                    # the second residual block
                    temp = img
                    # layer2_0_convolution1
                    img = modules[16](img)
                    # layer2_0_bn1
                    img = modules[17](img)
                    # layer2_0_ReLU
                    img = relu_specify_trajectory(img, relu_map[5])
                    # layer2_0_convolution2
                    img = modules[19](img)
                    # layer2_0_bn2
                    img = modules[20](img)
                    # downsample the initial input
                    downsample_conv, downsample_bn = modules[21]
                    temp = downsample_bn(downsample_conv(temp))
                    # simulate the addition operation in a residual block
                    img = img + temp
                    # ReLU
                    img = relu_specify_trajectory(img, relu_map[6])

                    temp = img
                    # layer2_1_convolution1
                    img = modules[23](img)
                    # layer2_1_bn1
                    img = modules[24](img)
                    # layer2_1_ReLU
                    img = relu_specify_trajectory(img, relu_map[7])
                    # layer2_1_convolution2
                    img = modules[26](img)
                    # layer2_1_bn2
                    img = modules[27](img)
                    # simulate the addition operation in a residual block
                    img = img + temp
                    # ReLU
                    img = relu_specify_trajectory(img, relu_map[8])

                    ##############################################################################################
                    # the third residual block
                    temp = img
                    # layer3_0_convolution1
                    img = modules[29](img)
                    # layer3_0_bn1
                    img = modules[30](img)
                    # layer3_0_ReLU
                    img = relu_specify_trajectory(img, relu_map[9])
                    # layer3_0_convolution2
                    img = modules[32](img)
                    # layer3_0_bn2
                    img = modules[33](img)
                    # downsample the initial input
                    downsample_conv, downsample_bn = modules[34]
                    temp = downsample_bn(downsample_conv(temp))
                    # simulate the addition operation in a residual block
                    img = img + temp
                    # ReLU
                    img = relu_specify_trajectory(img, relu_map[10])

                    temp = img
                    # layer3_1_convolution1
                    img = modules[36](img)
                    # layer3_1_bn1
                    img = modules[37](img)
                    # layer3_1_ReLU
                    img = relu_specify_trajectory(img, relu_map[11])
                    # layer3_1_convolution2
                    img = modules[39](img)
                    # layer3_1_bn2
                    img = modules[40](img)
                    # simulate the addition operation in a residual block
                    img = img + temp
                    # ReLU
                    img = relu_specify_trajectory(img, relu_map[12])

                    ##############################################################################################
                    # the fourth residual block
                    temp = img
                    # layer4_0_convolution1
                    img = modules[42](img)
                    # layer4_0_bn1
                    img = modules[43](img)
                    # layer4_0_ReLU
                    img = relu_specify_trajectory(img, relu_map[13])
                    # layer4_0_convolution2
                    img = modules[45](img)
                    # layer4_0_bn2
                    img = modules[46](img)
                    # downsample the initial input
                    downsample_conv, downsample_bn = modules[47]
                    temp = downsample_bn(downsample_conv(temp))
                    # simulate the addition operation in a residual block
                    img = img + temp
                    # ReLU
                    img = relu_specify_trajectory(img, relu_map[14])

                    temp = img
                    # layer4_1_convolution1
                    img = modules[49](img)
                    # layer4_1_bn1
                    img = modules[50](img)
                    # layer4_1_ReLU
                    img = relu_specify_trajectory(img, relu_map[15])
                    # layer4_1_convolution2
                    img = modules[52](img)
                    # layer4_1_bn2
                    img = modules[53](img)
                    # simulate the addition operation in a residual block
                    img = img + temp
                    # ReLU
                    img = relu_specify_trajectory(img, relu_map[16])

                    ##############################################################################################
                    # avgpool
                    img = modules[55](img)
                    # reshape dimensions
                    img = img.view(img.size(0), -1)
                    # fully connected layer
                    img = modules[56](img)

                    b_weight[i] = img

print(b_weight.shape)
torch.save(b_weight, "b_weight.pt")

# Load x_weight.pt
b_weight = torch.load("b_weight.pt")
biad = b_weight[0, level]
with open("bias.txt", 'w') as file:
    file.write(str(biad.item()))