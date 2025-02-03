"""
The main function of this code file is to use the official ResNet model for single-image category prediction.
First, you need to set "img_absolute_path" to the absolute path of the image to be predicted.
Second, you need to set "json_absolute_path" to the absolute path of the index file containing 1000 class indices. This index file is named "output_class_indices.json" in the current directory.
Third, you need to set "weights_absolute_path" to the absolute path of the ResNet model's weight file, which can be downloaded from the address provided in the paper.
Finally, executing this code file will output the predicted category information for the image.
"""


import torchvision.models as models
import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# Set the device based on CUDA availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load ResNet model.
model = models.resnet18(pretrained=False)
model.eval()

# Preprocess the input image.
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load image
img_absolute_path = "..."
image = Image.open(img_absolute_path)
image_tensor = transform(image).unsqueeze(0)

# Read class_indict
json_absolute_path = "..."
assert os.path.exists(json_absolute_path), "file: '{}' dose not exist.".format(json_absolute_path)

# Load model weights
weights_absolute_path = "..."
assert os.path.exists(weights_absolute_path), "file: '{}' dose not exist.".format(weights_absolute_path)
model.load_state_dict(torch.load(weights_absolute_path))

with open(json_absolute_path, "r") as f:
    class_indict = json.load(f)

with torch.no_grad():
    output = torch.squeeze(model(image_tensor)).cpu()
    predict = torch.softmax(output, dim=0)
    top5_probs, top5_indices = torch.topk(predict, 10)

for i in range(3):
    class_index = top5_indices[i].item()
    class_prob = top5_probs[i].item()
    print("Top {}: index: {}  class: {:10}   Classification probability: {:.5f}   Classification value: {:.6f}".format(i + 1,class_index, class_indict[str(class_index)],
                                                                          class_prob, output[class_index].numpy()))

plt.show()

# Get specified category information,Specifically, set which category in "[]" to output,For example, specify it as class 258
print("class: {:10}   Classification probability: {:.5f}   Classification value: {:.5f}".format(class_indict[str(258)], predict[258].numpy(),
                                                                  output[258].numpy()))
