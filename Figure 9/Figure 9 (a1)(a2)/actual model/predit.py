"""
The main function of this code file is to use the AlexNet model to classify and predict individual images.
Firstly, you need to set the parameter "img_path" to the absolute path of the predicted image.
Secondly, you need to set the parameter "weights_path" to the absolute path of the weight file
"""


import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from model import AlexNet


def main():
    # Set the device based on CUDA availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Perform initialization operations on the images
    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Load image
    img_path = "..."
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)

    # Read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # Create model
    model = AlexNet(num_classes=3).to(device)

    # Load model weights
    weights_path = "..."
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path))

    # Set the model to evaluation mode
    model.eval()
    with torch.no_grad():
        # Predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)

    print("class: {:10}   Classification probability: {:.3f}   Classification value: {:.3f}".format(class_indict[str(0)], predict[0].numpy(),
                                                                      output[0].numpy()))
    print("class: {:10}   Classification probability: {:.3f}   Classification value: {:.3f}".format(class_indict[str(1)], predict[1].numpy(),
                                                                      output[1].numpy()))
    print("class: {:10}   Classification probability: {:.3f}   Classification value: {:.3f}".format(class_indict[str(2)], predict[2].numpy(),
                                                                      output[2].numpy()))

    # Displaying the image currently undergoing probability prediction
    plt.show()


if __name__ == '__main__':
    main()
