"""
The main function of this code file is to use the trained DenseNet model to calculate the classification accuracy of a certain test set.
Firstly, you need to set the parameter "image_path" to the absolute path of the test set folder.
Secondly, you need to import the absolute path of the weight file through "torch. load ("... ")"
"""

import torch
from torch.utils.data import DataLoader
import os
from torchvision import transforms, datasets
import torchvision.models as models


# Set the device based on CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Perform initialization operations on the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Import the testing dataset
image_path = "..."
assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
# test_dataset = datasets.ImageFolder(root=os.path.join(image_path, "test"), transform=transform)
test_dataset = datasets.ImageFolder(image_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

model = models.densenet121(num_classes=3).to(device)

# Absolute path to import weight file
model.load_state_dict(torch.load("..."))
model.to(device)

# Set the model to evaluation mode
model.eval()

# Number of correctly predicted samples
correct = 0

# Total number of samples in the validation set
total = 0

# Disable gradient calculation during validation to save memory and computation time
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Compute accuracy
accuracy = 100 * correct / total
print("Accuracy: {:.2f}%".format(accuracy))