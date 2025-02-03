"""
The main function of this code file is to train an AlexNet model.
Firstly, you need to set the parameter "the-absolute path of the training set" to the absolute path of the training set.
Secondly, you need to set the parameter "best_model path" to the path and name of the last saved weight file
"""


import os
import sys
import json
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm
from model import AlexNet


def calculate_accuracy(output, target):
    _, predicted = torch.max(output, 1)
    correct = (predicted == target).sum().item()
    return correct


def main():
    # Set the device based on CUDA availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # Perform initialization operations on the images in the training and validation sets
    data_transform = {
        "train": transforms.Compose([transforms.Resize(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.485, 0.456, 0.406),
                                                          (0.229, 0.224, 0.225))])
    }

    # Import the training set
    the_absolute_path_of_the_training_set = r"..."
    assert os.path.exists(the_absolute_path_of_the_training_set), "{} path does not exist.".format(the_absolute_path_of_the_training_set)
    train_dataset = datasets.ImageFolder(root=the_absolute_path_of_the_training_set,
                                         transform=data_transform["train"])
    train_num = len(train_dataset)
    print("using {} images for training.".format(train_num))

    # Convert the class labels and index positions of the training dataset into a dictionary
    classes_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in classes_list.items())
    # Write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    # Initialize the AlexNet model
    net = AlexNet(num_classes=len(classes_list)).to(device)

    # Define the loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    epochs = 40
    best_accuracy = 0.0
    best_model_path = '...'   # Naming of weight files for training

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct_predictions += calculate_accuracy(outputs, labels.to(device))
            total_samples += labels.size(0)
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, running_loss / (step + 1))

        epoch_accuracy = correct_predictions / total_samples
        print("Epoch [{}/{}] Accuracy: {:.3f}".format(epoch + 1, epochs, epoch_accuracy))

        # Check if this epoch has the best accuracy so far
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            torch.save(net.state_dict(), best_model_path)
            print(f"New best model saved with accuracy: {best_accuracy:.3f}")

    print('Finished Training')
    print(f'Best model saved with accuracy: {best_accuracy:.3f}')


if __name__ == '__main__':
    main()

