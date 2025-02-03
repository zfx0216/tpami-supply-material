"""
This code file is mainly used to set training labels for the generated adversarial samples
Firstly, you need to import the absolute path of the folder where the images are stored.
Secondly, you need to set the starting number of the labels.
Thirdly, you need to set the label category
"""


import os

# Loading images
folder_path = r"..."

# The starting sequence number of the label
start_number = "..."

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    # To construct a label category, you need to set "1" or "2" here to indicate which branch you are labeling
    new_filename = f"1_{start_number}.png"
    new_file_path = os.path.join(folder_path, new_filename)
    os.rename(file_path, new_file_path)
    print(f"File {filename} has been renamed to {new_filename}")
    start_number += 1
