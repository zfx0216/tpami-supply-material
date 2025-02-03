"""
The main function of this code file is to select a specified number of adversarial samples
Firstly, you need to set the "source_folder_path" parameter to the absolute path of the folder where the original adversarial samples are stored.
Secondly, you need to set the "target_folder_path" parameter to the absolute path of the folder where the adversarial samples you have selected are stored.
Thirdly, you need to set the "num images to copy" parameter to the number of adversarial samples you want to select
"""


import os
import random
import shutil

# Source folder path
source_folder_path = r"..."
# Target folder path
target_folder_path = r"..."
# Number of images to be copied
num_images_to_copy = "..."

os.makedirs(target_folder_path, exist_ok=True)

file_list = os.listdir(source_folder_path)

selected_files = random.sample(file_list, num_images_to_copy)

for file_name in selected_files:
    src_file_path = os.path.join(source_folder_path, file_name)
    dst_file_path = os.path.join(target_folder_path, file_name)
    shutil.copy(src_file_path, dst_file_path)

print(f"Successfully copied {num_images_to_copy} images.")
