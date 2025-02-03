"""
The main function of this code file is to sort the images stored in the folder and rename each image according to its serial number
Firstly, you need to import the absolute path of the folder where the images are stored.
Secondly, you need to set a starting value for the number, such as "1"
"""


import os


def sort_func(file_name):
    return int(''.join(filter(str.isdigit, file_name)))

def rename_images(folder_path, start_index):
    file_list = os.listdir(folder_path)
    file_list = sorted(file_list, key=sort_func)

    temp_names = []
    for i, file_name in enumerate(file_list):
        file_extension = os.path.splitext(file_name)[1]
        temp_name = f"temp_{i}{file_extension}"
        os.rename(os.path.join(folder_path, file_name), os.path.join(folder_path, temp_name))
        temp_names.append(temp_name)

    for i, temp_name in enumerate(temp_names):
        file_extension = os.path.splitext(temp_name)[1]
        new_file_name = f"{start_index + i}{file_extension}"
        os.rename(os.path.join(folder_path, temp_name), os.path.join(folder_path, new_file_name))
        print(f"Renamed {temp_name} to {new_file_name}")


# The absolute path to the folder where imported images are stored
images_path = r"..."

# Set the starting number value for sorting
start_index = "..."

rename_images(images_path, start_index)