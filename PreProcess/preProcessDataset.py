import os
import shutil
from tqdm import tqdm

source_dir = r'D:\MIMIC-CXR\mimic-cxr-jpg_2.0.0\reports'  
target_dir = r'D:\MIMIC-CXR\mimic-cxr-jpg_2.0.0\images'
label = "label"
images = "images" 

index = 0

for subfolder in os.listdir(source_dir):
    # ex. D:\MIMIC-CXR\mimic-cxr-jpg_2.0.0\reports\p10
    subfolder_path = os.path.join(source_dir, subfolder)

    if os.path.isdir(subfolder_path):
        for sub_subfolder in tqdm(os.listdir(subfolder_path), desc="Processing {}: ".format(index + 1)):
            # ex. D:\MIMIC-CXR\mimic-cxr-jpg_2.0.0\reports\p10\p10000032
            sub_subfolder_path = os.path.join(subfolder_path, sub_subfolder)

            if os.path.isdir(sub_subfolder_path):
                for file in os.listdir(sub_subfolder_path):
                    name, _ = os.path.splitext(file)
                    file_path = os.path.join(sub_subfolder_path, file)

                    if os.path.isfile(file_path) and file.endswith('.txt'):
                        target_path = os.path.join(target_dir, subfolder, sub_subfolder, name)
                        target_file_path = os.path.join(target_path, label)
                        os.makedirs(target_file_path)
                        target_file_path = os.path.join(target_file_path, file)
                        shutil.move(file_path, target_file_path)

                        target_file_path = os.path.join(target_path, images)
                        os.makedirs(target_file_path)
                        for file in os.listdir(target_path):
                            if file.endswith('.jpg'):
                                shutil.move(os.path.join(target_path, file), target_file_path)
    index += 1

print("Files have been moved successfully.")