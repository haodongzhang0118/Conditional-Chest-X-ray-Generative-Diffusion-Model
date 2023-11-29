import os
from tqdm import tqdm
import shutil

path = r"D:\MIMIC-CXR\mimic-cxr-jpg_2.0.0\data"
target = r"D:\MIMIC-CXR\mimic-cxr-jpg_2.0.0\data_need"
index = 0

for subfolder in os.listdir(path):
    for sub_subfolder in tqdm(os.listdir(os.path.join(path, subfolder)), desc="Processing {}: ".format(index + 1)):
        folder_path = os.path.join(subfolder, sub_subfolder)
        for sub_sub_subfolder in os.listdir(os.path.join(path, folder_path)):
                folder_path1 = os.path.join(folder_path, sub_sub_subfolder)
                path_img = os.path.join(path, folder_path1, "images")
                if len(os.listdir(path_img)) == 1:
                    path_source = os.path.join(path, folder_path1)
                    path_target = os.path.join(target, folder_path1)
                    shutil.move(path_source, path_target)
    index += 1