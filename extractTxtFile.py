import os
from tqdm import tqdm
import shutil

# path = r"D:\MIMIC-CXR\mimic-cxr-jpg_2.0.0\data_need"
# target = r"D:\MIMIC-CXR\mimic-cxr-jpg_2.0.0\files"
path = r"C:\Users\11429\Desktop\CS 184A\Final Project\test1"
target = r"C:\Users\11429\Desktop\CS 184A\Final Project\files"
index = 0

for subfolder in os.listdir(path):
    for sub_subfolder in tqdm(os.listdir(os.path.join(path, subfolder)), desc="Processing {}: ".format(index + 1)):
        folder_path = os.path.join(subfolder, sub_subfolder)
        for sub_sub_subfolder in os.listdir(os.path.join(path, folder_path)):
                folder_path1 = os.path.join(folder_path, sub_sub_subfolder)
                path_txt = os.path.join(path, folder_path1, "label")
                target_path = os.path.join(target, folder_path1, "label")
                shutil.move(path_txt, target_path)
    index += 1