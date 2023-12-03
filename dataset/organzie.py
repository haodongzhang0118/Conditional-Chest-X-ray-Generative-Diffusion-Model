import os
import shutil
import sys
from tqdm import tqdm

def reorganize_images(base_dir):
    if not os.path.exists(base_dir):
        print(f"The directory {base_dir} doesn't exist!")
        return

    dir_list = os.listdir(base_dir)

    for fileIndex in tqdm(range(len(dir_list))):
        if dir_list[fileIndex].endswith(".jpg") or dir_list[fileIndex].endswith(".png"):

            label = dir_list[fileIndex].split('_')[0]
            label_dir = os.path.join(base_dir, label)

            if not os.path.exists(label_dir):
                os.mkdir(label_dir)

            src = os.path.join(base_dir, dir_list[fileIndex])
            dest = os.path.join(label_dir, dir_list[fileIndex])
            shutil.move(src, dest)

if __name__ == "__main__":
    base = sys.argv[1]
    reorganize_images(base)
    # for folder in os.listdir(base):
    #     for image in os.listdir(f"{base}/{folder}"):
    #         with open(f"{base}/{folder}/{image}", "rb") as f:
    #             print(f)

    #         break
    #     break

