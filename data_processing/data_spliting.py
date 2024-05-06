import os
import random
import shutil
import random
random.seed(42)

def split_folder_randomly(source_folder, dest_folder1, dest_folder2, split_ratio=0.8):
    os.makedirs(dest_folder1, exist_ok=True)
    os.makedirs(dest_folder2, exist_ok=True)
    files = os.listdir(source_folder)
    random.shuffle(files)
    num_files_dest1 = int(len(files) * split_ratio)
    files_dest1 = files[:num_files_dest1]
    files_dest2 = files[num_files_dest1:]
    for file in files_dest1:
        shutil.move(os.path.join(source_folder, file), dest_folder1)
    for file in files_dest2:
        shutil.move(os.path.join(source_folder, file), dest_folder2)


source_folder = "/scratch/hh3043/ML_contest/separate/trai_img"
dest_folder1 = "/scratch/hh3043/ML_contest/separate/train_img"
dest_folder2 = "/scratch/hh3043/ML_contest/separate/val_img"
split_ratio = 0.8

split_folder_randomly(source_folder, dest_folder1, dest_folder2, split_ratio)