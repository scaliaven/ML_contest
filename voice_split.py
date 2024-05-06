import os
from tqdm import tqdm
from spleeter.separator import Separator

sedperator = Separator("spleeter:2stems")

input_dir = "/scratch/hh3043/ML_contest/dataset/train_mp3s"
output_dir = "/scratch/hh3043/ML_contest/separate/train"
input_files = [os.path.join(input_dir, filename) for filename in os.listdir(input_dir) if filename.endswith('.mp3')]
for input_file in tqdm(input_files):
    sedperator.separate_to_file(input_file, output_dir)

input_dir = "/scratch/hh3043/ML_contest/dataset/test_mp3s"
output_dir = "/scratch/hh3043/ML_contest/separate/test"
input_files = [os.path.join(input_dir, filename) for filename in os.listdir(input_dir) if filename.endswith('.mp3')]
for input_file in tqdm(input_files):
    sedperator.separate_to_file(input_file, output_dir)