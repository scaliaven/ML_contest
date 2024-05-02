import librosa
import librosa.display
import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm
# processing the data and augment it by converting 1 .mp3 file into 1 images
# use mel spectrogram

def mp3_to_image(indir, outdir, length):
    for i in tqdm(range(length)):
        y, sr = librosa.load(f'{indir}/{i}.mp3')
        melspec = librosa.feature.melspectrogram(y = y, sr=sr)
        melspec = librosa.power_to_db(melspec, ref = np.max)
        size = melspec.shape
        melspec = melspec.reshape(1, size[0], size[1])
        melspec = torch.from_numpy(melspec)
        image_path = f'{outdir}/{i}.pt'
        torch.save(melspec, image_path)

mp3_to_image("/scratch/hh3043/ML_contest/dataset/train_mp3s", "/scratch/hh3043/ML_contest/dataset/trai_gray_img", 11886)
mp3_to_image("/scratch/hh3043/ML_contest/dataset/test_mp3s", "/scratch/hh3043/ML_contest/dataset/test_gray_img", 2447)
