import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
# processing the data and augment it by converting 1 .mp3 file into 1 images
# use mel spectrogram

def mp3_to_image(indir, outdir, length):
    for i in tqdm(range(length)):
        y, sr = librosa.load(f'{indir}/{i}.mp3')
        fig, ax = plt.subplots(figsize = (10, 10), dpi = 100)
        melspec = librosa.feature.melspectrogram(y = y, sr=sr)
        melspec = librosa.power_to_db(melspec, ref = np.max)
        img = librosa.display.specshow(melspec) 
        ax.axis('off')
        image_path = f'{outdir}/{i}.png'
        plt.savefig(image_path, transparent = True)
        plt.close(fig)

mp3_to_image("/scratch/hh3043/ML_contest/dataset/train_mp3s", "/scratch/hh3043/ML_contest/dataset/train_img", 11886)
mp3_to_image("/scratch/hh3043/ML_contest/dataset/test_mp3s", "/scratch/hh3043/ML_contest/dataset/test_img", 2447)
