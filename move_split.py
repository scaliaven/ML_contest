import os
import shutil

def move_vocals_to_test_img(root_dir, outputdir):
    for subdir, dirs, files in os.walk(root_dir):
        for dir_name in dirs:
            vocals_path = os.path.join(subdir, dir_name, 'vocals.wav')
            if os.path.exists(vocals_path):
                destination = os.path.join(outputdir, f"{dir_name}.wav")
                shutil.move(vocals_path, destination)

root_directory = '/scratch/hh3043/ML_contest/separate/train/'
outputdir = '/scratch/hh3043/ML_contest/separate/train_mp3s'
move_vocals_to_test_img(root_directory, outputdir)

root_directory = '/scratch/hh3043/ML_contest/separate/test/'
outputdir = '/scratch/hh3043/ML_contest/separate/test_mp3s'
move_vocals_to_test_img(root_directory, outputdir)