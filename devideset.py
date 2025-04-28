import os
from Preprocessing import *
import librosa

original_dir = "rmslience/js"
img_name_list = os.listdir(original_dir)

train_dir = 'mfcc/train'
val_dir = 'mfcc/val'
test_dir = 'mfcc/test'

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

for i in range(len(img_name_list)):
    if i < len(img_name_list) * 0.7:
        result_dir = train_dir
    elif i < len(img_name_list) * 0.8:
        result_dir = val_dir
    else:
        result_dir = test_dir

    img_name = img_name_list[i]
    audio_path = os.path.join(original_dir, img_name)
    audio, sr = librosa.load(audio_path, sr=None)
    mfcc_features = transform_mfcc(audio)
    output_file = os.path.splitext(img_name)[0] + '.npy'
    output_path = os.path.join(result_dir, output_file)
    np.save(output_path, mfcc_features)
